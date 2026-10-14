/**
 * metadata-coverage.test.ts - Store-level metadata coverage: one key's eligible
 * documents, covered documents, and per-type counts, checked against
 * listMetadata's exact-key report with every window removed, plus the SDK
 * boundary and the one-statement read.
 */

import { describe, test, expect, beforeAll, afterAll, beforeEach, afterEach } from "vitest";
import { mkdtemp, rm, writeFile, mkdir } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { createStore, insertContent, insertDocument, hashContent, type Store } from "../src/store.js";
import {
  countDocumentsPendingMetadata,
  getMetadataCoverage,
  listMetadata,
  MetadataBindingBudgetError,
  MetadataOptionError,
  replaceDocumentMetadata,
  type MetadataCoverageOptions,
  type MetadataCoverageResult,
} from "../src/metadata-store.js";
import type { Database, SQLiteValue } from "../src/db.js";
import { MetadataFilterError, parseMetadataFilter, type MetadataFilter } from "../src/metadata-filter.js";
import { METADATA_EXTRACTION_VERSION, type DocumentMetadata } from "../src/metadata.js";
import { createStore as createQmdStore, type QMDStore } from "../src/index.js";

let testDir: string;
let store: Store;

beforeAll(async () => {
  testDir = await mkdtemp(join(tmpdir(), "qmd-metadata-coverage-"));
});

afterAll(async () => {
  await rm(testDir, { recursive: true, force: true });
});

beforeEach(() => {
  store = createStore(join(testDir, `test-${Date.now()}-${Math.random().toString(36).slice(2)}.sqlite`));
});

afterEach(() => {
  store.close();
});

let documentCounter = 0;

/** Insert an active document with extracted metadata. */
async function insertMetadataDoc(collection: string, metadata: DocumentMetadata): Promise<number> {
  documentCounter += 1;
  const path = `doc-${documentCounter}.md`;
  const content = `# doc ${documentCounter}\n\nbody of doc ${documentCounter}\n`;
  const now = new Date().toISOString();
  const hash = await hashContent(content);
  insertContent(store.db, hash, content, now);
  const documentId = insertDocument(store.db, collection, path, path, hash, now, now);
  replaceDocumentMetadata(store.db, documentId, { metadata, extractionVersion: METADATA_EXTRACTION_VERSION });
  return documentId;
}

/**
 * listMetadata's report for exactly this key, every window removed and
 * minCount 1, in coverage's shape. listMetadata's `documents` counts every
 * active document in scope. The eligible denominator is its
 * `filteredDocuments` when a filter is given, and otherwise the active
 * documents less those pending a current, error-free extraction.
 */
function discoveryCoverage(db: Database, options: MetadataCoverageOptions): MetadataCoverageResult {
  const report = listMetadata(db, {
    collection: options.collection,
    filter: options.filter,
    match: { field: "key", operator: "eq", value: options.key },
    keyLimit: Infinity,
    valueLimit: Infinity,
  });
  const collectionNames = options.collection === undefined ? undefined : [options.collection].flat();
  const summary = report.keys.find(candidate => candidate.key === options.key);
  return {
    key: options.key,
    eligibleDocuments: options.filter ? report.filteredDocuments! : report.documents - countDocumentsPendingMetadata(db, collectionNames),
    coveredDocuments: summary?.documents ?? 0,
    types: (summary?.types ?? []).map(({ type, documents, distinctValues, multiValued }) => ({ type, documents, distinctValues, multiValued })),
  };
}

/** Coverage for these options, after checking it against the discovery translation. */
function coverage(options: MetadataCoverageOptions): MetadataCoverageResult {
  const result = getMetadataCoverage(store.db, options);
  expect(result).toEqual(discoveryCoverage(store.db, options));
  return result;
}

/** The database with every executed statement's SQL recorded, and whether a transaction held it, plus every prepared statement's SQL in `prepares`. */
function observeExecutions(db: Database, executions: { sql: string; inTransaction: boolean }[], prepares?: string[]): Database {
  return new Proxy(db, {
    get(target, property) {
      if (property === "prepare") {
        return (sql: string) => {
          prepares?.push(sql);
          const statement = target.prepare(sql);
          return new Proxy(statement, {
            get(prepared, member) {
              const value = prepared[member as keyof typeof prepared];
              if (typeof value !== "function") return value;
              if (member === "get" || member === "all" || member === "run" || member === "iterate") {
                return (...params: SQLiteValue[]) => {
                  executions.push({ sql, inTransaction: target.inTransaction });
                  return (value as (...args: SQLiteValue[]) => unknown).apply(prepared, params);
                };
              }
              return value.bind(prepared);
            },
          });
        };
      }
      // Bun's Database keeps private fields, so its methods must run against the real instance.
      const member = target[property as keyof Database];
      return typeof member === "function" ? member.bind(target) : member;
    },
  });
}

function planOf(sql: string): string[] {
  const placeholders = Array.from(sql.matchAll(/\?/g), () => null);
  const rows = store.db.prepare(`EXPLAIN QUERY PLAN ${sql}`).all(...placeholders) as { detail: string }[];
  return rows.map(row => row.detail);
}

describe("getMetadataCoverage", () => {
  beforeEach(async () => {
    await insertMetadataDoc("notes", { priority: 3, team: "core", tags: ["a", "b"], owner: ["ana"] });
    await insertMetadataDoc("notes", { priority: "high", team: "core", tags: ["b"], owner: "bo" });
    await insertMetadataDoc("notes", { priority: true, team: "docs" });
    await insertMetadataDoc("notes", { priority: 3, team: "docs" });
    await insertMetadataDoc("notes", {});
    // Precomposed é, decomposed e + U+0301, e, and E: four values under BINARY.
    await insertMetadataDoc("work", { priority: 1, team: "core", tags: ["é", "é", "e", "E"] });
    await insertMetadataDoc("work", { priority: "1", team: "ops" });
    await insertMetadataDoc("work", { priority: false, team: "ops" });
    await insertMetadataDoc("other", { priority: 7 });

    // Filtering cannot see these; each would add priority 9 and team core.
    const inactive = await insertMetadataDoc("notes", { priority: 9, team: "core" });
    const pending = await insertMetadataDoc("notes", { priority: 9, team: "core" });
    const stale = await insertMetadataDoc("work", { priority: 9, team: "core" });
    const errored = await insertMetadataDoc("work", { priority: 9, team: "core" });
    store.db.prepare(`UPDATE documents SET active = 0 WHERE id = ?`).run(inactive);
    store.db.prepare(`DELETE FROM document_metadata WHERE document_id = ?`).run(pending);
    store.db.prepare(`UPDATE document_metadata SET extraction_version = ? WHERE document_id = ?`).run(METADATA_EXTRACTION_VERSION - 1, stale);
    store.db.prepare(`UPDATE document_metadata SET extraction_error = 'boom' WHERE document_id = ?`).run(errored);
  });

  test("partitions covered documents by stored type, in discovery's order, inside the extraction gate", () => {
    const result = coverage({ key: "priority" });

    // Nine eligible documents: the empty-metadata one stays eligible; inactive, pending, stale and errored do not.
    expect(result).toEqual({
      key: "priority",
      eligibleDocuments: 9,
      coveredDocuments: 8,
      types: [
        // Number 1 and boolean true share an integer encoding and stay in separate types; so do 1 and "1".
        { type: "number", documents: 4, distinctValues: 3, multiValued: false },
        { type: "boolean", documents: 2, distinctValues: 2, multiValued: false },
        { type: "string", documents: 2, distinctValues: 2, multiValued: false },
      ],
    });
    expect(result.types.reduce((sum, type) => sum + type.documents, 0)).toBe(result.coveredDocuments);
  });

  test("counts documents and values apart across repeated and multi-entry arrays", () => {
    // Seven value rows: two documents hold several entries and b repeats across two documents' arrays.
    // Extraction removes repeats inside one array (normalizeMetadataArray), so a repeated value spans documents.
    expect(coverage({ key: "tags" })).toEqual({
      key: "tags",
      eligibleDocuments: 9,
      coveredDocuments: 3,
      types: [{ type: "string", documents: 3, distinctValues: 6, multiValued: true }],
    });
    expect(coverage({ key: "team" }).types).toEqual([{ type: "string", documents: 7, distinctValues: 3, multiValued: false }]);
  });

  test("a one-element array counts as a scalar", () => {
    expect(coverage({ key: "owner" }).types).toEqual([{ type: "string", documents: 2, distinctValues: 2, multiValued: false }]);
  });

  test("a missing key keeps its eligible documents and covers none", () => {
    expect(coverage({ key: "missing" })).toEqual({ key: "missing", eligibleDocuments: 9, coveredDocuments: 0, types: [] });
  });

  test("scopes to one collection, a collection list, or an empty scope", () => {
    expect(coverage({ key: "priority", collection: "work" })).toEqual({
      key: "priority",
      eligibleDocuments: 3,
      coveredDocuments: 3,
      types: [
        { type: "boolean", documents: 1, distinctValues: 1, multiValued: false },
        { type: "number", documents: 1, distinctValues: 1, multiValued: false },
        { type: "string", documents: 1, distinctValues: 1, multiValued: false },
      ],
    });
    expect(coverage({ key: "priority", collection: ["notes", "work"] })).toMatchObject({ eligibleDocuments: 8, coveredDocuments: 7 });
    expect(coverage({ key: "priority", collection: "missing" })).toEqual({ key: "priority", eligibleDocuments: 0, coveredDocuments: 0, types: [] });
  });

  test("counts only documents a filter admits, through nested typed conditions", () => {
    expect(coverage({ key: "priority", filter: { field: "team", operator: "eq", value: "core" } })).toEqual({
      key: "priority",
      eligibleDocuments: 3,
      coveredDocuments: 3,
      types: [
        { type: "number", documents: 2, distinctValues: 2, multiValued: false },
        { type: "string", documents: 1, distinctValues: 1, multiValued: false },
      ],
    });

    const nested: MetadataFilter = {
      operator: "and",
      operands: [
        { field: "team", operator: "in", value: ["core", "docs"] },
        {
          operator: "or",
          operands: [
            { field: "priority", operator: "gte", value: 2 },
            { operator: "not", operand: { field: "priority", operator: "type", value: "number" } },
          ],
        },
      ],
    };
    expect(coverage({ key: "priority", filter: nested })).toEqual({
      key: "priority",
      eligibleDocuments: 4,
      coveredDocuments: 4,
      types: [
        { type: "number", documents: 2, distinctValues: 1, multiValued: false },
        { type: "boolean", documents: 1, distinctValues: 1, multiValued: false },
        { type: "string", documents: 1, distinctValues: 1, multiValued: false },
      ],
    });
  });

  test("executes one statement in a transaction that materializes the eligible documents and key rows once", () => {
    const executions: { sql: string; inTransaction: boolean }[] = [];
    const observed = observeExecutions(store.db, executions);
    const options: MetadataCoverageOptions[] = [
      { key: "priority" },
      { key: "tags", collection: "notes", filter: { field: "team", operator: "eq", value: "core" } },
    ];
    // Table statistics change the planner's choices, so check with and without them.
    for (const statistics of [false, true]) {
      if (statistics) store.db.exec("ANALYZE");
      for (const option of options) {
        executions.length = 0;
        expect(getMetadataCoverage(observed, option)).toEqual(getMetadataCoverage(store.db, option));
        expect(executions.map(execution => execution.inTransaction)).toEqual([true]);
        const plan = planOf(executions[0]!.sql);
        expect(plan.filter(step => step === "MATERIALIZE eligible"), executions[0]!.sql).toHaveLength(1);
        expect(plan.filter(step => step === "MATERIALIZE key_rows"), executions[0]!.sql).toHaveLength(1);
        expect(plan.some(step => step === "CO-ROUTINE eligible" || step === "CO-ROUTINE key_rows"), executions[0]!.sql).toBe(false);
      }
    }
  });

  test("rejects a filter and key whose bindings together exceed the statement budget, before preparing a statement", () => {
    const executions: { sql: string; inTransaction: boolean }[] = [];
    const prepares: string[] = [];
    const observed = observeExecutions(store.db, executions, prepares);
    const members = Array.from({ length: 64 }, (_, index) => `v${index}`);
    // 1 root and 8 groups of 30 conditions: 249 nodes, inside every parser limit.
    // 240 conditions binding field and value per member: 30,720, and the key makes 30,721.
    const wideFilter = parseMetadataFilter({
      operator: "and",
      operands: Array.from({ length: 8 }, () => ({
        operator: "and",
        operands: Array.from({ length: 30 }, () => ({ field: "status", operator: "all", value: members })),
      })),
    });

    try {
      getMetadataCoverage(observed, { key: "status", filter: wideFilter });
      throw new Error("expected getMetadataCoverage to throw");
    } catch (error) {
      expect(error).toBeInstanceOf(MetadataBindingBudgetError);
      const budgetError = error as MetadataBindingBudgetError;
      expect(budgetError.bindings).toBe(30721);
      expect(budgetError.budget).toBe(30000);
      expect(budgetError.message).toBe("filter and key together bind 30721 SQL parameters, over the budget of 30000. Shorten their membership lists.");
    }
    expect(prepares).toEqual([]);
    expect(executions).toEqual([]);
  });
});

describe("SDK getMetadataCoverage", () => {
  let qmd: QMDStore;

  beforeAll(async () => {
    const docsDir = join(testDir, "sdk-docs");
    const rejectedDir = join(testDir, "sdk-rejected");
    await mkdir(docsDir, { recursive: true });
    await mkdir(rejectedDir, { recursive: true });
    const frontmatter = (status: string) => `---\nqmd:\n  metadata:\n    status: ${status}\n---\n\n`;
    // The guide's example: a and b are drafts, c has no status, d holds both values.
    await writeFile(join(docsDir, "a.md"), `${frontmatter("draft")}# A\n`);
    await writeFile(join(docsDir, "b.md"), `${frontmatter("draft")}# B\n`);
    await writeFile(join(docsDir, "c.md"), "# C\n");
    await writeFile(join(docsDir, "d.md"), `${frontmatter("[draft, published]")}# D\n`);
    await writeFile(join(rejectedDir, "e.md"), `${frontmatter("[]")}# E\n`);

    qmd = await createQmdStore({
      dbPath: join(testDir, "sdk.sqlite"),
      config: { collections: { docs: { path: docsDir, pattern: "**/*.md" }, rejected: { path: rejectedDir, pattern: "**/*.md" } } },
    });
    await qmd.update();
  });

  afterAll(async () => {
    await qmd.close();
  });

  test("runs the guide's four-document example, with an empty-array document outside eligibility", async () => {
    const example = { key: "status", eligibleDocuments: 4, coveredDocuments: 3, types: [{ type: "string", documents: 3, distinctValues: 2, multiValued: true }] };
    expect(await qmd.getMetadataCoverage({ key: "status", collection: "docs" })).toEqual(example);

    // Extraction rejects e's empty array: listMetadata counts it as active, coverage leaves it out.
    expect((await qmd.listMetadata()).documents).toBe(5);
    expect(await qmd.getMetadataCoverage({ key: "status" })).toEqual(example);
  });

  test("rejects a malformed key or filter at the runtime boundary", async () => {
    await expect(qmd.getMetadataCoverage({ key: "" })).rejects.toThrow(/^Invalid key: expected a non-empty string of at most 128 UTF-8 bytes, received ""$/);
    await expect(qmd.getMetadataCoverage({ key: "k".repeat(129) })).rejects.toThrow(MetadataOptionError);
    expect(await qmd.getMetadataCoverage({ key: "k".repeat(128) })).toEqual({ key: "k".repeat(128), eligibleDocuments: 4, coveredDocuments: 0, types: [] });

    // Plain-JS callers bypass the declarations, so the SDK must reject a bad AST at runtime.
    const untrusted: MetadataCoverageOptions = JSON.parse('{"key":"status","filter":{"field":"status","operator":"equal","value":"x"}}');
    await expect(qmd.getMetadataCoverage(untrusted)).rejects.toThrow(MetadataFilterError);
  });
});
