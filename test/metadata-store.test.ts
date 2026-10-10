/**
 * metadata-store.test.ts - Metadata persistence lifecycle: schema wiring,
 * replacement semantics, reindex synchronization, and cleanup cascades.
 */

import { describe, test, expect, beforeAll, afterAll, beforeEach, afterEach } from "vitest";
import { mkdtemp, rm, writeFile, mkdir } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import {
  createStore,
  reindexCollection,
  insertContent,
  insertDocument,
  hashContent,
  removeCollection,
  deleteInactiveDocuments,
  getStatus,
  type Store,
} from "../src/store.js";
import {
  initializeMetadataSchema,
  syncDocumentMetadata,
  syncExternalDocumentMetadata,
  replaceDocumentMetadata,
  countDocumentsPendingMetadata,
  getMetadataByFilepath,
} from "../src/metadata-store.js";
import { METADATA_EXTRACTION_VERSION, type DocumentMetadata } from "../src/metadata.js";
import { openDatabase, type Database, type SQLiteValue } from "../src/db.js";
import { VECTOR_PARTITION_VERSION, getUserVersion, runStoreMigrations } from "../src/store-migrations.js";
import { LEGACY_VEC_TABLE, vecLayout } from "../src/vec-layout.js";

let testDir: string;
let store: Store;

beforeAll(async () => {
  testDir = await mkdtemp(join(tmpdir(), "qmd-metadata-store-"));
});

afterAll(async () => {
  await rm(testDir, { recursive: true, force: true });
});

beforeEach(() => {
  const dbPath = join(testDir, `test-${Date.now()}-${Math.random().toString(36).slice(2)}.sqlite`);
  store = createStore(dbPath);
});

afterEach(() => {
  store.close();
});

function buildDoc(frontmatterYaml: string | null, body: string): string {
  if (frontmatterYaml === null) return body;
  return `---\n${frontmatterYaml}---\n\n${body}`;
}

async function insertDoc(collection: string, path: string, content: string): Promise<number> {
  const now = new Date().toISOString();
  const hash = await hashContent(content);
  insertContent(store.db, hash, content, now);
  return insertDocument(store.db, collection, path, path, hash, now, now);
}

function getMetadataRow(documentId: number): { metadata_json: string; extraction_version: number; extraction_error: string | null } | undefined {
  return store.db.prepare(`
    SELECT metadata_json, extraction_version, extraction_error
    FROM document_metadata WHERE document_id = ?
  `).get(documentId) as { metadata_json: string; extraction_version: number; extraction_error: string | null } | undefined;
}

function getMetadataState(documentId: number): { source: string; metadata_json: string; extraction_error: string | null; extracted_at: string } | undefined {
  return store.db.prepare(`
    SELECT source, metadata_json, extraction_error, extracted_at
    FROM document_metadata WHERE document_id = ?
  `).get(documentId) as { source: string; metadata_json: string; extraction_error: string | null; extracted_at: string } | undefined;
}

function countValueRows(documentId: number): number {
  const row = store.db.prepare(`SELECT COUNT(*) as c FROM document_metadata_values WHERE document_id = ?`)
    .get(documentId) as { c: number };
  return row.c;
}

describe("syncDocumentMetadata", () => {
  test("persists metadata and value rows for a new document", async () => {
    const content = buildDoc("qmd:\n  metadata:\n    topics: [a, b]\n    priority: 3\n", "# Doc\n");
    const documentId = await insertDoc("notes", "doc.md", content);

    const extraction = syncDocumentMetadata(store.db, documentId, content, "doc.md");
    expect(extraction?.error).toBeUndefined();

    const row = getMetadataRow(documentId);
    expect(JSON.parse(row!.metadata_json)).toEqual({ topics: ["a", "b"], priority: 3 });
    expect(row!.extraction_version).toBe(METADATA_EXTRACTION_VERSION);
    expect(row!.extraction_error).toBeNull();
    expect(countValueRows(documentId)).toBe(3);
  });

  test("records extraction state for documents without metadata", async () => {
    const content = "# Plain doc\n";
    const documentId = await insertDoc("notes", "plain.md", content);

    syncDocumentMetadata(store.db, documentId, content, "plain.md");

    const row = getMetadataRow(documentId);
    expect(JSON.parse(row!.metadata_json)).toEqual({});
    expect(row!.extraction_error).toBeNull();
    expect(countValueRows(documentId)).toBe(0);
  });

  test("onlyIfStale skips documents with a current extraction row", async () => {
    const content = buildDoc("qmd:\n  metadata:\n    status: ok\n", "# Doc\n");
    const documentId = await insertDoc("notes", "doc.md", content);

    expect(syncDocumentMetadata(store.db, documentId, content, "doc.md")).not.toBeNull();
    expect(syncDocumentMetadata(store.db, documentId, content, "doc.md", { onlyIfStale: true })).toBeNull();
    expect(syncDocumentMetadata(store.db, documentId, content, "doc.md")).not.toBeNull();
  });

  test("invalid metadata replaces prior rows with empty metadata and an error", async () => {
    const documentId = await insertDoc("notes", "doc.md", "# Doc\n");

    const validContent = buildDoc("qmd:\n  metadata:\n    status: ok\n", "# Doc\n");
    syncDocumentMetadata(store.db, documentId, validContent, "doc.md");
    expect(countValueRows(documentId)).toBe(1);

    const invalidContent = buildDoc("qmd:\n  metadata:\n    status: null\n", "# Doc\n");
    const extraction = syncDocumentMetadata(store.db, documentId, invalidContent, "doc.md");
    expect(extraction?.error).toMatch(/null is not supported/);

    const row = getMetadataRow(documentId);
    expect(JSON.parse(row!.metadata_json)).toEqual({});
    expect(row!.extraction_error).toMatch(/null is not supported/);
    expect(countValueRows(documentId)).toBe(0);
  });

  test("metadata removal clears value rows", async () => {
    const documentId = await insertDoc("notes", "doc.md", "# Doc\n");

    syncDocumentMetadata(store.db, documentId, buildDoc("qmd:\n  metadata:\n    status: ok\n", "# Doc\n"), "doc.md");
    expect(countValueRows(documentId)).toBe(1);

    syncDocumentMetadata(store.db, documentId, "# Doc\n", "doc.md");
    expect(countValueRows(documentId)).toBe(0);
    expect(getMetadataRow(documentId)!.extraction_error).toBeNull();
  });

  test("an identical external answer keeps its row until frontmatter extraction takes the row back", async () => {
    const content = buildDoc("qmd:\n  metadata:\n    status: draft\n", "# Doc\n");
    const documentId = await insertDoc("notes", "doc.md", content);
    syncDocumentMetadata(store.db, documentId, content, "doc.md");

    const published = { metadata: { status: "published" }, extractionVersion: METADATA_EXTRACTION_VERSION };
    syncExternalDocumentMetadata(store.db, documentId, published);
    const external = getMetadataState(documentId);
    expect(external).toMatchObject({ source: "external", metadata_json: '{"status":"published"}', extraction_error: null });

    await new Promise(resolve => setTimeout(resolve, 5));
    syncExternalDocumentMetadata(store.db, documentId, { ...published, metadata: { status: "published" } });
    expect(getMetadataState(documentId)).toEqual(external);

    // An external row is stale for frontmatter, and the re-extraction writes its origin on conflict.
    expect(syncDocumentMetadata(store.db, documentId, content, "doc.md", { onlyIfStale: true })).not.toBeNull();
    expect(getMetadataState(documentId)).toMatchObject({ source: "frontmatter", metadata_json: '{"status":"draft"}' });
    expect(syncDocumentMetadata(store.db, documentId, content, "doc.md", { onlyIfStale: true })).toBeNull();
  });

  test("two paths sharing one content hash keep separate metadata rows", async () => {
    const content = buildDoc("qmd:\n  metadata:\n    status: shared\n", "# Same content\n");
    const firstId = await insertDoc("notes", "first.md", content);
    const secondId = await insertDoc("docs", "second.md", content);

    syncDocumentMetadata(store.db, firstId, content, "first.md");
    replaceDocumentMetadata(store.db, secondId, {
      metadata: { status: "overridden" },
      extractionVersion: METADATA_EXTRACTION_VERSION,
    });

    expect(JSON.parse(getMetadataRow(firstId)!.metadata_json)).toEqual({ status: "shared" });
    expect(JSON.parse(getMetadataRow(secondId)!.metadata_json)).toEqual({ status: "overridden" });
  });
});

describe("countDocumentsPendingMetadata", () => {
  test("counts unextracted, stale, and errored documents", async () => {
    const extractedId = await insertDoc("notes", "extracted.md", "# A\n");
    const pendingId = await insertDoc("notes", "pending.md", "# B\n");
    const erroredId = await insertDoc("notes", "errored.md", "# C\n");
    const staleId = await insertDoc("notes", "stale.md", "# D\n");

    syncDocumentMetadata(store.db, extractedId, "# A\n", "extracted.md");
    replaceDocumentMetadata(store.db, erroredId, {
      metadata: {},
      error: "boom",
      extractionVersion: METADATA_EXTRACTION_VERSION,
    });
    replaceDocumentMetadata(store.db, staleId, {
      metadata: {},
      extractionVersion: METADATA_EXTRACTION_VERSION - 1,
    });

    expect(countDocumentsPendingMetadata(store.db)).toBe(3);
    expect(getStatus(store.db).pendingMetadata).toBe(3);

    // Deactivated documents leave the pending count.
    store.db.prepare(`UPDATE documents SET active = 0 WHERE id = ?`).run(pendingId);
    expect(countDocumentsPendingMetadata(store.db)).toBe(2);
  });
});

describe("reindexCollection metadata synchronization", () => {
  let collectionDir: string;

  beforeEach(async () => {
    collectionDir = join(testDir, `coll-${Date.now()}-${Math.random().toString(36).slice(2)}`);
    await mkdir(collectionDir, { recursive: true });
  });

  async function reindex() {
    return reindexCollection(store, collectionDir, "**/*.md", "notes");
  }

  function getMetadataForPath(path: string): DocumentMetadata | undefined {
    return getMetadataByFilepath(store.db, [`qmd://notes/${path}`]).get(`qmd://notes/${path}`);
  }

  test("extracts metadata for new, changed, and unchanged documents", async () => {
    await writeFile(join(collectionDir, "doc.md"), buildDoc("qmd:\n  metadata:\n    status: draft\n", "# Doc\n"));

    const firstResult = await reindex();
    expect(firstResult.indexed).toBe(1);
    expect(firstResult.metadataErrors).toBe(0);
    expect(getMetadataForPath("doc.md")).toEqual({ status: "draft" });
    expect(countDocumentsPendingMetadata(store.db)).toBe(0);

    // Unchanged content on a later pass keeps extraction current.
    const secondResult = await reindex();
    expect(secondResult.unchanged).toBe(1);
    expect(countDocumentsPendingMetadata(store.db)).toBe(0);

    // Edited metadata replaces the previous rows.
    await writeFile(join(collectionDir, "doc.md"), buildDoc("qmd:\n  metadata:\n    status: published\n", "# Doc\n"));
    await reindex();
    expect(getMetadataForPath("doc.md")).toEqual({ status: "published" });
  });

  test("backfills documents indexed before the metadata schema existed", async () => {
    await writeFile(join(collectionDir, "doc.md"), buildDoc("qmd:\n  metadata:\n    status: ok\n", "# Doc\n"));
    await reindex();

    // Simulate a pre-metadata index: drop the extraction rows.
    store.db.prepare(`DELETE FROM document_metadata`).run();
    expect(countDocumentsPendingMetadata(store.db)).toBe(1);

    const result = await reindex();
    expect(result.unchanged).toBe(1);
    expect(countDocumentsPendingMetadata(store.db)).toBe(0);
    expect(getMetadataForPath("doc.md")).toEqual({ status: "ok" });
  });

  test("re-extracts an unchanged file whose metadata came from an older extraction version", async () => {
    await writeFile(join(collectionDir, "doc.md"), buildDoc("qmd:\n  metadata:\n    status: ok\n", "# Doc\n"));
    await reindex();

    store.db.prepare(`UPDATE document_metadata SET extraction_version = 0`).run();
    expect(countDocumentsPendingMetadata(store.db)).toBe(1);

    const result = await reindex();
    expect(result.unchanged).toBe(1);
    const row = store.db.prepare(`SELECT extraction_version FROM document_metadata`).get() as { extraction_version: number };
    expect(row.extraction_version).toBe(METADATA_EXTRACTION_VERSION);
    expect(countDocumentsPendingMetadata(store.db)).toBe(0);
    expect(getMetadataForPath("doc.md")).toEqual({ status: "ok" });
  });

  test("counts extraction errors without aborting the collection", async () => {
    await writeFile(join(collectionDir, "bad.md"), buildDoc("qmd:\n  metadata:\n    mixed: [1, two]\n", "# Bad\n"));
    await writeFile(join(collectionDir, "good.md"), buildDoc("qmd:\n  metadata:\n    status: ok\n", "# Good\n"));

    const result = await reindex();
    expect(result.indexed).toBe(2);
    expect(result.metadataErrors).toBe(1);
    expect(getMetadataForPath("good.md")).toEqual({ status: "ok" });
    expect(getMetadataForPath("bad.md")).toEqual({});
    expect(countDocumentsPendingMetadata(store.db)).toBe(1);
  });

  test("deactivation excludes metadata from batch loads; reactivation restores it", async () => {
    await writeFile(join(collectionDir, "doc.md"), buildDoc("qmd:\n  metadata:\n    status: ok\n", "# Doc\n"));
    await reindex();

    await rm(join(collectionDir, "doc.md"));
    const removedResult = await reindex();
    expect(removedResult.removed).toBe(1);
    expect(getMetadataForPath("doc.md")).toBeUndefined();

    await writeFile(join(collectionDir, "doc.md"), buildDoc("qmd:\n  metadata:\n    status: ok\n", "# Doc\n"));
    await reindex();
    expect(getMetadataForPath("doc.md")).toEqual({ status: "ok" });
  });

  test("hard deletion cascades metadata rows", async () => {
    await writeFile(join(collectionDir, "doc.md"), buildDoc("qmd:\n  metadata:\n    status: ok\n", "# Doc\n"));
    await reindex();

    await rm(join(collectionDir, "doc.md"));
    await reindex();

    deleteInactiveDocuments(store.db);
    expect((store.db.prepare(`SELECT COUNT(*) as c FROM document_metadata`).get() as { c: number }).c).toBe(0);
    expect((store.db.prepare(`SELECT COUNT(*) as c FROM document_metadata_values`).get() as { c: number }).c).toBe(0);
  });

  test("collection removal cascades metadata rows", async () => {
    await writeFile(join(collectionDir, "doc.md"), buildDoc("qmd:\n  metadata:\n    status: ok\n", "# Doc\n"));
    await reindex();

    removeCollection(store.db, "notes");
    expect((store.db.prepare(`SELECT COUNT(*) as c FROM document_metadata`).get() as { c: number }).c).toBe(0);
    expect((store.db.prepare(`SELECT COUNT(*) as c FROM document_metadata_values`).get() as { c: number }).c).toBe(0);
  });
});

describe("document_metadata.source migration", () => {
  type ColumnInfo = { name: string; dflt_value: string | null; notnull: number };

  function metadataColumns(db: Database): ColumnInfo[] {
    return db.prepare(`PRAGMA table_info(document_metadata)`).all() as ColumnInfo[];
  }

  /** Rebuild document_metadata in its layout from before metadata sources, rows included. */
  function restorePreSourceMetadataTable(db: Database): void {
    db.exec(`PRAGMA foreign_keys = OFF`);
    db.exec(`
      CREATE TABLE document_metadata_before_sources (
        document_id INTEGER PRIMARY KEY,
        metadata_json TEXT NOT NULL DEFAULT '{}',
        extraction_version INTEGER NOT NULL,
        extraction_error TEXT,
        extracted_at TEXT NOT NULL,
        FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE
      )
    `);
    db.exec(`
      INSERT INTO document_metadata_before_sources
      SELECT document_id, metadata_json, extraction_version, extraction_error, extracted_at FROM document_metadata
    `);
    db.exec(`DROP TABLE document_metadata`);
    db.exec(`ALTER TABLE document_metadata_before_sources RENAME TO document_metadata`);
    db.exec(`PRAGMA foreign_keys = ON`);
  }

  async function seedFrontmatterRow(): Promise<void> {
    const content = buildDoc("qmd:\n  metadata:\n    status: ok\n", "# Doc\n");
    const documentId = await insertDoc("notes", "doc.md", content);
    syncDocumentMetadata(store.db, documentId, content, "doc.md");
  }

  test("a fresh index declares source with the frontmatter default and keeps the vector user_version", () => {
    expect(metadataColumns(store.db).find(column => column.name === "source"))
      .toMatchObject({ dflt_value: "'frontmatter'", notnull: 1 });
    expect(getUserVersion(store.db)).toBe(VECTOR_PARTITION_VERSION);
    expect(() => store.db.exec(`INSERT INTO document_metadata (document_id, extraction_version, extracted_at, source) VALUES (1, 1, 'now', 'yaml')`))
      .toThrow(/CHECK constraint failed/);
  });

  test("an index from before metadata sources gains the column at open while the vector step defers", async () => {
    await seedFrontmatterRow();
    restorePreSourceMetadataTable(store.db);
    store.db.exec(`CREATE VIRTUAL TABLE ${LEGACY_VEC_TABLE} USING vec0(hash_seq TEXT PRIMARY KEY, embedding float[3] distance_metric=cosine)`);
    store.db.exec(`PRAGMA user_version = 1`);
    expect(metadataColumns(store.db).map(column => column.name)).not.toContain("source");

    // initializeDatabase's order on a host without sqlite-vec: metadata schema, then store migrations.
    const withoutVec = openDatabase(store.dbPath);
    try {
      initializeMetadataSchema(withoutVec);
      runStoreMigrations(withoutVec, { installFtsSyncTriggers: () => {}, sqliteVecAvailable: false });

      expect(getUserVersion(withoutVec)).toBe(1);
      expect(vecLayout(withoutVec).kind).toBe("legacy");
      expect(withoutVec.prepare(`SELECT source, metadata_json, extraction_error FROM document_metadata`).all())
        .toEqual([{ source: "frontmatter", metadata_json: '{"status":"ok"}', extraction_error: null }]);
    } finally {
      withoutVec.close();
    }

    expect(countDocumentsPendingMetadata(store.db)).toBe(0);
    expect(getMetadataByFilepath(store.db, ["qmd://notes/doc.md"]).get("qmd://notes/doc.md")).toEqual({ status: "ok" });
  });

  test("an opener whose first check loses the race to another opener's ALTER skips its own", async () => {
    await seedFrontmatterRow();
    restorePreSourceMetadataTable(store.db);

    const first = openDatabase(store.dbPath);
    const second = openDatabase(store.dbPath);
    try {
      // The second opener adds the column between the first opener's quick
      // check and its IMMEDIATE transaction.
      let raced = false;
      const racingFirst: Database = {
        get inTransaction() { return first.inTransaction; },
        exec: (sql: string) => first.exec(sql),
        loadExtension: (path: string) => first.loadExtension(path),
        transaction: first.transaction.bind(first),
        close: () => first.close(),
        prepare: (sql: string) => {
          const statement = first.prepare(sql);
          if (raced || !sql.includes("PRAGMA table_info(document_metadata)")) return statement;
          raced = true;
          return {
            run: statement.run.bind(statement),
            get: statement.get.bind(statement),
            iterate: statement.iterate.bind(statement),
            all: <T>(...params: SQLiteValue[]): T[] => {
              const columnsBeforeRace = statement.all<T>(...params);
              initializeMetadataSchema(second);
              return columnsBeforeRace;
            },
          };
        },
      };

      expect(() => initializeMetadataSchema(racingFirst)).not.toThrow();
      expect(raced).toBe(true);
      expect(metadataColumns(first).filter(column => column.name === "source")).toHaveLength(1);
      expect(first.prepare(`SELECT source FROM document_metadata`).all()).toEqual([{ source: "frontmatter" }]);
    } finally {
      first.close();
      second.close();
    }
  });
});
