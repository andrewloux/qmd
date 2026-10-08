/**
 * Deep vector inspection validates recorded embedding coverage and the
 * bidirectional relationship between relational row mappings and sqlite-vec.
 */
import { afterEach, describe, expect, test } from "vitest";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { openDatabase } from "../src/db.js";
import { LlamaCpp, type EmbeddingResult } from "../src/llm.js";
import {
  createStore,
  deactivateDocument,
  generateEmbeddings,
  getEmbeddingFingerprint,
  insertContent,
  insertDocument,
  type Store,
} from "../src/store.js";
import {
  LEGACY_VEC_TABLE,
  VEC_ROWS_TABLE,
  VEC_TABLE,
  allocateCollectionId,
  resolveCollectionId,
  vecInteger,
} from "../src/vec-layout.js";
import { inspectVectorIndex } from "../src/vector-inspection.js";

const MODEL = "hf:test/vector-inspection.gguf";
const OTHER_MODEL = "hf:test/vector-inspection-other.gguf";

let store: Store | null = null;
let dir: string | null = null;

async function openStore(): Promise<Store> {
  dir = await mkdtemp(join(tmpdir(), "qmd-vector-inspection-"));
  store = createStore(join(dir, "index.sqlite"));
  return store;
}

afterEach(async () => {
  store?.close();
  store = null;
  if (dir) await rm(dir, { recursive: true, force: true });
  dir = null;
});

function insertDoc(s: Store, collection: string, hash: string, path: string): void {
  const now = new Date().toISOString();
  insertContent(s.db, hash, `# ${hash}\n\nBody for ${hash}.`, now);
  insertDocument(s.db, collection, path, hash, hash, now, now);
}

function insertVector(s: Store, collection: string, hash: string): void {
  const now = new Date().toISOString();
  s.insertEmbedding(hash, 0, 0, new Float32Array([1, 2, 3]), MODEL, now, 1, getEmbeddingFingerprint(MODEL));
  expect(resolveCollectionId(s.db, collection)).toBeDefined();
}

function inspect(s: Store, model: string = MODEL) {
  return inspectVectorIndex(
    s.db,
    model,
    getEmbeddingFingerprint(model),
    () => s.getHashesNeedingEmbedding(model),
  );
}

function mappedRow(s: Store, hash: string): { rowid: number; collectionId: number } {
  const row = s.db.prepare(`
    SELECT id AS rowid, collection_id AS collectionId
    FROM ${VEC_ROWS_TABLE}
    WHERE hash = ? AND seq = 0
  `).get(hash) as { rowid: number; collectionId: number } | undefined;
  if (!row) throw new Error(`missing vector mapping for ${hash}`);
  return row;
}

class FakeLlm extends LlamaCpp {
  constructor() {
    super({ embedModel: MODEL });
  }

  override async tokenize(text: string): ReturnType<LlamaCpp["tokenize"]> {
    return new Array(Math.max(1, Math.ceil(text.length / 16))).fill(1);
  }

  override async detokenize(tokens: Awaited<ReturnType<LlamaCpp["tokenize"]>>): Promise<string> {
    return "x".repeat(tokens.length * 16);
  }

  override async embed(): Promise<EmbeddingResult> {
    return { embedding: [0.1, 0.2, 0.3], model: MODEL };
  }

  override async embedBatch(texts: string[]): Promise<EmbeddingResult[]> {
    return texts.map(() => ({ embedding: [0.1, 0.2, 0.3], model: MODEL }));
  }
}

describe("inspectVectorIndex", () => {
  test("evaluates pending coverage inside the same snapshot and accepts an empty absent index", async () => {
    const s = await openStore();
    let observedTransaction = false;

    const result = inspectVectorIndex(
      s.db,
      MODEL,
      getEmbeddingFingerprint(MODEL),
      () => {
        observedTransaction = s.db.inTransaction;
        return s.getHashesNeedingEmbedding(MODEL);
      },
    );

    expect(observedTransaction).toBe(true);
    expect(result).toMatchObject({
      partitionState: "absent",
      activeDocuments: 0,
      needsEmbedding: 0,
      inconsistentChunkLayouts: 0,
      requiredPartitionRows: 0,
      missingRequiredPartitionRows: 0,
      inconsistentPeerRows: 0,
      structurallyReady: true,
    });
  });

  test("reports required active rows missing from an absent index", async () => {
    const s = await openStore();
    insertDoc(s, "docs", "absent-hash", "absent.md");
    const now = new Date().toISOString();
    s.db.prepare(`
      INSERT INTO content_vectors
        (hash, seq, pos, model, embed_fingerprint, total_chunks, embedded_at)
      VALUES (?, 0, 0, ?, ?, 1, ?)
    `).run("absent-hash", MODEL, getEmbeddingFingerprint(MODEL), now);

    expect(inspect(s)).toMatchObject({
      partitionState: "absent",
      activeDocuments: 1,
      needsEmbedding: 0,
      requiredPartitionRows: 1,
      missingRequiredPartitionRows: 1,
      inconsistentPeerRows: 0,
      structurallyReady: false,
    });
  });

  test("reports legacy vector storage without claiming peer counts", async () => {
    const s = await openStore();
    insertDoc(s, "docs", "legacy-hash", "legacy.md");
    const now = new Date().toISOString();
    s.db.prepare(`
      INSERT INTO content_vectors
        (hash, seq, pos, model, embed_fingerprint, total_chunks, embedded_at)
      VALUES (?, 0, 0, ?, ?, 1, ?)
    `).run("legacy-hash", MODEL, getEmbeddingFingerprint(MODEL), now);
    s.db.exec(`
      CREATE VIRTUAL TABLE ${LEGACY_VEC_TABLE}
      USING vec0(hash_seq TEXT PRIMARY KEY, embedding float[3] distance_metric=cosine)
    `);

    expect(inspect(s)).toMatchObject({
      partitionState: "legacy",
      activeDocuments: 1,
      needsEmbedding: 0,
      requiredPartitionRows: 1,
      missingRequiredPartitionRows: null,
      inconsistentPeerRows: null,
      structurallyReady: false,
    });
  });

  test("reports an index opened without sqlite-vec as unreadable", async () => {
    const s = await openStore();
    s.ensureVecTable(3);
    insertDoc(s, "docs", "unreadable-hash", "unreadable.md");
    insertVector(s, "docs", "unreadable-hash");
    const dbPath = s.dbPath;
    s.close();
    store = null;

    const rawDb = openDatabase(dbPath);
    try {
      expect(inspectVectorIndex(
        rawDb,
        MODEL,
        getEmbeddingFingerprint(MODEL),
        () => 0,
      )).toMatchObject({
        partitionState: "unreadable",
        activeDocuments: 1,
        requiredPartitionRows: 1,
        missingRequiredPartitionRows: null,
        inconsistentPeerRows: null,
        structurallyReady: false,
      });
    } finally {
      rawDb.close();
    }
  });

  test("scopes required rows to the selected model and accepts coherent inactive cache rows", async () => {
    const s = await openStore();
    s.ensureVecTable(3);
    insertDoc(s, "docs", "active-hash", "active.md");
    insertVector(s, "docs", "active-hash");
    insertDoc(s, "archive", "cached-hash", "cached.md");
    insertVector(s, "archive", "cached-hash");
    deactivateDocument(s.db, "archive", "cached.md");

    expect(inspect(s)).toEqual({
      model: MODEL,
      embeddingFingerprint: getEmbeddingFingerprint(MODEL),
      partitionState: "checked",
      activeDocuments: 1,
      needsEmbedding: 0,
      inconsistentChunkLayouts: 0,
      requiredPartitionRows: 1,
      missingRequiredPartitionRows: 0,
      inconsistentPeerRows: 0,
      structurallyReady: true,
    });

    const other = inspect(s, OTHER_MODEL);
    expect(other).toMatchObject({
      model: OTHER_MODEL,
      embeddingFingerprint: getEmbeddingFingerprint(OTHER_MODEL),
      needsEmbedding: 1,
      inconsistentChunkLayouts: 1,
      requiredPartitionRows: 0,
      structurallyReady: false,
    });
  });

  test("rejects excess, non-contiguous, and off-generation chunks missed by ordinary health", async () => {
    const s = await openStore();
    s.ensureVecTable(3);
    const now = new Date().toISOString();
    const fingerprint = getEmbeddingFingerprint(MODEL);

    insertDoc(s, "docs", "excess", "excess.md");
    for (const seq of [0, 1, 2, 3]) {
      s.insertEmbedding("excess", seq, seq * 10, new Float32Array([1, 2, 3]), MODEL, now, 3, fingerprint);
    }

    insertDoc(s, "docs", "gap", "gap.md");
    for (const seq of [0, 2, 3]) {
      s.insertEmbedding("gap", seq, seq * 10, new Float32Array([1, 2, 3]), MODEL, now, 3, fingerprint);
    }

    insertDoc(s, "docs", "off-generation", "off-generation.md");
    s.insertEmbedding("off-generation", 0, 0, new Float32Array([1, 2, 3]), MODEL, now, 1, fingerprint);
    s.insertEmbedding(
      "off-generation",
      1,
      10,
      new Float32Array([1, 2, 3]),
      OTHER_MODEL,
      now,
      2,
      getEmbeddingFingerprint(OTHER_MODEL),
    );

    expect(s.getHashesNeedingEmbedding(MODEL)).toBe(0);
    expect(inspect(s)).toMatchObject({
      needsEmbedding: 0,
      inconsistentChunkLayouts: 3,
      requiredPartitionRows: 8,
      missingRequiredPartitionRows: 0,
      inconsistentPeerRows: 0,
      structurallyReady: false,
    });
  });

  test("rejects a null total_chunks layout missed by ordinary health", async () => {
    const s = await openStore();
    s.ensureVecTable(3);
    insertDoc(s, "docs", "null-total", "null-total.md");
    insertVector(s, "docs", "null-total");

    // Rebuild this temp fixture with the same production columns and a
    // nullable total_chunks so the inspection can diagnose stored corruption
    // that the ordinary aggregate comparison treats as SQL NULL.
    s.db.exec(`
      DROP INDEX idx_content_vectors_model_fingerprint;
      ALTER TABLE content_vectors RENAME TO strict_content_vectors;
      CREATE TABLE content_vectors (
        hash TEXT NOT NULL,
        seq INTEGER NOT NULL DEFAULT 0,
        pos INTEGER NOT NULL DEFAULT 0,
        model TEXT NOT NULL,
        embed_fingerprint TEXT NOT NULL DEFAULT '',
        total_chunks INTEGER,
        embedded_at TEXT NOT NULL,
        PRIMARY KEY (hash, seq)
      );
      INSERT INTO content_vectors
        (hash, seq, pos, model, embed_fingerprint, total_chunks, embedded_at)
      SELECT hash, seq, pos, model, embed_fingerprint, NULL, embedded_at
      FROM strict_content_vectors;
      DROP TABLE strict_content_vectors;
    `);

    expect(s.getHashesNeedingEmbedding(MODEL)).toBe(0);
    expect(inspect(s)).toMatchObject({
      needsEmbedding: 0,
      inconsistentChunkLayouts: 1,
      requiredPartitionRows: 1,
      missingRequiredPartitionRows: 0,
      inconsistentPeerRows: 0,
      structurallyReady: false,
    });
  });

  test("rejects invalid recorded chunk positions missed by ordinary health", async () => {
    const s = await openStore();
    s.ensureVecTable(3);
    const now = new Date().toISOString();
    const fingerprint = getEmbeddingFingerprint(MODEL);

    const malformedStarts = [
      { hash: "negative-pos", pos: -1 },
      { hash: "float-pos", pos: 0.5 },
      { hash: "unsafe-pos", pos: 9_007_199_254_740_992 },
      { hash: "nonzero-start", pos: 5 },
    ];
    for (const { hash, pos } of malformedStarts) {
      insertDoc(s, "docs", hash, `${hash}.md`);
      s.insertEmbedding(hash, 0, pos, new Float32Array([1, 2, 3]), MODEL, now, 1, fingerprint);
    }

    insertDoc(s, "docs", "nonmonotone-pos", "nonmonotone-pos.md");
    s.insertEmbedding("nonmonotone-pos", 0, 0, new Float32Array([1, 2, 3]), MODEL, now, 2, fingerprint);
    s.insertEmbedding("nonmonotone-pos", 1, 0, new Float32Array([1, 2, 3]), MODEL, now, 2, fingerprint);

    expect(s.getHashesNeedingEmbedding(MODEL)).toBe(0);
    expect(inspect(s)).toMatchObject({
      needsEmbedding: 0,
      inconsistentChunkLayouts: 5,
      requiredPartitionRows: 6,
      missingRequiredPartitionRows: 0,
      inconsistentPeerRows: 0,
      structurallyReady: false,
    });
  });

  test("counts a malformed mapping collection id instead of throwing", async () => {
    const s = await openStore();
    s.ensureVecTable(3);
    insertDoc(s, "docs", "malformed-map", "malformed-map.md");
    insertVector(s, "docs", "malformed-map");
    const row = mappedRow(s, "malformed-map");
    s.db.prepare(`UPDATE ${VEC_ROWS_TABLE} SET collection_id = ? WHERE id = ?`)
      .run("oops", row.rowid);

    expect(inspect(s)).toMatchObject({
      requiredPartitionRows: 1,
      missingRequiredPartitionRows: 1,
      inconsistentPeerRows: 1,
      structurallyReady: false,
    });
  });

  test("streams vec0 rows and detects mapping-only, vec-only, and partition-mismatched rowids", async () => {
    const s = await openStore();
    s.ensureVecTable(3);
    for (const hash of ["map-only", "partition-mismatch", "healthy"]) {
      insertDoc(s, "docs", hash, `${hash}.md`);
      insertVector(s, "docs", hash);
    }
    const otherCollectionId = allocateCollectionId(s.db, "other");

    const mapOnly = mappedRow(s, "map-only");
    s.db.prepare(`DELETE FROM ${VEC_TABLE} WHERE rowid = ?`).run(vecInteger(mapOnly.rowid));

    const mismatch = mappedRow(s, "partition-mismatch");
    s.db.prepare(`DELETE FROM ${VEC_TABLE} WHERE rowid = ?`).run(vecInteger(mismatch.rowid));
    s.db.prepare(`INSERT INTO ${VEC_TABLE} (rowid, collection_id, embedding) VALUES (?, ?, ?)`)
      .run(vecInteger(mismatch.rowid), vecInteger(otherCollectionId), new Float32Array([1, 2, 3]));

    const vecOnlyRowid = 90_001;
    s.db.prepare(`INSERT INTO ${VEC_TABLE} (rowid, collection_id, embedding) VALUES (?, ?, ?)`)
      .run(vecInteger(vecOnlyRowid), vecInteger(otherCollectionId), new Float32Array([1, 2, 3]));

    expect(inspect(s)).toMatchObject({
      partitionState: "checked",
      needsEmbedding: 0,
      inconsistentChunkLayouts: 0,
      requiredPartitionRows: 3,
      missingRequiredPartitionRows: 2,
      inconsistentPeerRows: 3,
      structurallyReady: false,
    });
  });

  test("ordinary embedding copies a missing collection partition", async () => {
    const s = await openStore();
    s.ensureVecTable(3);
    insertDoc(s, "first", "shared-hash", "shared.md");
    insertVector(s, "first", "shared-hash");
    insertDoc(s, "second", "shared-hash", "shared.md");
    s.llm = new FakeLlm();

    expect(inspect(s)).toMatchObject({
      missingRequiredPartitionRows: 1,
      inconsistentPeerRows: 0,
      structurallyReady: false,
    });

    const embedded = await generateEmbeddings(s, { model: MODEL });
    expect(embedded).toMatchObject({ chunksCopied: 1, chunksEmbedded: 0, errors: 0 });
    expect(inspect(s)).toMatchObject({
      missingRequiredPartitionRows: 0,
      inconsistentPeerRows: 0,
      structurallyReady: true,
    });
  });

  test("whole-index forced embedding rebuilds peer corruption", async () => {
    const s = await openStore();
    s.ensureVecTable(3);
    insertDoc(s, "docs", "corrupt-hash", "corrupt.md");
    insertVector(s, "docs", "corrupt-hash");
    const row = mappedRow(s, "corrupt-hash");
    s.db.prepare(`DELETE FROM ${VEC_TABLE} WHERE rowid = ?`).run(vecInteger(row.rowid));
    s.llm = new FakeLlm();

    expect(inspect(s)).toMatchObject({
      missingRequiredPartitionRows: 1,
      inconsistentPeerRows: 1,
      structurallyReady: false,
    });

    const embedded = await generateEmbeddings(s, { model: MODEL, force: true });
    expect(embedded).toMatchObject({ chunksEmbedded: 1, errors: 0 });
    expect(inspect(s)).toMatchObject({
      missingRequiredPartitionRows: 0,
      inconsistentPeerRows: 0,
      structurallyReady: true,
    });
  });
});
