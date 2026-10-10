/**
 * The embedding fingerprint names the chunk-boundary algorithm and the
 * index-wide chunk strategy. Vectors stored under another fingerprint become
 * pending, and one successful embed run replaces them.
 */
import { afterEach, describe, expect, test } from "vitest";
import { createHash } from "node:crypto";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { LlamaCpp, setDefaultLlamaCpp, type EmbeddingResult } from "../src/llm.js";
import {
  CHUNKER_VERSION,
  chunkDocumentByTokens,
  createStore,
  generateEmbeddings,
  getEmbeddingChunkStrategy,
  getEmbeddingFingerprint,
  getHashesForEmbedding,
  getHashesNeedingEmbedding,
  getIndexEmbeddingFingerprint,
  insertContent,
  insertDocument,
  type ChunkStrategy,
  type Store,
} from "../src/store.js";
import { VEC_ROWS_TABLE, VEC_TABLE, vecInteger } from "../src/vec-layout.js";
import { inspectVectorIndex } from "../src/vector-inspection.js";

const MODEL = "hf:test/embedding-fingerprint.gguf";

/** The fingerprint QMD computed for MODEL before CHUNKER_VERSION existed. */
const PRE_VERSION_FINGERPRINT = "d78403";

/**
 * Chunk-boundary digest of GOLDEN_CORPUS for each CHUNKER_VERSION. A boundary
 * change fails the pin test: bump CHUNKER_VERSION in src/store.ts and record
 * the new digest here. Embedding input formats change the fingerprint on
 * their own and need no version bump.
 */
const LAYOUT_SHA256_BY_VERSION: Record<number, string> = {
  1: "e5335afa65dcf7847b53faf0a8b853c0ef6ad7d78883ab591040f4159bbd0b4c",
};

const sha256 = (text: string) => createHash("sha256").update(text).digest("hex");

/** Counts `tokensPerChar` tokens per character; texts that `fails` selects get no vector. */
class CountingTokenLlm extends LlamaCpp {
  readonly tokensPerChar: number;
  readonly fails: (text: string) => boolean;

  constructor(tokensPerChar: number, fails: (text: string) => boolean = () => false) {
    super({ embedModel: MODEL });
    this.tokensPerChar = tokensPerChar;
    this.fails = fails;
  }

  override async tokenize(text: string): ReturnType<LlamaCpp["tokenize"]> {
    return new Array(Math.max(1, Math.ceil(text.length * this.tokensPerChar))).fill(1);
  }

  override async detokenize(tokens: Awaited<ReturnType<LlamaCpp["tokenize"]>>): Promise<string> {
    return "x".repeat(Math.ceil(tokens.length / this.tokensPerChar));
  }

  override async embed(text: string): Promise<EmbeddingResult | null> {
    return this.fails(text) ? null : { embedding: [0.1, 0.2, 0.3], model: MODEL };
  }

  override async embedBatch(texts: string[]): Promise<(EmbeddingResult | null)[]> {
    return texts.map(text => (this.fails(text) ? null : { embedding: [0.1, 0.2, 0.3], model: MODEL }));
  }
}

const MARKDOWN = Array.from({ length: 24 }, (_, index) => [
  `## Section ${index}`,
  "",
  `Paragraph ${index} carries ordinary prose. `.repeat(12),
  "",
  "```ts",
  `export const value${index} = ${index};`,
  "```",
  "",
].join("\n")).join("\n");

/** One 6,000-character line: the chunker re-cuts each over-limit first-pass chunk. */
const DENSE_LINE = Array.from({ length: 1_200 }, (_, index) => `w${String(index).padStart(4, "0")}`).join("");

const UNICODE = Array.from({ length: 300 }, (_, index) => `漢字かな 😀 emoji ${index}. `).join("");

const TYPESCRIPT = Array.from({ length: 60 }, (_, index) => [
  `export function handler${index}(input: string): string {`,
  "  const value = input.trim();",
  `  return value + "-${index}";`,
  "}",
  "",
].join("\n")).join("\n");

const OTHER_BODY = "# Other\n\nAn unrelated document with one chunk.";

/** "###" headings near the first-pass cuts, so the h3 break score decides where chunks end. */
const PROBE_H3 = Array.from({ length: 12 }, (_, index) =>
  `### Step ${index}\n\n` + `Step ${index} explains one synthetic task in plain words. `.repeat(6)).join("\n\n");

/** Inputs whose boundaries the pin covers: break points, re-cuts, equal starts, surrogate pairs, AST and one chunk. */
const GOLDEN_CORPUS: Array<{ name: string; body: string; tokensPerChar: number; filepath?: string; strategy?: ChunkStrategy }> = [
  { name: "markdown", body: MARKDOWN, tokensPerChar: 1 / 4 },
  { name: "dense-recut", body: DENSE_LINE, tokensPerChar: 5 / 6 },
  { name: "dense-equal-starts", body: DENSE_LINE, tokensPerChar: 2_303 / 2_700 },
  { name: "unicode", body: UNICODE, tokensPerChar: 1 / 3 },
  { name: "typescript-ast", body: TYPESCRIPT, tokensPerChar: 1 / 4, filepath: "handlers.ts", strategy: "auto" },
  { name: "short", body: "# Title\n\nOne short paragraph.", tokensPerChar: 1 / 4 },
  { name: "heading-boundary", body: PROBE_H3, tokensPerChar: 1 / 4 },
];

async function chunkWith(body: string, tokensPerChar: number, filepath?: string, strategy?: ChunkStrategy) {
  setDefaultLlamaCpp(new CountingTokenLlm(tokensPerChar));
  try {
    return await chunkDocumentByTokens(body, undefined, undefined, undefined, filepath, strategy);
  } finally {
    setDefaultLlamaCpp(null);
  }
}

async function layoutSha256(): Promise<string> {
  const layouts: unknown[] = [];
  for (const { name, body, tokensPerChar, filepath, strategy } of GOLDEN_CORPUS) {
    const chunks = await chunkWith(body, tokensPerChar, filepath, strategy);
    layouts.push([name, chunks.map(chunk => [chunk.pos, chunk.text.length, sha256(chunk.text)])]);
  }
  return sha256(JSON.stringify(layouts));
}

let store: Store | null = null;
let dir: string | null = null;

async function openStore(): Promise<Store> {
  dir ??= await mkdtemp(join(tmpdir(), "qmd-embedding-fingerprint-"));
  store = createStore(join(dir, "index.sqlite"));
  return store;
}

afterEach(async () => {
  store?.close();
  store = null;
  if (dir) await rm(dir, { recursive: true, force: true });
  dir = null;
});

function seedDocument(s: Store, collection: string, hash: string, path: string, body: string): void {
  const now = new Date().toISOString();
  insertContent(s.db, hash, body, now);
  insertDocument(s.db, collection, path, hash, hash, now, now);
}

function storedRows(s: Store, hash: string) {
  return s.db.prepare(`
    SELECT seq, pos, total_chunks AS total, embed_fingerprint AS fingerprint
    FROM content_vectors
    WHERE hash = ?
    ORDER BY seq
  `).all(hash) as Array<{ seq: number; pos: number; total: number; fingerprint: string }>;
}

/** Each partition mapping row of the hash, with whether its vec0 row exists. */
function partitionPeers(s: Store, hash: string) {
  const rows = s.db.prepare(`SELECT id, seq FROM ${VEC_ROWS_TABLE} WHERE hash = ? ORDER BY seq`).all(hash) as Array<{ id: number; seq: number }>;
  const vector = s.db.prepare(`SELECT COUNT(*) AS count FROM ${VEC_TABLE} WHERE rowid = ?`);
  return rows.map(({ id, seq }) => ({ id, seq, vector: (vector.get(vecInteger(id)) as { count: number }).count }));
}

function fingerprintsInIndex(s: Store): string[] {
  return (s.db.prepare(`SELECT DISTINCT embed_fingerprint AS fingerprint FROM content_vectors ORDER BY fingerprint`).all() as Array<{ fingerprint: string }>)
    .map(row => row.fingerprint);
}

function inspect(s: Store) {
  return inspectVectorIndex(s.db, MODEL, getIndexEmbeddingFingerprint(s.db, MODEL), () => s.getHashesNeedingEmbedding(MODEL));
}

/** Rows an earlier fingerprint stored for `hash`: `chunks` sequences, each with a partition vector. */
function seedStaleRows(s: Store, hash: string, chunks: number): void {
  const now = new Date().toISOString();
  for (let seq = 0; seq < chunks; seq++) {
    s.insertEmbedding(hash, seq, seq * 100, new Float32Array([1, 0, 0]), MODEL, now, chunks, PRE_VERSION_FINGERPRINT);
  }
}

describe("chunker version", () => {
  test("pins the chunk layout of the current chunker version", async () => {
    expect({ chunkerVersion: CHUNKER_VERSION, layoutSha256: await layoutSha256() })
      .toEqual({ chunkerVersion: CHUNKER_VERSION, layoutSha256: LAYOUT_SHA256_BY_VERSION[CHUNKER_VERSION] });
  });

  test("differs from the fingerprint computed before the chunker was versioned", () => {
    expect(getEmbeddingFingerprint(MODEL)).toMatch(/^[0-9a-f]{6}$/);
    expect(getEmbeddingFingerprint(MODEL)).not.toBe(PRE_VERSION_FINGERPRINT);
  });

  test("gives each chunk strategy its own fingerprint, regex by default", () => {
    expect(getEmbeddingFingerprint(MODEL)).toBe(getEmbeddingFingerprint(MODEL, "regex"));
    expect(getEmbeddingFingerprint(MODEL, "auto")).not.toBe(getEmbeddingFingerprint(MODEL, "regex"));
  });
});

describe("index-wide chunk strategy", () => {
  test("reads regex for an index without a stored strategy", async () => {
    const s = await openStore();
    expect(getEmbeddingChunkStrategy(s.db)).toBe("regex");
    expect(getIndexEmbeddingFingerprint(s.db, MODEL)).toBe(getEmbeddingFingerprint(MODEL, "regex"));
  });

  test("an explicit strategy rebuilds every document, and an omitted one keeps it after reopening", async () => {
    let s = await openStore();
    seedDocument(s, "docs", "md", "notes.md", MARKDOWN);
    seedDocument(s, "docs", "ts", "handlers.ts", TYPESCRIPT);
    s.llm = new CountingTokenLlm(1 / 4);
    expect(await generateEmbeddings(s, { model: MODEL })).toMatchObject({ docsProcessed: 2, errors: 0 });
    expect(fingerprintsInIndex(s)).toEqual([getEmbeddingFingerprint(MODEL, "regex")]);

    const autoChunks = (await chunkWith(MARKDOWN, 1 / 4, "notes.md", "auto")).length
      + (await chunkWith(TYPESCRIPT, 1 / 4, "handlers.ts", "auto")).length;
    expect(await generateEmbeddings(s, { model: MODEL, chunkStrategy: "auto" }))
      .toMatchObject({ docsProcessed: 2, chunksEmbedded: autoChunks, errors: 0 });
    expect(fingerprintsInIndex(s)).toEqual([getEmbeddingFingerprint(MODEL, "auto")]);
    expect(getEmbeddingChunkStrategy(s.db)).toBe("auto");

    s.close();
    s = await openStore();
    s.llm = new CountingTokenLlm(1 / 4);
    expect(getEmbeddingChunkStrategy(s.db)).toBe("auto");
    expect(s.getHashesNeedingEmbedding(MODEL)).toBe(0);
    expect(await generateEmbeddings(s, { model: MODEL })).toMatchObject({ docsProcessed: 0, chunksEmbedded: 0 });
    expect(inspect(s)).toMatchObject({ needsEmbedding: 0, structurallyReady: true });
  });

  test("a strategy switch scoped to one collection leaves the other collection pending", async () => {
    const s = await openStore();
    seedDocument(s, "a", "in-a", "a.md", MARKDOWN);
    seedDocument(s, "b", "in-b", "b.md", OTHER_BODY);
    s.llm = new CountingTokenLlm(1 / 4);
    await generateEmbeddings(s, { model: MODEL });
    expect(s.getHashesNeedingEmbedding(MODEL)).toBe(0);

    await generateEmbeddings(s, { model: MODEL, collection: "a", chunkStrategy: "auto" });

    expect(getEmbeddingChunkStrategy(s.db)).toBe("auto");
    expect(getHashesNeedingEmbedding(s.db, "a", MODEL)).toBe(0);
    expect(getHashesNeedingEmbedding(s.db, "b", MODEL)).toBe(1);
    expect(getHashesNeedingEmbedding(s.db, "b", MODEL, "regex")).toBe(0);
    expect(storedRows(s, "in-b").map(row => row.fingerprint)).toEqual([getEmbeddingFingerprint(MODEL, "regex")]);
  });

  test("rejects an invalid explicit or stored strategy before any vector write", async () => {
    const s = await openStore();
    seedDocument(s, "docs", "doc", "doc.md", OTHER_BODY);
    s.llm = new CountingTokenLlm(1 / 4);

    await expect(generateEmbeddings(s, { model: MODEL, chunkStrategy: "semantic" as ChunkStrategy }))
      .rejects.toThrow('chunkStrategy must be "auto" or "regex" (got "semantic")');
    expect(getEmbeddingChunkStrategy(s.db)).toBe("regex");
    expect(storedRows(s, "doc")).toEqual([]);

    s.db.prepare(`INSERT INTO store_config (key, value) VALUES ('embedding_chunk_strategy', 'semantic')`).run();
    const stored = 'store_config embedding_chunk_strategy must be "auto" or "regex" (got "semantic")';
    await expect(generateEmbeddings(s, { model: MODEL })).rejects.toThrow(stored);
    expect(() => getHashesNeedingEmbedding(s.db, undefined, MODEL)).toThrow(stored);
    expect(storedRows(s, "doc")).toEqual([]);
  });
});

describe("embedding replacement", () => {
  test("one run replaces a longer stale layout, retires its partition peers and counts only current writes", async () => {
    const s = await openStore();
    s.ensureVecTable(3);
    seedDocument(s, "docs", "other", "other.md", OTHER_BODY);
    s.llm = new CountingTokenLlm(1 / 4);
    await generateEmbeddings(s, { model: MODEL });
    const otherRows = storedRows(s, "other");
    const otherPeers = partitionPeers(s, "other");

    seedDocument(s, "docs", "doc", "doc.md", MARKDOWN);
    const expected = await chunkWith(MARKDOWN, 1 / 4);
    seedStaleRows(s, "doc", expected.length + 2);
    expect(s.getHashesNeedingEmbedding(MODEL)).toBe(1);

    expect(await generateEmbeddings(s, { model: MODEL }))
      .toMatchObject({ docsProcessed: 1, chunksEmbedded: expected.length, errors: 0 });

    const current = getEmbeddingFingerprint(MODEL);
    expect(storedRows(s, "doc")).toEqual(
      expected.map((chunk, seq) => ({ seq, pos: chunk.pos, total: expected.length, fingerprint: current })),
    );
    expect(partitionPeers(s, "doc").map(({ seq, vector }) => ({ seq, vector })))
      .toEqual(expected.map((_, seq) => ({ seq, vector: 1 })));
    expect(storedRows(s, "other")).toEqual(otherRows);
    expect(partitionPeers(s, "other")).toEqual(otherPeers);
    expect(inspect(s)).toMatchObject({
      needsEmbedding: 0,
      inconsistentChunkLayouts: 0,
      missingRequiredPartitionRows: 0,
      inconsistentPeerRows: 0,
      structurallyReady: true,
    });
    expect(await generateEmbeddings(s, { model: MODEL })).toMatchObject({ docsProcessed: 0, chunksEmbedded: 0 });
  });

  test("a completed replacement interrupted before cleanup stays pending until a run retires its stale tail", async () => {
    const s = await openStore();
    s.ensureVecTable(3);
    seedDocument(s, "docs", "doc", "doc.md", MARKDOWN);
    s.llm = new CountingTokenLlm(1 / 4);
    const expected = await chunkWith(MARKDOWN, 1 / 4);
    seedStaleRows(s, "doc", expected.length + 2);
    const current = getEmbeddingFingerprint(MODEL);
    const currentRows = expected.map((chunk, seq) => ({ seq, pos: chunk.pos, total: expected.length, fingerprint: current }));
    // Every current row commits in the first embedding batch; progress reporting then stops the run before cleanup.
    const interrupted = new Error("interrupted after the current rows committed");
    await expect(generateEmbeddings(s, {
      model: MODEL,
      onProgress: (progress) => {
        if (progress.chunksEmbedded === expected.length) throw interrupted;
      },
    })).rejects.toBe(interrupted);

    expect(storedRows(s, "doc")).toEqual([
      ...currentRows,
      ...[expected.length, expected.length + 1].map(seq => ({ seq, pos: seq * 100, total: expected.length + 2, fingerprint: PRE_VERSION_FINGERPRINT })),
    ]);
    expect(partitionPeers(s, "doc")).toHaveLength(expected.length + 2);
    expect(getHashesNeedingEmbedding(s.db, undefined, MODEL)).toBe(1);
    expect(getHashesNeedingEmbedding(s.db, "docs", MODEL)).toBe(1);
    expect(getHashesForEmbedding(s.db, MODEL).map(row => row.hash)).toEqual(["doc"]);
    expect(inspect(s)).toMatchObject({ needsEmbedding: 1, structurallyReady: false });

    expect(await generateEmbeddings(s, { model: MODEL }))
      .toMatchObject({ docsProcessed: 1, chunksEmbedded: expected.length, errors: 0 });

    expect(storedRows(s, "doc")).toEqual(currentRows);
    expect(partitionPeers(s, "doc").map(({ seq, vector }) => ({ seq, vector })))
      .toEqual(expected.map((_, seq) => ({ seq, vector: 1 })));
    expect(inspect(s)).toMatchObject({
      needsEmbedding: 0,
      inconsistentChunkLayouts: 0,
      missingRequiredPartitionRows: 0,
      inconsistentPeerRows: 0,
      structurallyReady: true,
    });
    expect(await generateEmbeddings(s, { model: MODEL })).toMatchObject({ docsProcessed: 0, chunksEmbedded: 0 });
  });

  test("a permanently failed replacement stays pending with consistent peers and keeps unrelated hashes", async () => {
    const s = await openStore();
    s.ensureVecTable(3);
    seedDocument(s, "docs", "other", "other.md", OTHER_BODY);
    s.llm = new CountingTokenLlm(1 / 4);
    await generateEmbeddings(s, { model: MODEL });
    const otherRows = storedRows(s, "other");
    const otherPeers = partitionPeers(s, "other");

    const body = `${MARKDOWN}\nFAIL-MARKER closes the document.\n`;
    seedDocument(s, "docs", "doc", "doc.md", body);
    const expected = await chunkWith(body, 1 / 4);
    const failing = expected.filter(chunk => chunk.text.includes("FAIL-MARKER")).length;
    expect(failing).toBeGreaterThan(0);
    expect(failing).toBeLessThan(expected.length);
    seedStaleRows(s, "doc", expected.length + 2);
    s.llm = new CountingTokenLlm(1 / 4, text => text.includes("FAIL-MARKER"));

    const result = await generateEmbeddings(s, { model: MODEL });

    expect(result).toMatchObject({ docsProcessed: 1, chunksEmbedded: 0, errors: failing });
    expect(storedRows(s, "doc")).toEqual([]);
    expect(partitionPeers(s, "doc")).toEqual([]);
    expect(s.getHashesNeedingEmbedding(MODEL)).toBe(1);
    expect(storedRows(s, "other")).toEqual(otherRows);
    expect(partitionPeers(s, "other")).toEqual(otherPeers);
    expect(inspect(s)).toMatchObject({
      needsEmbedding: 1,
      inconsistentChunkLayouts: 0,
      missingRequiredPartitionRows: 0,
      inconsistentPeerRows: 0,
      structurallyReady: false,
    });
  });

  test("a replacement that writes no current row keeps the stale rows and stays pending", async () => {
    const s = await openStore();
    s.ensureVecTable(3);
    // The run's first chunk sizes the vector table, so a succeeding document comes first by path.
    seedDocument(s, "docs", "first", "a-first.md", OTHER_BODY);
    seedDocument(s, "docs", "doc", "doc.md", MARKDOWN);
    seedStaleRows(s, "doc", 3);
    const staleRows = storedRows(s, "doc");
    const stalePeers = partitionPeers(s, "doc");
    const docChunks = (await chunkWith(MARKDOWN, 1 / 4)).length;
    s.llm = new CountingTokenLlm(1 / 4, text => text.includes("Paragraph"));

    expect(await generateEmbeddings(s, { model: MODEL }))
      .toMatchObject({ docsProcessed: 2, chunksEmbedded: 1, errors: docChunks });

    expect(storedRows(s, "doc")).toEqual(staleRows);
    expect(partitionPeers(s, "doc")).toEqual(stalePeers);
    expect(s.getHashesNeedingEmbedding(MODEL)).toBe(1);
  });
});
