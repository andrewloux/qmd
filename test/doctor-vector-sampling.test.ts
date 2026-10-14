import { afterEach, beforeEach, describe, expect, test } from "vitest";
import { spawnSync } from "node:child_process";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { isBun } from "../src/db.js";
import { chunkDocumentByTokens, createStore, extractTitle, formatDocForEmbedding, getEmbeddingVectorSamples, type Store } from "../src/store.js";
import { checkEmbeddingVectorSamples } from "../src/cli/qmd.js";
import { setDefaultLlamaCpp, LlamaCpp } from "../src/llm.js";

const projectRoot = join(dirname(fileURLToPath(import.meta.url)), "..");
let store: Store;

beforeEach(() => { store = createStore(":memory:"); });
afterEach(() => { setDefaultLlamaCpp(null); store.close(); });

function addDocument(hash: string, path: string, active: boolean = true): void {
  store.insertContent(hash, `Body for ${hash}`, "2026-01-01");
  store.insertDocument("test", path, hash, hash, "2026-01-01", "2026-01-01");
  if (!active) store.deactivateDocument("test", path);
}

function addChunk(hash: string, seq: number, model: string = "model", fingerprint: string = "current", pos: number = 0): void {
  store.db.prepare(`
    INSERT INTO content_vectors (hash, seq, model, embed_fingerprint, pos, embedded_at)
    VALUES (?, ?, ?, ?, ?, '2026-01-01')
  `).run(hash, seq, model, fingerprint, pos);
}

describe("doctor vector sampling", () => {
  async function addStoredPassage(seq: number, positionExists: boolean = true, vectorMatches: boolean = true): Promise<void> {
    let expectedText = "";
    class PassageLlm extends LlamaCpp {
      async tokenize(text: string) { return new Array(Math.ceil(text.length / 16)).fill(1); }
      async embed(text: string) {
        return { embedding: text === expectedText ? [1, 0] : [0, 1], model: "model" };
      }
    }
    setDefaultLlamaCpp(new PassageLlm());
    const body = "First passage. ".repeat(300) + "\n\nSecond passage. ".repeat(300);
    const chunks = await chunkDocumentByTokens(body);
    const passage = chunks[1]!;
    expect(passage.pos).toBeGreaterThan(0);
    expectedText = formatDocForEmbedding(passage.text, extractTitle(body, "sample.md"), "model");
    store.insertContent("sample", body, "2026-01-01");
    store.insertDocument("test", "sample.md", "Sample", "sample", "2026-01-01", "2026-01-01");
    store.ensureVecTable(2);
    store.insertEmbedding("sample", seq, positionExists ? passage.pos : 1,
      new Float32Array(vectorMatches ? [1, 0] : [0, 1]), "model", "2026-01-01", 1, "current");
  }

  test("reports a mismatch when the stored passage sits at another sequence", async () => {
    await addStoredPassage(0);
    const result = await checkEmbeddingVectorSamples(store.db, "model", "current");
    expect(result.ok).toBe(false);
    expect(result.details).toContain("chunk no longer exists");
  });

  test("reports a mismatch when the stored sequence is beyond today's chunk count", async () => {
    await addStoredPassage(100);
    const result = await checkEmbeddingVectorSamples(store.db, "model", "current");
    expect(result.ok).toBe(false);
    expect(result.details).toContain("chunk no longer exists");
  });

  test("checks the chunk at the stored sequence when two chunks share a start", async () => {
    let expectedText = "";
    class EqualStartsLlm extends LlamaCpp {
      // 2,303 tokens per 2,700 characters re-cuts each first-pass chunk into starts 0, 765, 1,530, 2,295, 2,295, ...
      async tokenize(text: string) { return new Array(Math.ceil(text.length * 2_303 / 2_700)).fill(1); }
      async embed(text: string) {
        return { embedding: text === expectedText ? [1, 0] : [0, 1], model: "model" };
      }
    }
    setDefaultLlamaCpp(new EqualStartsLlm());
    const body = Array.from({ length: 1_200 }, (_, index) => `w${String(index).padStart(4, "0")}`).join("");
    const chunks = await chunkDocumentByTokens(body);
    expect(chunks[3]!.pos).toBe(chunks[4]!.pos);
    expect(chunks[3]!.text).not.toBe(chunks[4]!.text);
    expectedText = formatDocForEmbedding(chunks[4]!.text, extractTitle(body, "dense.md"), "model");
    store.insertContent("dense", body, "2026-01-01");
    store.insertDocument("test", "dense.md", "Dense", "dense", "2026-01-01", "2026-01-01");
    store.ensureVecTable(2);
    store.insertEmbedding("dense", 4, chunks[4]!.pos, new Float32Array([1, 0]), "model", "2026-01-01", chunks.length, "current");

    expect(await checkEmbeddingVectorSamples(store.db, "model", "current"))
      .toEqual({ ok: true, details: "1 sampled chunk reproduce stored vectors" });
  });

  test("reconstructs chunks with the index's stored chunk strategy", async () => {
    let expectedText = "";
    class QuarterTokenLlm extends LlamaCpp {
      async tokenize(text: string) { return new Array(Math.ceil(text.length / 4)).fill(1); }
      async embed(text: string) {
        return { embedding: text === expectedText ? [1, 0] : [0, 1], model: "model" };
      }
    }
    setDefaultLlamaCpp(new QuarterTokenLlm());
    const body = Array.from({ length: 60 }, (_, index) => [
      `export function handler${index}(input: string): string {`,
      "  const value = input.trim();",
      `  return value + "-${index}";`,
      "}",
      "",
    ].join("\n")).join("\n");
    const auto = await chunkDocumentByTokens(body, undefined, undefined, undefined, "handlers.ts", "auto");
    const regex = await chunkDocumentByTokens(body, undefined, undefined, undefined, "handlers.ts", "regex");
    const seq = auto.findIndex((chunk, index) => chunk.pos !== regex[index]?.pos || chunk.text !== regex[index]?.text);
    expect(seq).toBeGreaterThanOrEqual(0);
    expectedText = formatDocForEmbedding(auto[seq]!.text, extractTitle(body, "handlers.ts"), "model");
    store.insertContent("code", body, "2026-01-01");
    store.insertDocument("test", "handlers.ts", "Handlers", "code", "2026-01-01", "2026-01-01");
    store.db.prepare(`INSERT INTO store_config (key, value) VALUES ('embedding_chunk_strategy', 'auto')`).run();
    store.ensureVecTable(2);
    store.insertEmbedding("code", seq, auto[seq]!.pos, new Float32Array([1, 0]), "model", "2026-01-01", auto.length, "current");

    expect(await checkEmbeddingVectorSamples(store.db, "model", "current"))
      .toEqual({ ok: true, details: "1 sampled chunk reproduce stored vectors" });
  });

  test("still rejects a changed vector at the matching position", async () => {
    await addStoredPassage(1, true, false);
    const result = await checkEmbeddingVectorSamples(store.db, "model", "current");
    expect(result.ok).toBe(false);
    expect(result.details).toContain("stored vector distance");
  });

  test("does not substitute the sequence number when the stored position disappeared", async () => {
    await addStoredPassage(1, false);
    const result = await checkEmbeddingVectorSamples(store.db, "model", "current");
    expect(result.ok).toBe(false);
    expect(result.details).toContain("chunk no longer exists");
  });

  test("samples each eligible chunk once and uses the first active path", () => {
    addDocument("shared", "z.md");
    addDocument("shared", "b.md");
    addDocument("shared", "a.md", false);
    addChunk("shared", 0);
    addChunk("shared", 1, "model", "current", 1234);
    addDocument("other", "other.md");
    addChunk("other", 0);
    addDocument("inactive", "inactive.md", false);
    addChunk("inactive", 0);
    addDocument("stale", "stale.md");
    addChunk("stale", 0, "model", "old");
    addDocument("different-model", "different.md");
    addChunk("different-model", 0, "other-model");
    addChunk("orphan", 0);

    const samples = getEmbeddingVectorSamples(store.db, "model", "current", 20);
    expect(samples.sort((a, b) => a.hash.localeCompare(b.hash) || a.seq - b.seq)).toEqual([
      { hash: "other", seq: 0, pos: 0, body: "Body for other", path: "other.md" },
      { hash: "shared", seq: 0, pos: 0, body: "Body for shared", path: "b.md" },
      { hash: "shared", seq: 1, pos: 1234, body: "Body for shared", path: "b.md" },
    ]);
    expect(getEmbeddingVectorSamples(store.db, "model", "current", 2)).toHaveLength(2);
    expect(getEmbeddingVectorSamples(store.db, "model", "current", 0)).toEqual([]);
    expect(getEmbeddingVectorSamples(store.db, "missing", "current")).toEqual([]);
  });

  test("samples large documents with duplicate paths within a 32 MiB SQLite budget", () => {
    // SQLite's hard heap limit is process-wide and cannot be raised again.
    const worker = join(projectRoot, "test", "_helpers", "doctor-vector-sample-worker.ts");
    const args = isBun ? [worker] : [join(projectRoot, "node_modules", "tsx", "dist", "cli.mjs"), worker];
    const result = spawnSync(process.execPath, args, { encoding: "utf8", timeout: 20_000 });
    expect(result.error).toBeUndefined();
    expect(result.stderr).toBe("");
    expect(result.status).toBe(0);
    expect(JSON.parse(result.stdout)).toEqual({ samples: 3, distinctChunks: 3, bodiesComplete: true });
  });

  test("the doctor check verifies large documents with duplicate paths within a 32 MiB SQLite budget", () => {
    // SQLite's hard heap limit is process-wide and cannot be raised again.
    const worker = join(projectRoot, "test", "_helpers", "doctor-vector-check-worker.ts");
    const args = isBun ? [worker] : [join(projectRoot, "node_modules", "tsx", "dist", "cli.mjs"), worker];
    const result = spawnSync(process.execPath, args, { encoding: "utf8", timeout: 60_000 });
    expect(result.error).toBeUndefined();
    expect(result.stderr).toBe("");
    expect(result.status).toBe(0);
    const out = JSON.parse(result.stdout);
    expect(out).toMatchObject({ ok: true, details: "3 sampled chunks reproduce stored vectors" });
    // The budget only binds where SQLite tracks memory; on Bun under Linux it must.
    if (isBun && process.platform === "linux") expect(out.limitBinds).toBe(true);
  });
});
