/**
 * The qmd CLI reads and records the index-wide chunk strategy without loading
 * a model: embed's zero-work preflight, an explicit strategy with no pending
 * work, and doctor's handling of unversioned vectors.
 */
import { afterEach, beforeEach, describe, expect, test } from "vitest";
import { spawnSync } from "node:child_process";
import { mkdir, mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import {
  createStore,
  getEmbeddingChunkStrategy,
  getEmbeddingFingerprint,
  insertContent,
  insertDocument,
  type ChunkStrategy,
} from "../src/store.js";

const projectRoot = join(dirname(fileURLToPath(import.meta.url)), "..");
const qmdScript = join(projectRoot, "src", "cli", "qmd.ts");
const isBunRuntime = typeof (globalThis as { Bun?: unknown }).Bun !== "undefined";
const runnerArgs = isBunRuntime ? [qmdScript] : [join(projectRoot, "node_modules", "tsx", "dist", "cli.mjs"), qmdScript];
const MODEL = "hf:test/cli-embedding-policy.gguf";
const NOW = "2026-01-01T00:00:00.000Z";

let dir: string;
let dbPath: string;
let configDir: string;

beforeEach(async () => {
  dir = await mkdtemp(join(tmpdir(), "qmd-cli-embedding-policy-"));
  dbPath = join(dir, "index.sqlite");
  configDir = join(dir, "config");
  await mkdir(join(dir, "a"), { recursive: true });
  await mkdir(join(dir, "b"), { recursive: true });
  await mkdir(configDir, { recursive: true });
  await writeFile(join(configDir, "index.yml"), [
    "collections:",
    "  a:",
    `    path: ${JSON.stringify(join(dir, "a"))}`,
    '    pattern: "**/*.md"',
    "  b:",
    `    path: ${JSON.stringify(join(dir, "b"))}`,
    '    pattern: "**/*.md"',
    "models:",
    `  embed: ${MODEL}`,
    "",
  ].join("\n"));
});

afterEach(async () => {
  await rm(dir, { recursive: true, force: true });
});

function runQmd(args: string[]) {
  return spawnSync(process.execPath, [...runnerArgs, ...args], {
    cwd: dir,
    encoding: "utf8",
    timeout: 120_000,
    env: { ...process.env, INDEX_PATH: dbPath, QMD_CONFIG_DIR: configDir, PWD: dir, QMD_DOCTOR_DEVICE_PROBE: "0" },
  });
}

/**
 * One single-chunk document per collection. Each document's vector carries
 * the fingerprint of its strategy, or the given fingerprint text; `stored`
 * is the recorded index-wide strategy (null records none).
 */
function seedIndex(vectors: Record<"a" | "b", ChunkStrategy | { fingerprint: string }>, stored: ChunkStrategy | null): void {
  const s = createStore(dbPath);
  try {
    s.ensureVecTable(2);
    for (const name of ["a", "b"] as const) {
      const hash = `hash-${name}`;
      insertContent(s.db, hash, `# ${name}\n\nBody of ${name}.`, NOW);
      insertDocument(s.db, name, `${name}.md`, name, hash, NOW, NOW);
      const vector = vectors[name];
      const fingerprint = typeof vector === "string" ? getEmbeddingFingerprint(MODEL, vector) : vector.fingerprint;
      s.insertEmbedding(hash, 0, 0, new Float32Array([1, 0]), MODEL, NOW, 1, fingerprint);
    }
    if (stored !== null) {
      s.db.prepare(`INSERT INTO store_config (key, value) VALUES ('embedding_chunk_strategy', ?)`).run(stored);
    }
  } finally {
    s.close();
  }
}

function readIndex(): { strategy: ChunkStrategy; fingerprints: Record<string, string> } {
  const s = createStore(dbPath);
  try {
    const rows = s.db.prepare(`SELECT hash, embed_fingerprint AS fingerprint FROM content_vectors ORDER BY hash`).all() as Array<{ hash: string; fingerprint: string }>;
    return { strategy: getEmbeddingChunkStrategy(s.db), fingerprints: Object.fromEntries(rows.map(row => [row.hash, row.fingerprint])) };
  } finally {
    s.close();
  }
}

describe("qmd embed chunk strategy", () => {
  test("an omitted strategy uses the stored policy and returns before any work", () => {
    seedIndex({ a: "auto", b: "auto" }, "auto");

    const result = runQmd(["embed"]);

    expect(result.status).toBe(0);
    expect(result.stdout).toContain("All content hashes already have embeddings.");
    expect(readIndex().strategy).toBe("auto");
  });

  test("an explicit strategy with no pending work records the policy without loading a model", () => {
    // An auto index after a regex switch scoped to collection a: b stays complete under auto.
    seedIndex({ a: "regex", b: "auto" }, "regex");

    const result = runQmd(["embed", "-c", "b", "--chunk-strategy", "auto"]);

    expect(result.status).toBe(0);
    expect(result.stdout).not.toContain("All content hashes already have embeddings.");
    expect(result.stdout).toContain("No non-empty documents to embed.");
    expect(readIndex()).toEqual({
      strategy: "auto",
      fingerprints: { "hash-a": getEmbeddingFingerprint(MODEL, "regex"), "hash-b": getEmbeddingFingerprint(MODEL, "auto") },
    });
  });
});

describe("qmd doctor unversioned vectors", () => {
  test("leaves empty-fingerprint vectors pending for qmd embed", () => {
    seedIndex({ a: { fingerprint: "" }, b: "regex" }, null);

    const result = runQmd(["doctor"]);

    expect(result.stdout).not.toContain("legacy fingerprint adoption");
    expect(result.stdout).toContain("1 active documents need embeddings");
    expect(readIndex().fingerprints).toEqual({ "hash-a": "", "hash-b": getEmbeddingFingerprint(MODEL, "regex") });
  });
});
