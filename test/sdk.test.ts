/**
 * sdk.test.ts - Unit tests for the QMD SDK (library mode)
 *
 * Tests the public API exposed via `@tobilu/qmd` (src/index.ts).
 * Uses inline config (no YAML files) to verify the SDK works self-contained.
 */

import { describe, test, expect, beforeAll, afterAll, beforeEach, afterEach, vi } from "vitest";
import { mkdtemp, writeFile, mkdir, rm, rename } from "node:fs/promises";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { existsSync, writeFileSync, mkdirSync, readFileSync, utimesSync } from "node:fs";
import YAML from "yaml";
import {
  createStore,
  type QMDStore,
  type CollectionConfig,
  type DocumentMetadata,
  type DocumentMetadataSource,
  type StoreOptions,
  type UpdateProgress,
  type SearchOptions,
  type LexSearchOptions,
  type VectorSearchOptions,
  type ExpandQueryOptions,
} from "../src/index.js";
import * as llmModule from "../src/llm.js";
import { setDefaultLlamaCpp } from "../src/llm.js";
import { VEC_COLLECTION_IDS_TABLE, VEC_ROWS_TABLE } from "../src/vec-layout.js";

// =============================================================================
// Test Helpers
// =============================================================================

let testDir: string;
let docsDir: string;
let notesDir: string;

beforeAll(async () => {
  testDir = await mkdtemp(join(tmpdir(), "qmd-sdk-test-"));
  docsDir = join(testDir, "docs");
  notesDir = join(testDir, "notes");

  // Create test directories with sample markdown files
  await mkdir(docsDir, { recursive: true });
  await mkdir(notesDir, { recursive: true });

  await writeFile(join(docsDir, "readme.md"), "# Getting Started\n\nThis is the getting started guide for the project.\n");
  await writeFile(join(docsDir, "auth.md"), "# Authentication\n\nAuthentication uses JWT tokens for session management.\nUsers log in with email and password.\n");
  await writeFile(join(docsDir, "api.md"), "# API Reference\n\n## Endpoints\n\n### POST /login\nAuthenticate a user.\n\n### GET /users\nList all users.\n");
  await writeFile(join(notesDir, "meeting-2025-01.md"), "# January Planning Meeting\n\nDiscussed Q1 roadmap and resource allocation.\n");
  await writeFile(join(notesDir, "meeting-2025-02.md"), "# February Standup\n\nReviewed sprint progress. Authentication feature is on track.\n");
  await writeFile(join(notesDir, "ideas.md"), "# Project Ideas\n\n- Build a search engine\n- Create a knowledge base\n- Implement vector search\n");
});

afterAll(async () => {
  try {
    await rm(testDir, { recursive: true, force: true });
  } catch {
    // Ignore cleanup errors
  }
});

function freshDbPath(): string {
  return join(testDir, `test-${Date.now()}-${Math.random().toString(36).slice(2)}.sqlite`);
}

// =============================================================================
// Constructor Tests
// =============================================================================

describe("createStore", () => {
  test("creates store with inline config", async () => {
    const store = await createStore({
      dbPath: freshDbPath(),
      config: {
        collections: {
          docs: { path: docsDir, pattern: "**/*.md" },
        },
      },
    });

    expect(store).toBeDefined();
    expect(store.dbPath).toBeTruthy();
    expect(store.internal).toBeDefined();
    await store.close();
  });

  test("creates store with YAML config file", async () => {
    const configPath = join(testDir, "test-config.yml");
    const config: CollectionConfig = {
      collections: {
        docs: { path: docsDir, pattern: "**/*.md" },
      },
    };
    writeFileSync(configPath, YAML.stringify(config));

    const store = await createStore({
      dbPath: freshDbPath(),
      configPath,
    });

    expect(store).toBeDefined();
    await store.close();
  });

  test("throws if dbPath is missing", async () => {
    await expect(
      createStore({ dbPath: "", config: { collections: {} } })
    ).rejects.toThrow("dbPath is required");
  });

  test("opens with just dbPath (DB-only mode)", async () => {
    const store = await createStore({ dbPath: freshDbPath() } as StoreOptions);
    expect(store).toBeDefined();
    // No collections yet — fresh DB
    const collections = await store.listCollections();
    expect(collections).toEqual([]);
    await store.close();
  });

  test("throws if both configPath and config are provided", async () => {
    await expect(
      createStore({
        dbPath: freshDbPath(),
        configPath: "/some/path.yml",
        config: { collections: {} },
      })
    ).rejects.toThrow("Provide either configPath or config, not both");
  });

  test("creates database file on disk", async () => {
    const dbPath = freshDbPath();
    const store = await createStore({
      dbPath,
      config: { collections: {} },
    });

    expect(existsSync(dbPath)).toBe(true);
    await store.close();
  });

  test("store.dbPath matches the provided path", async () => {
    const dbPath = freshDbPath();
    const store = await createStore({
      dbPath,
      config: { collections: {} },
    });

    expect(store.dbPath).toBe(dbPath);
    await store.close();
  });
});

// =============================================================================
// Collection Management Tests
// =============================================================================

describe("collection management", () => {
  let store: QMDStore;

  beforeEach(async () => {
    store = await createStore({
      dbPath: freshDbPath(),
      config: { collections: {} },
    });
  });

  afterEach(async () => {
    await store.close();
  });

  test("addCollection adds a collection to inline config", async () => {
    await store.addCollection("docs", { path: docsDir, pattern: "**/*.md" });

    const collections = await store.listCollections();
    const names = collections.map(c => c.name);
    expect(names).toContain("docs");
  });

  test("addCollection with default pattern", async () => {
    await store.addCollection("notes", { path: notesDir });

    const collections = await store.listCollections();
    expect(collections.find(c => c.name === "notes")).toBeDefined();
  });

  test("removeCollection removes existing collection", async () => {
    await store.addCollection("docs", { path: docsDir, pattern: "**/*.md" });
    const removed = await store.removeCollection("docs");

    expect(removed).toBe(true);
    const collections = await store.listCollections();
    expect(collections.map(c => c.name)).not.toContain("docs");
  });

  test("removeCollection returns false for non-existent collection", async () => {
    const removed = await store.removeCollection("nonexistent");
    expect(removed).toBe(false);
  });

  test("renameCollection renames a collection", async () => {
    await store.addCollection("old-name", { path: docsDir, pattern: "**/*.md" });
    const renamed = await store.renameCollection("old-name", "new-name");

    expect(renamed).toBe(true);
    const names = (await store.listCollections()).map(c => c.name);
    expect(names).toContain("new-name");
    expect(names).not.toContain("old-name");
  });

  test("renameCollection returns false for non-existent source", async () => {
    const renamed = await store.renameCollection("nonexistent", "new-name");
    expect(renamed).toBe(false);
  });

  test("renameCollection throws if target exists", async () => {
    await store.addCollection("a", { path: docsDir, pattern: "**/*.md" });
    await store.addCollection("b", { path: notesDir, pattern: "**/*.md" });

    await expect(store.renameCollection("a", "b")).rejects.toThrow("already exists");
  });

  test("listCollections returns empty array for empty config", async () => {
    const collections = await store.listCollections();
    expect(collections).toEqual([]);
  });

  test("multiple collections can be added", async () => {
    await store.addCollection("docs", { path: docsDir, pattern: "**/*.md" });
    await store.addCollection("notes", { path: notesDir, pattern: "**/*.md" });

    const names = (await store.listCollections()).map(c => c.name);
    expect(names).toContain("docs");
    expect(names).toContain("notes");
    expect(names).toHaveLength(2);
  });
});

// =============================================================================
// Context Management Tests
// =============================================================================

describe("context management", () => {
  let store: QMDStore;

  beforeEach(async () => {
    store = await createStore({
      dbPath: freshDbPath(),
      config: {
        collections: {
          docs: { path: docsDir, pattern: "**/*.md" },
          notes: { path: notesDir, pattern: "**/*.md" },
        },
      },
    });
  });

  afterEach(async () => {
    await store.close();
  });

  test("addContext adds context to a collection path", async () => {
    const added = await store.addContext("docs", "/auth", "Authentication docs");
    expect(added).toBe(true);

    const contexts = await store.listContexts();
    expect(contexts).toContainEqual({
      collection: "docs",
      path: "/auth",
      context: "Authentication docs",
    });
  });

  test("addContext returns false for non-existent collection", async () => {
    const added = await store.addContext("nonexistent", "/path", "Some context");
    expect(added).toBe(false);
  });

  test("removeContext removes existing context", async () => {
    await store.addContext("docs", "/auth", "Authentication docs");
    const removed = await store.removeContext("docs", "/auth");

    expect(removed).toBe(true);
    const contexts = await store.listContexts();
    expect(contexts.find(c => c.path === "/auth")).toBeUndefined();
  });

  test("removeContext returns false for non-existent context", async () => {
    const removed = await store.removeContext("docs", "/nonexistent");
    expect(removed).toBe(false);
  });

  test("setGlobalContext sets and retrieves global context", async () => {
    await store.setGlobalContext("Global knowledge base");
    const global = await store.getGlobalContext();

    expect(global).toBe("Global knowledge base");
  });

  test("setGlobalContext with undefined clears it", async () => {
    await store.setGlobalContext("Some context");
    await store.setGlobalContext(undefined);
    const global = await store.getGlobalContext();

    expect(global).toBeUndefined();
  });

  test("listContexts includes global context", async () => {
    await store.setGlobalContext("Global context");
    const contexts = await store.listContexts();

    expect(contexts).toContainEqual({
      collection: "*",
      path: "/",
      context: "Global context",
    });
  });

  test("listContexts returns contexts across multiple collections", async () => {
    await store.addContext("docs", "/", "Documentation");
    await store.addContext("notes", "/", "Personal notes");

    const contexts = await store.listContexts();
    expect(contexts.filter(c => c.path === "/")).toHaveLength(2);
  });

  test("multiple contexts on same collection", async () => {
    await store.addContext("docs", "/auth", "Auth docs");
    await store.addContext("docs", "/api", "API docs");

    const contexts = (await store.listContexts()).filter(c => c.collection === "docs");
    expect(contexts).toHaveLength(2);
    expect(contexts.map(c => c.path).sort()).toEqual(["/api", "/auth"]);
  });

  test("addContext overwrites existing context for same path", async () => {
    await store.addContext("docs", "/auth", "Old context");
    await store.addContext("docs", "/auth", "New context");

    const contexts = (await store.listContexts()).filter(c => c.path === "/auth");
    expect(contexts).toHaveLength(1);
    expect(contexts[0]!.context).toBe("New context");
  });
});

// =============================================================================
// Inline Config Isolation Tests
// =============================================================================

describe("inline config isolation", () => {
  test("inline config does not write any files to disk", async () => {
    const configDir = join(testDir, "should-not-exist");
    const store = await createStore({
      dbPath: freshDbPath(),
      config: {
        collections: {
          docs: { path: docsDir, pattern: "**/*.md" },
        },
      },
    });

    await store.addCollection("notes", { path: notesDir, pattern: "**/*.md" });
    await store.addContext("docs", "/", "Documentation");

    expect(existsSync(configDir)).toBe(false);
    await store.close();
  });

  test("inline config mutations persist within session", async () => {
    const store = await createStore({
      dbPath: freshDbPath(),
      config: { collections: {} },
    });

    await store.addCollection("docs", { path: docsDir, pattern: "**/*.md" });
    await store.addContext("docs", "/", "My docs");

    // Verify the mutations are visible
    const collections = await store.listCollections();
    expect(collections.map(c => c.name)).toContain("docs");

    const contexts = await store.listContexts();
    expect(contexts).toContainEqual({
      collection: "docs",
      path: "/",
      context: "My docs",
    });

    await store.close();
  });

  test("two stores with different inline configs are independent", async () => {
    const store1 = await createStore({
      dbPath: freshDbPath(),
      config: {
        collections: {
          docs: { path: docsDir, pattern: "**/*.md" },
        },
      },
    });

    // Close first store (resets config source)
    await store1.close();

    const store2 = await createStore({
      dbPath: freshDbPath(),
      config: {
        collections: {
          notes: { path: notesDir, pattern: "**/*.md" },
        },
      },
    });

    const names = (await store2.listCollections()).map(c => c.name);
    expect(names).toContain("notes");
    expect(names).not.toContain("docs");

    await store2.close();
  });
});

// =============================================================================
// YAML Config File Tests
// =============================================================================

describe("YAML config file mode", () => {
  test("loads collections from YAML file", async () => {
    const configPath = join(testDir, `config-${Date.now()}.yml`);
    const config: CollectionConfig = {
      collections: {
        docs: { path: docsDir, pattern: "**/*.md" },
        notes: { path: notesDir, pattern: "**/*.md" },
      },
    };
    writeFileSync(configPath, YAML.stringify(config));

    const store = await createStore({ dbPath: freshDbPath(), configPath });
    const names = (await store.listCollections()).map(c => c.name);

    expect(names).toContain("docs");
    expect(names).toContain("notes");
    await store.close();
  });

  test("addCollection persists to YAML file", async () => {
    const configPath = join(testDir, `config-persist-${Date.now()}.yml`);
    writeFileSync(configPath, YAML.stringify({ collections: {} }));

    const store = await createStore({ dbPath: freshDbPath(), configPath });
    await store.addCollection("newcol", { path: docsDir, pattern: "**/*.md" });
    await store.close();

    // Read the YAML file directly and verify
    const raw = readFileSync(configPath, "utf-8");
    const parsed = YAML.parse(raw) as CollectionConfig;
    expect(parsed.collections).toHaveProperty("newcol");
    expect(parsed.collections.newcol!.path).toBe(docsDir);
  });

  test("context persists to YAML file", async () => {
    const configPath = join(testDir, `config-ctx-${Date.now()}.yml`);
    writeFileSync(configPath, YAML.stringify({
      collections: { docs: { path: docsDir, pattern: "**/*.md" } },
    }));

    const store = await createStore({ dbPath: freshDbPath(), configPath });
    await store.addContext("docs", "/api", "API documentation");
    await store.close();

    const raw = readFileSync(configPath, "utf-8");
    const parsed = YAML.parse(raw) as CollectionConfig;
    expect(parsed.collections.docs!.context).toEqual({ "/api": "API documentation" });
  });

  test("non-existent config file returns empty collections", async () => {
    const configPath = join(testDir, "nonexistent-config.yml");
    const store = await createStore({ dbPath: freshDbPath(), configPath });
    const collections = await store.listCollections();

    expect(collections).toEqual([]);
    await store.close();
  });
});

// =============================================================================
// Search Tests (BM25 - no LLM needed)
// =============================================================================

describe("searchLex (BM25)", () => {
  let store: QMDStore;
  let dbPath: string;

  beforeAll(async () => {
    dbPath = join(testDir, "search-test.sqlite");
    store = await createStore({
      dbPath,
      config: {
        collections: {
          docs: { path: docsDir, pattern: "**/*.md" },
          notes: { path: notesDir, pattern: "**/*.md" },
        },
      },
    });

    // Index documents manually using internal store
    const now = new Date().toISOString();
    const { internal } = store;
    const fs = require("fs");

    // Index docs collection
    for (const file of ["readme.md", "auth.md", "api.md"]) {
      const fullPath = join(docsDir, file);
      const content = fs.readFileSync(fullPath, "utf-8");
      const hash = require("crypto").createHash("sha256").update(content).digest("hex");
      const title = content.match(/^#\s+(.+)/m)?.[1] || file;

      internal.insertContent(hash, content, now);
      internal.insertDocument("docs", `qmd://docs/${file}`, title, hash, now, now);
    }

    // Index notes collection
    for (const file of ["meeting-2025-01.md", "meeting-2025-02.md", "ideas.md"]) {
      const fullPath = join(notesDir, file);
      const content = fs.readFileSync(fullPath, "utf-8");
      const hash = require("crypto").createHash("sha256").update(content).digest("hex");
      const title = content.match(/^#\s+(.+)/m)?.[1] || file;

      internal.insertContent(hash, content, now);
      internal.insertDocument("notes", `qmd://notes/${file}`, title, hash, now, now);
    }
  });

  afterAll(async () => {
    await store.close();
  });

  test("searchLex returns results for matching query", async () => {
    const results = await store.searchLex("authentication");
    expect(results.length).toBeGreaterThan(0);
  });

  test("searchLex results have expected shape", async () => {
    const results = await store.searchLex("authentication");
    expect(results.length).toBeGreaterThan(0);

    const result = results[0]!;
    expect(result).toHaveProperty("filepath");
    expect(result).toHaveProperty("score");
    expect(result).toHaveProperty("title");
    expect(result).toHaveProperty("docid");
    expect(result).toHaveProperty("collectionName");
    expect(typeof result.score).toBe("number");
    expect(result.score).toBeGreaterThan(0);
  });

  test("searchLex respects limit option", async () => {
    const results = await store.searchLex("meeting", { limit: 1 });
    expect(results.length).toBeLessThanOrEqual(1);
  });

  test("searchLex with collection filter", async () => {
    const results = await store.searchLex("authentication", { collection: "notes" });
    for (const r of results) {
      expect(r.collectionName).toBe("notes");
    }
  });

  test("searchLex returns empty for non-matching query", async () => {
    const results = await store.searchLex("xyznonexistentterm123");
    expect(results).toHaveLength(0);
  });

  test("searchLex finds documents across collections", async () => {
    const results = await store.searchLex("authentication", { limit: 10 });
    const collections = new Set(results.map(r => r.collectionName));
    // Auth appears in both docs/auth.md and notes/meeting-2025-02.md
    expect(collections.size).toBeGreaterThanOrEqual(1);
  });
});

// =============================================================================
// Unified search() API Tests
// =============================================================================

describe("search (unified API)", () => {
  let store: QMDStore;

  beforeAll(async () => {
    store = await createStore({
      dbPath: join(testDir, "unified-search-test.sqlite"),
      config: {
        collections: {
          docs: { path: docsDir, pattern: "**/*.md" },
          notes: { path: notesDir, pattern: "**/*.md" },
        },
      },
    });
    await store.update();
  });

  afterAll(async () => {
    await store.close();
  });

  test("search() requires query or queries", async () => {
    await expect(store.search({} as SearchOptions)).rejects.toThrow("requires either 'query' or 'queries'");
  });

  test("search() with pre-expanded queries and rerank:false", async () => {
    const results = await store.search({
      queries: [
        { type: "lex", query: "authentication JWT" },
        { type: "lex", query: "login session" },
      ],
      rerank: false,
    });
    expect(results.length).toBeGreaterThan(0);
  });

  test("search() forwards candidateLimit to structured search", async () => {
    const results = await store.search({
      queries: [
        { type: "lex", query: "authentication" },
        { type: "lex", query: "meeting" },
      ],
      limit: 5,
      candidateLimit: 1,
      rerank: false,
    });

    expect(results).toHaveLength(1);
  });

  // Tests below use search({ query: ... }) which triggers LLM query expansion
  describe.skipIf(!!process.env.CI)("with LLM query expansion", () => {
    test("search() with query and rerank:false returns results", async () => {
      const results = await store.search({ query: "authentication", rerank: false });
      expect(results.length).toBeGreaterThan(0);
      expect(results[0]).toHaveProperty("file");
      expect(results[0]).toHaveProperty("score");
      expect(results[0]).toHaveProperty("title");
      expect(results[0]).toHaveProperty("bestChunk");
      expect(results[0]).toHaveProperty("docid");
    }, 90000);

    test("search() with intent and rerank:false returns results", async () => {
      const results = await store.search({
        query: "meeting",
        intent: "quarterly planning and roadmap",
        rerank: false,
      });
      expect(results.length).toBeGreaterThan(0);
    }, 60000);

    test("search() with collection filter", async () => {
      const results = await store.search({
        query: "authentication",
        collection: "docs",
        rerank: false,
      });
      for (const r of results) {
        expect(r.file).toMatch(/^qmd:\/\/docs\//);
      }
    });

    test("search() with collections filter", async () => {
      const results = await store.search({
        query: "authentication",
        collections: ["docs"],
        rerank: false,
      });
      for (const r of results) {
        expect(r.file).toMatch(/^qmd:\/\/docs\//);
      }
    });

    test("search() with limit", async () => {
      const results = await store.search({ query: "meeting", limit: 1, rerank: false });
      expect(results.length).toBeLessThanOrEqual(1);
    });

    test("search() returns empty for non-matching query", async () => {
      const results = await store.search({ query: "xyznonexistentterm123", rerank: false });
      expect(results).toHaveLength(0);
    });
  });
});

// =============================================================================
// Document Retrieval Tests
// =============================================================================

describe("get and multiGet", () => {
  let store: QMDStore;

  beforeAll(async () => {
    store = await createStore({
      dbPath: join(testDir, "get-test.sqlite"),
      config: {
        collections: {
          docs: { path: docsDir, pattern: "**/*.md" },
        },
      },
    });

    // Index documents
    const now = new Date().toISOString();
    const { internal } = store;
    const fs = require("fs");

    for (const file of ["readme.md", "auth.md", "api.md"]) {
      const fullPath = join(docsDir, file);
      const content = fs.readFileSync(fullPath, "utf-8");
      const hash = require("crypto").createHash("sha256").update(content).digest("hex");
      const title = content.match(/^#\s+(.+)/m)?.[1] || file;

      internal.insertContent(hash, content, now);
      internal.insertDocument("docs", `qmd://docs/${file}`, title, hash, now, now);
    }
  });

  afterAll(async () => {
    await store.close();
  });

  test("get retrieves a document by path", async () => {
    const result = await store.get("qmd://docs/auth.md");

    expect("error" in result).toBe(false);
    if (!("error" in result)) {
      expect(result.title).toBe("Authentication");
      expect(result.collectionName).toBe("docs");
    }
  });

  test("get with includeBody returns body content", async () => {
    const result = await store.get("qmd://docs/auth.md", { includeBody: true });

    if (!("error" in result)) {
      expect(result.body).toBeDefined();
      expect(result.body).toContain("JWT tokens");
    }
  });

  test("get returns not_found for missing document", async () => {
    const result = await store.get("qmd://docs/nonexistent.md");

    expect("error" in result).toBe(true);
    if ("error" in result) {
      expect(result.error).toBe("not_found");
    }
  });

  test("get by docid", async () => {
    // First get a document to find its docid
    const doc = await store.get("qmd://docs/readme.md");
    if (!("error" in doc)) {
      const byDocid = await store.get(`#${doc.docid}`);
      expect("error" in byDocid).toBe(false);
      if (!("error" in byDocid)) {
        expect(byDocid.docid).toBe(doc.docid);
      }
    }
  });

  test("multiGet retrieves a document by docid", async () => {
    const doc = await store.get("qmd://docs/readme.md");
    if (!("error" in doc)) {
      const { docs, errors } = await store.multiGet(`#${doc.docid}`, { includeBody: true });
      expect(errors).toHaveLength(0);
      expect(docs).toHaveLength(1);
      expect(docs[0]!.doc.docid).toBe(doc.docid);
      expect(docs[0]!.skipped).toBe(false);
      if (!docs[0]!.skipped) {
        expect(docs[0]!.doc.body).toContain("getting started guide");
      }
    }
  });

  test("multiGet retrieves multiple documents", async () => {
    const { docs, errors } = await store.multiGet("qmd://docs/*.md");
    expect(docs.length).toBeGreaterThan(0);
  });
});

// =============================================================================
// Index Health Tests
// =============================================================================

describe("index health", () => {
  let store: QMDStore;

  beforeEach(async () => {
    store = await createStore({
      dbPath: freshDbPath(),
      config: {
        collections: {
          docs: { path: docsDir, pattern: "**/*.md" },
        },
      },
    });
  });

  afterEach(async () => {
    await store.close();
  });

  test("getStatus returns valid structure", async () => {
    const status = await store.getStatus();

    expect(status).toHaveProperty("totalDocuments");
    expect(status).toHaveProperty("needsEmbedding");
    expect(status).toHaveProperty("hasVectorIndex");
    expect(status).toHaveProperty("collections");
    expect(typeof status.totalDocuments).toBe("number");
  });

  test("getIndexHealth returns valid structure", async () => {
    const health = await store.getIndexHealth();

    expect(health).toHaveProperty("needsEmbedding");
    expect(health).toHaveProperty("totalDocs");
    expect(typeof health.needsEmbedding).toBe("number");
    expect(typeof health.totalDocs).toBe("number");
  });

  test("fresh store has zero documents", async () => {
    const status = await store.getStatus();
    expect(status.totalDocuments).toBe(0);
  });
});

// =============================================================================
// Update Tests
// =============================================================================

describe("update", () => {
  test("indexes files and returns correct stats", async () => {
    const store = await createStore({
      dbPath: freshDbPath(),
      config: {
        collections: {
          docs: { path: docsDir, pattern: "**/*.md" },
        },
      },
    });

    const result = await store.update();

    expect(result.collections).toBe(1);
    expect(result.indexed).toBe(3); // readme.md, auth.md, api.md
    expect(result.updated).toBe(0);
    expect(result.unchanged).toBe(0);
    expect(result.removed).toBe(0);
    expect(result.skipped).toBe(0);
    expect(result.staleVectorsRemoved).toBe(0);
    expect(result.vectorsCopied).toBe(0);
    expect(typeof result.needsEmbedding).toBe("number");
    expect(result.metadataErrors).toBe(0);

    await store.close();
  });

  test("second update shows unchanged files", async () => {
    const store = await createStore({
      dbPath: freshDbPath(),
      config: {
        collections: {
          docs: { path: docsDir, pattern: "**/*.md" },
        },
      },
    });

    await store.update();
    const result = await store.update();

    expect(result.indexed).toBe(0);
    expect(result.unchanged).toBe(3);

    await store.close();
  });

  test("update with onProgress callback fires", async () => {
    const store = await createStore({
      dbPath: freshDbPath(),
      config: {
        collections: {
          docs: { path: docsDir, pattern: "**/*.md" },
        },
      },
    });

    const progress: UpdateProgress[] = [];
    await store.update({
      onProgress: (info) => progress.push(info),
    });

    expect(progress.length).toBeGreaterThan(0);
    expect(progress[0]!.collection).toBe("docs");
    expect(progress[0]!.current).toBeGreaterThanOrEqual(1);
    expect(progress[0]!.total).toBe(3);

    await store.close();
  });

  test("update with collection filter", async () => {
    const store = await createStore({
      dbPath: freshDbPath(),
      config: {
        collections: {
          docs: { path: docsDir, pattern: "**/*.md" },
          notes: { path: notesDir, pattern: "**/*.md" },
        },
      },
    });

    const result = await store.update({ collections: ["docs"] });

    expect(result.collections).toBe(1);
    expect(result.indexed).toBe(3); // Only docs

    await store.close();
  });

  test("update multiple collections", async () => {
    const store = await createStore({
      dbPath: freshDbPath(),
      config: {
        collections: {
          docs: { path: docsDir, pattern: "**/*.md" },
          notes: { path: notesDir, pattern: "**/*.md" },
        },
      },
    });

    const result = await store.update();

    expect(result.collections).toBe(2);
    expect(result.indexed).toBe(6); // 3 docs + 3 notes

    await store.close();
  });

  test("documents are searchable after update", async () => {
    const store = await createStore({
      dbPath: freshDbPath(),
      config: {
        collections: {
          docs: { path: docsDir, pattern: "**/*.md" },
        },
      },
    });

    await store.update();

    const results = await store.searchLex("authentication");
    expect(results.length).toBeGreaterThan(0);

    await store.close();
  });
});

describe("update with a metadata source", () => {
  type SourceDocument = Parameters<DocumentMetadataSource>[0];
  type MetadataState = { source: string; metadata_json: string; extraction_error: string | null; extracted_at: string };

  // Older than the racy-sync window, so the stat fast path trusts these files.
  const settledTime = new Date(Date.now() - 60 * 60 * 1000);

  async function settledCollection(files: Record<string, string>): Promise<string> {
    const dir = join(testDir, `metadata-source-${Date.now()}-${Math.random().toString(36).slice(2)}`);
    for (const [path, content] of Object.entries(files)) {
      const filePath = join(dir, path);
      await mkdir(dirname(filePath), { recursive: true });
      await writeFile(filePath, content);
      utimesSync(filePath, settledTime, settledTime);
    }
    return dir;
  }

  function storeOver(collections: Record<string, string>): Promise<QMDStore> {
    return createStore({
      dbPath: freshDbPath(),
      config: {
        collections: Object.fromEntries(Object.entries(collections).map(([name, path]) => [name, { path, pattern: "**/*.md" }])),
      },
    });
  }

  function metadataState(store: QMDStore, path: string): MetadataState | undefined {
    return store.internal.db.prepare(`
      SELECT dm.source, dm.metadata_json, dm.extraction_error, dm.extracted_at
      FROM documents d JOIN document_metadata dm ON dm.document_id = d.id
      WHERE d.active = 1 AND d.path = ?
    `).get(path) as MetadataState | undefined;
  }

  function documentHash(store: QMDStore, path: string): string | undefined {
    const row = store.internal.db.prepare(`SELECT hash FROM documents WHERE active = 1 AND path = ?`).get(path) as { hash: string } | undefined;
    return row?.hash;
  }

  /** A JavaScript caller's source, whose answer the TypeScript signature cannot check. */
  function untypedSource(answer: (document: SourceDocument) => unknown): DocumentMetadataSource {
    return answer as DocumentMetadataSource;
  }

  const waitForNewTimestamp = () => new Promise(resolve => setTimeout(resolve, 5));

  test("one unchanged file moves from frontmatter to external, changed, empty, and back to frontmatter", async () => {
    const dir = await settledCollection({
      "note.md": "---\nqmd:\n  metadata:\n    author: human\n---\n\n# Note\n\nRetry twice.\n",
    });
    const store = await storeOver({ notes: dir });
    const asked: SourceDocument[] = [];
    const answer = (metadata: DocumentMetadata): DocumentMetadataSource => (document) => {
      asked.push(document);
      return metadata;
    };
    const assistant = { author: "assistant", provider: "anthropic" };

    try {
      expect(await store.update()).toMatchObject({ indexed: 1, metadataErrors: 0 });
      expect(metadataState(store, "note.md")).toMatchObject({ source: "frontmatter", metadata_json: '{"author":"human"}', extraction_error: null });
      const hash = documentHash(store, "note.md");

      // The source replaces the native YAML claim, which stays searchable text.
      await store.update({ metadata: answer(assistant) });
      const external = metadataState(store, "note.md");
      expect(external).toMatchObject({ source: "external", metadata_json: JSON.stringify(assistant), extraction_error: null });
      expect(asked).toEqual([{ collection: "notes", path: "note.md", hash }]);
      expect(await store.searchLex("retry", { filter: { field: "author", operator: "eq", value: "assistant" } })).toHaveLength(1);
      expect(await store.searchLex("retry", { filter: { field: "author", operator: "eq", value: "human" } })).toEqual([]);
      expect(await store.searchLex("human")).toHaveLength(1);
      expect((await store.listMetadata()).keys.map(key => key.key)).toEqual(["author", "provider"]);

      // The same answer keeps the row and its extraction time.
      await waitForNewTimestamp();
      await store.update({ metadata: answer({ ...assistant }) });
      expect(metadataState(store, "note.md")).toEqual(external);

      // A changed answer through the hash fast path: mtime moved, bytes did not.
      const touched = new Date(settledTime.getTime() - 60_000);
      utimesSync(join(dir, "note.md"), touched, touched);
      expect(await store.update({ metadata: answer({ author: "human", reviewed: true }) }))
        .toMatchObject({ unchanged: 1, updated: 0, metadataErrors: 0 });
      expect(asked.at(-1)).toEqual({ collection: "notes", path: "note.md", hash });
      expect(metadataState(store, "note.md")).toMatchObject({ source: "external", metadata_json: '{"author":"human","reviewed":true}' });

      // An empty answer clears every key.
      await store.update({ metadata: answer({}) });
      expect(metadataState(store, "note.md")).toMatchObject({ source: "external", metadata_json: "{}", extraction_error: null });
      expect(await store.searchLex("retry", { filter: { field: "author", operator: "exists", value: false } })).toHaveLength(1);

      // A default update re-extracts the frontmatter and takes the row back.
      expect(await store.update()).toMatchObject({ unchanged: 1, metadataErrors: 0 });
      const frontmatter = metadataState(store, "note.md");
      expect(frontmatter).toMatchObject({ source: "frontmatter", metadata_json: '{"author":"human"}', extraction_error: null });
      await waitForNewTimestamp();
      await store.update();
      expect(metadataState(store, "note.md")).toEqual(frontmatter);
    } finally {
      await store.close();
    }
  });

  test("asks once per admitted nonblank file with its path and indexed hash, reading no bytes on the stat fast path", async () => {
    const dir = await settledCollection({
      "a.md": "# A\n\nalpha body\n",
      "sub/b.md": "# B\n\nbeta body\n",
      "blank.md": " \n\t\n",
    });
    const store = await storeOver({ notes: dir });
    const asked: SourceDocument[] = [];
    const recordPath: DocumentMetadataSource = (document) => {
      asked.push(document);
      return { path: document.path };
    };

    try {
      expect(await store.update({ metadata: recordPath })).toMatchObject({ indexed: 2, metadataErrors: 0 });
      expect(asked.map(document => document.path).sort()).toEqual(["a.md", "sub/b.md"]);
      for (const document of asked) {
        expect(document).toEqual({ collection: "notes", path: document.path, hash: documentHash(store, document.path) });
      }

      // A same-size rewrite that keeps mtime passes the stat fast path, so QMD
      // asks with the indexed hash and never sees the new bytes.
      const indexedHash = documentHash(store, "a.md");
      writeFileSync(join(dir, "a.md"), "# A\n\nALPHA BODY\n");
      utimesSync(join(dir, "a.md"), settledTime, settledTime);
      asked.length = 0;
      expect(await store.update({ metadata: recordPath })).toMatchObject({ unchanged: 2, updated: 0 });
      expect(asked.find(document => document.path === "a.md")?.hash).toBe(indexedHash);
      expect(documentHash(store, "a.md")).toBe(indexedHash);
    } finally {
      await store.close();
    }
  });

  test("invalid answers keep the scan going, sum across collections, and stay out of filtered search", async () => {
    const notes = await settledCollection({
      "valid.md": "# Valid\n\nshared term\n",
      "then.md": "# Then\n\nshared term\n",
      "missing.md": "# Missing\n\nshared term\n",
    });
    const archive = await settledCollection({
      "map.md": "# Map\n\nshared term\n",
      "huge.md": "# Huge\n\nshared term\n",
    });
    const answers: Record<string, unknown> = {
      "valid.md": { status: "ok" },
      "then.md": { then: "x" },
      "missing.md": undefined,
      "map.md": new Map([["status", "ok"]]),
      "huge.md": { notes: Array.from({ length: 128 }, (_, index) => String(index).padEnd(1024, "x")) },
    };
    const answerByPath = untypedSource(document => answers[document.path]);
    const store = await storeOver({ notes, archive });

    try {
      expect(await store.update({ metadata: answerByPath })).toMatchObject({ indexed: 5, metadataErrors: 3 });
      for (const path of ["missing.md", "map.md", "huge.md"]) {
        expect({ path, state: metadataState(store, path) }).toEqual({
          path,
          state: expect.objectContaining({ source: "external", metadata_json: "{}", extraction_error: expect.stringMatching(/^metadata source: /) }),
        });
      }
      expect(metadataState(store, "then.md")).toMatchObject({ metadata_json: '{"then":"x"}', extraction_error: null });

      expect(await store.searchLex("shared", { limit: 10 })).toHaveLength(5);
      const withoutStatus = await store.searchLex("shared", { limit: 10, filter: { field: "status", operator: "exists", value: false } });
      expect(withoutStatus.map(result => result.displayPath)).toEqual(["notes/then.md"]);
      expect((await store.getStatus()).pendingMetadata).toBe(3);

      // Every update asks again, so unchanged invalid answers count again.
      expect((await store.update({ metadata: answerByPath })).metadataErrors).toBe(3);
    } finally {
      await store.close();
    }
  });

  test("a throwing source rejects with collection, path and cause, keeping only committed batches", async () => {
    const dir = await settledCollection(Object.fromEntries(
      Array.from({ length: 501 }, (_, index) => [`f${String(index).padStart(3, "0")}.md`, `# File ${index}\n\nbody ${index}\n`]),
    ));
    const store = await storeOver({ notes: dir });
    const asked: string[] = [];
    const failure = new Error("source map has no row");
    // 500 files fill a scan batch at most, so the 501st ask runs after at least one commit.
    const failOn501st: DocumentMetadataSource = (document) => {
      asked.push(document.path);
      if (asked.length === 501) throw failure;
      return { position: asked.length };
    };

    try {
      const error = await store.update({ metadata: failOn501st }).then(() => undefined, (err: unknown) => err);
      expect(error).toBeInstanceOf(Error);
      expect((error as Error).message).toBe(`Metadata source failed for notes/${asked[500]}: source map has no row`);
      expect((error as Error).cause).toBe(failure);

      // Committed batches hold a prefix of the ask order, each with its whole external answer.
      const rows = store.internal.db.prepare(`
        SELECT d.path, dm.source, dm.metadata_json, dm.extraction_error
        FROM documents d LEFT JOIN document_metadata dm ON dm.document_id = d.id
        WHERE d.active = 1
      `).all() as { path: string; source: string | null; metadata_json: string | null; extraction_error: string | null }[];
      expect(rows.length).toBeGreaterThan(0);
      expect(rows.length).toBeLessThan(501);
      expect(new Set(rows.map(row => row.path))).toEqual(new Set(asked.slice(0, rows.length)));
      for (const row of rows) {
        expect(row).toEqual({
          path: row.path,
          source: "external",
          metadata_json: JSON.stringify({ position: asked.indexOf(row.path) + 1 }),
          extraction_error: null,
        });
      }

      const completed = await store.update({ metadata: () => ({ position: 0 }) });
      expect(completed).toMatchObject({ indexed: 501 - rows.length, unchanged: rows.length, metadataErrors: 0 });
      expect((await store.getStatus()).pendingMetadata).toBe(0);
    } finally {
      await store.close();
    }
  }, 60_000);

  test("a thenable answer rejects the update and its abandoned promise stays handled", async () => {
    const dir = await settledCollection({ "a.md": "# A\n\nalpha\n" });
    const store = await storeOver({ notes: dir });
    const unhandled: unknown[] = [];
    const recordUnhandled = (reason: unknown) => { unhandled.push(reason); };
    process.on("unhandledRejection", recordUnhandled);

    try {
      await expect(store.update({ metadata: untypedSource(async () => ({ status: "ok" })) })).rejects.toThrow(
        "Metadata source returned a thenable for notes/a.md. DocumentMetadataSource must return metadata synchronously.",
      );
      await expect(store.update({ metadata: untypedSource(() => Promise.reject(new Error("late rejection"))) })).rejects.toThrow(TypeError);
      await new Promise(resolve => setImmediate(resolve));
      expect(unhandled).toEqual([]);
      expect(documentHash(store, "a.md")).toBeUndefined();
    } finally {
      process.off("unhandledRejection", recordUnhandled);
      await store.close();
    }
  });

  test("update reads the metadata option once and rejects a non-function before any write", async () => {
    const notes = await settledCollection({ "a.md": "# A\n\nalpha\n" });
    const archive = await settledCollection({ "b.md": "# B\n\nbeta\n" });
    const store = await storeOver({ notes, archive });

    try {
      await expect(store.update({ metadata: "author" as never })).rejects.toThrow("update() metadata must be a function, received string");
      expect((await store.getStatus()).totalDocuments).toBe(0);

      let reads = 0;
      let assigned: DocumentMetadataSource = () => {
        options.metadata = () => ({ origin: "replacement" });
        return { origin: "original" };
      };
      const options = {
        get metadata(): DocumentMetadataSource {
          reads++;
          return assigned;
        },
        set metadata(source: DocumentMetadataSource) {
          assigned = source;
        },
      };
      await store.update(options);
      expect(reads).toBe(1);
      expect(metadataState(store, "a.md")?.metadata_json).toBe('{"origin":"original"}');
      expect(metadataState(store, "b.md")?.metadata_json).toBe('{"origin":"original"}');
    } finally {
      await store.close();
    }
  });
});

describe("embed", () => {
  function createFakeTokenizer() {
    return {
      async tokenize(text: string) {
        return new Array(Math.max(1, Math.ceil(text.length / 16))).fill(1);
      },
      async detokenize(tokens: readonly number[]) {
        return "x".repeat(tokens.length * 16);
      },
    };
  }

  function createFakeEmbedLlm() {
    const embedBatchCalls: string[][] = [];
    return {
      ...createFakeTokenizer(),
      embedBatchCalls,
      async embed(_text: string) {
        return { embedding: [0.1, 0.2, 0.3], model: "fake-embed" };
      },
      async embedBatch(texts: string[]) {
        embedBatchCalls.push([...texts]);
        return texts.map((_text, index) => ({
          embedding: [index + 1, index + 2, index + 3],
          model: "fake-embed",
        }));
      },
    };
  }

  test("store.embed forwards batch limit options", async () => {
    const store = await createStore({
      dbPath: freshDbPath(),
      config: {
        collections: {
          docs: { path: docsDir, pattern: "**/*.md" },
        },
      },
    });

    const fakeLlm = createFakeEmbedLlm();
    setDefaultLlamaCpp(createFakeTokenizer() as any);
    store.internal.llm = fakeLlm as any;

    try {
      await store.update();
      const result = await store.embed({
        maxDocsPerBatch: 1,
        maxBatchBytes: 1024 * 1024,
      });

      expect(fakeLlm.embedBatchCalls).toHaveLength(3);
      expect(fakeLlm.embedBatchCalls.map(call => call.length)).toEqual([1, 1, 1]);
      expect(result.docsProcessed).toBe(3);
      expect(result.chunksEmbedded).toBe(3);
    } finally {
      setDefaultLlamaCpp(null);
      await store.close();
    }
  });

  test("store.embed forwards maxDurationMs to the embedding session", async () => {
    const store = await createStore({
      dbPath: freshDbPath(),
      config: {
        collections: {
          docs: { path: docsDir, pattern: "**/*.md" },
        },
      },
    });

    const fakeLlm = createFakeEmbedLlm();
    const sessionSpy = vi.spyOn(llmModule, "withLLMSessionForLlm");
    setDefaultLlamaCpp(createFakeTokenizer() as any);
    store.internal.llm = fakeLlm as any;

    try {
      await store.update();
      const rows: Array<[maxDurationMs: number | undefined, maxDuration: number]> = [
        [undefined, 30 * 60 * 1000],
        [60 * 60 * 1000, 60 * 60 * 1000],
        [0, 0],
      ];
      for (const [maxDurationMs, maxDuration] of rows) {
        // force re-embeds every document, so each row opens a session over pending work.
        const result = await store.embed({ force: true, maxDurationMs });

        expect(result.docsProcessed).toBe(3);
        expect(sessionSpy).toHaveBeenLastCalledWith(
          fakeLlm,
          expect.any(Function),
          expect.objectContaining({ maxDuration, name: "generateEmbeddings" }),
        );
      }
    } finally {
      sessionSpy.mockRestore();
      setDefaultLlamaCpp(null);
      await store.close();
    }
  });

  test("store.embed scopes pending documents to the requested collection", async () => {
    const store = await createStore({
      dbPath: freshDbPath(),
      config: {
        collections: {
          docs: { path: docsDir, pattern: "**/*.md" },
          notes: { path: notesDir, pattern: "**/*.md" },
        },
      },
    });

    const fakeLlm = createFakeEmbedLlm();
    setDefaultLlamaCpp(createFakeTokenizer() as any);
    store.internal.llm = fakeLlm as any;

    try {
      await store.update();
      const result = await store.embed({ collection: "docs" });

      const vectorCounts = store.internal.db.prepare(`
        SELECT d.collection, COUNT(DISTINCT v.hash) AS count
        FROM documents d
        LEFT JOIN content_vectors v ON v.hash = d.hash AND v.seq = 0
        WHERE d.active = 1
        GROUP BY d.collection
        ORDER BY d.collection
      `).all() as Array<{ collection: string; count: number }>;

      expect(result.docsProcessed).toBe(3);
      expect(result.chunksEmbedded).toBe(3);
      expect(vectorCounts).toEqual([
        { collection: "docs", count: 3 },
        { collection: "notes", count: 0 },
      ]);
    } finally {
      setDefaultLlamaCpp(null);
      await store.close();
    }
  });

  test("store.embed with force only clears the requested collection", async () => {
    const store = await createStore({
      dbPath: freshDbPath(),
      config: {
        collections: {
          docs: { path: docsDir, pattern: "**/*.md" },
          notes: { path: notesDir, pattern: "**/*.md" },
        },
      },
    });

    const fakeLlm = createFakeEmbedLlm();
    setDefaultLlamaCpp(createFakeTokenizer() as any);
    store.internal.llm = fakeLlm as any;

    const vectorCounts = () => store.internal.db.prepare(`
      SELECT d.collection, COUNT(DISTINCT v.hash) AS count
      FROM documents d
      LEFT JOIN content_vectors v ON v.hash = d.hash AND v.seq = 0
      WHERE d.active = 1
      GROUP BY d.collection
      ORDER BY d.collection
    `).all() as Array<{ collection: string; count: number }>;

    try {
      await store.update();
      await store.embed();
      expect(vectorCounts()).toEqual([
        { collection: "docs", count: 3 },
        { collection: "notes", count: 3 },
      ]);

      const result = await store.embed({ force: true, collection: "docs" });

      expect(result.docsProcessed).toBe(3);
      expect(result.chunksEmbedded).toBe(3);
      expect(vectorCounts()).toEqual([
        { collection: "docs", count: 3 },
        { collection: "notes", count: 3 },
      ]);
    } finally {
      setDefaultLlamaCpp(null);
      await store.close();
    }
  });

  test("store.update drops stale vector rows and copies rows into a collection that gained an embedded hash", async () => {
    const leftDir = join(testDir, `vector-rows-left-${Date.now()}`);
    const rightDir = join(testDir, `vector-rows-right-${Date.now()}`);
    await mkdir(leftDir, { recursive: true });
    await mkdir(rightDir, { recursive: true });
    const shared = "# Shared\n\nEmbedded once, then joins a second collection.\n";
    await writeFile(join(leftDir, "shared.md"), shared);
    await writeFile(join(rightDir, "gone.md"), "# Gone\n\nDisappears before the second update.\n");

    const store = await createStore({
      dbPath: freshDbPath(),
      config: {
        collections: {
          left: { path: leftDir, pattern: "**/*.md" },
          right: { path: rightDir, pattern: "**/*.md" },
        },
      },
    });
    setDefaultLlamaCpp(createFakeTokenizer() as any);
    store.internal.llm = createFakeEmbedLlm() as any;

    const partitionsOf = (path: string): string[] => {
      const doc = store.internal.db.prepare(`SELECT hash FROM documents WHERE path = ? LIMIT 1`).get(path) as { hash: string };
      const rows = store.internal.db.prepare(`
        SELECT ci.name AS collection FROM ${VEC_ROWS_TABLE} vr
        JOIN ${VEC_COLLECTION_IDS_TABLE} ci ON ci.id = vr.collection_id
        WHERE vr.hash = ?
        ORDER BY ci.name
      `).all(doc.hash) as Array<{ collection: string }>;
      return rows.map((row) => row.collection);
    };

    try {
      await store.update();
      await store.embed();
      expect(partitionsOf("shared.md")).toEqual(["left"]);
      expect(partitionsOf("gone.md")).toEqual(["right"]);

      await writeFile(join(rightDir, "shared.md"), shared);
      await rm(join(rightDir, "gone.md"));
      const result = await store.update();

      expect(result.removed).toBe(1);
      expect(result.staleVectorsRemoved).toBe(1);
      expect(result.vectorsCopied).toBe(1);
      expect(result.needsEmbedding).toBe(0);
      expect(partitionsOf("shared.md")).toEqual(["left", "right"]);
      expect(partitionsOf("gone.md")).toEqual([]);
    } finally {
      setDefaultLlamaCpp(null);
      await store.close();
    }
  });

  test("store.update keeps the vectors of a document moved to another collection", async () => {
    const fromDir = join(testDir, `vector-move-from-${Date.now()}`);
    const toDir = join(testDir, `vector-move-to-${Date.now()}`);
    await mkdir(fromDir, { recursive: true });
    await mkdir(toDir, { recursive: true });
    await writeFile(join(fromDir, "moved.md"), "# Moved\n\nEmbedded in one collection, then moved to another.\n");

    const store = await createStore({
      dbPath: freshDbPath(),
      config: {
        collections: {
          from: { path: fromDir, pattern: "**/*.md" },
          to: { path: toDir, pattern: "**/*.md" },
        },
      },
    });
    setDefaultLlamaCpp(createFakeTokenizer() as any);
    store.internal.llm = createFakeEmbedLlm() as any;

    const db = store.internal.db;
    const hashOf = () =>
      (db.prepare(`SELECT hash FROM documents WHERE path = 'moved.md' LIMIT 1`).get() as { hash: string }).hash;
    const chunkCount = (hash: string) =>
      (db.prepare(`SELECT COUNT(*) AS n FROM content_vectors WHERE hash = ?`).get(hash) as { n: number }).n;
    const partitionsOf = (hash: string): string[] =>
      (db.prepare(`
        SELECT ci.name AS collection FROM ${VEC_ROWS_TABLE} vr
        JOIN ${VEC_COLLECTION_IDS_TABLE} ci ON ci.id = vr.collection_id
        WHERE vr.hash = ?
        ORDER BY ci.name
      `).all(hash) as Array<{ collection: string }>).map((row) => row.collection);

    try {
      await store.update();
      await store.embed();
      const hash = hashOf();
      const embeddedChunks = chunkCount(hash);
      expect(embeddedChunks).toBeGreaterThan(0);
      expect(partitionsOf(hash)).toEqual(["from"]);

      await rename(join(fromDir, "moved.md"), join(toDir, "moved.md"));
      const result = await store.update();

      expect(result.removed).toBe(1);
      expect(result.indexed).toBe(1);
      expect(chunkCount(hash)).toBe(embeddedChunks);
      expect(partitionsOf(hash)).toEqual(["to"]);
      expect(result.vectorsCopied).toBe(1);
      expect(result.staleVectorsRemoved).toBe(1);
      expect(result.needsEmbedding).toBe(0);
    } finally {
      setDefaultLlamaCpp(null);
      await store.close();
    }
  });

  test("store.embed rejects invalid batch limits", async () => {
    const store = await createStore({
      dbPath: freshDbPath(),
      config: { collections: {} },
    });

    try {
      await expect(store.embed({ maxDocsPerBatch: 0 })).rejects.toThrow("maxDocsPerBatch");
      await expect(store.embed({ maxBatchBytes: 0 })).rejects.toThrow("maxBatchBytes");
    } finally {
      setDefaultLlamaCpp(null);
      await store.close();
    }
  });
});

// =============================================================================
// Lifecycle Tests
// =============================================================================

describe("lifecycle", () => {
  test("close() is async and does not throw", async () => {
    const store = await createStore({
      dbPath: freshDbPath(),
      config: { collections: {} },
    });

    // close() should return a promise
    const result = store.close();
    expect(result).toBeInstanceOf(Promise);
    await result;
  });

  test("close() makes subsequent operations throw", async () => {
    const store = await createStore({
      dbPath: freshDbPath(),
      config: { collections: {} },
    });

    await store.close();

    // Database operations should fail after close
    await expect(store.getStatus()).rejects.toThrow();
  });

  test("multiple stores can coexist with different databases", async () => {
    const store1 = await createStore({
      dbPath: freshDbPath(),
      config: {
        collections: {
          docs: { path: docsDir, pattern: "**/*.md" },
        },
      },
    });

    // Note: since config source is module-level, we close store1 first
    await store1.close();

    const store2 = await createStore({
      dbPath: freshDbPath(),
      config: {
        collections: {
          notes: { path: notesDir, pattern: "**/*.md" },
        },
      },
    });

    const names = (await store2.listCollections()).map(c => c.name);
    expect(names).toContain("notes");
    expect(names).not.toContain("docs");

    await store2.close();
  });
});

// =============================================================================
// Config Initialization Tests
// =============================================================================

describe("config initialization", () => {
  test("inline config with global_context is preserved", async () => {
    const store = await createStore({
      dbPath: freshDbPath(),
      config: {
        global_context: "System knowledge base",
        collections: {
          docs: { path: docsDir, pattern: "**/*.md" },
        },
      },
    });

    const global = await store.getGlobalContext();
    expect(global).toBe("System knowledge base");
    await store.close();
  });

  test("inline config with pre-existing contexts is preserved", async () => {
    const store = await createStore({
      dbPath: freshDbPath(),
      config: {
        collections: {
          docs: {
            path: docsDir,
            pattern: "**/*.md",
            context: { "/auth": "Authentication docs" },
          },
        },
      },
    });

    const contexts = await store.listContexts();
    expect(contexts).toContainEqual({
      collection: "docs",
      path: "/auth",
      context: "Authentication docs",
    });
    await store.close();
  });

  test("inline config with empty collections object works", async () => {
    const store = await createStore({
      dbPath: freshDbPath(),
      config: { collections: {} },
    });

    expect(await store.listCollections()).toEqual([]);
    expect(await store.listContexts()).toEqual([]);
    await store.close();
  });

  test("inline config with multiple collection options", async () => {
    const store = await createStore({
      dbPath: freshDbPath(),
      config: {
        collections: {
          docs: {
            path: docsDir,
            pattern: "**/*.md",
            ignore: ["drafts/**"],
            includeByDefault: true,
          },
          notes: {
            path: notesDir,
            pattern: "**/*.md",
            includeByDefault: false,
          },
        },
      },
    });

    const collections = await store.listCollections();
    expect(collections).toHaveLength(2);
    await store.close();
  });
});

// =============================================================================
// Type Export Tests (compile-time checks, runtime verification)
// =============================================================================

describe("type exports", () => {
  test("StoreOptions type is usable", () => {
    const opts: StoreOptions = {
      dbPath: "/tmp/test.sqlite",
      config: { collections: {} },
    };
    expect(opts.dbPath).toBe("/tmp/test.sqlite");
  });

  test("CollectionConfig type is usable", () => {
    const config: CollectionConfig = {
      global_context: "test",
      collections: {
        test: { path: "/tmp", pattern: "**/*.md" },
      },
    };
    expect(config.collections).toHaveProperty("test");
  });

  test("QMDStore type exposes expected methods", async () => {
    const store = await createStore({
      dbPath: freshDbPath(),
      config: { collections: {} },
    });

    // Verify all methods exist
    expect(typeof store.search).toBe("function");
    expect(typeof store.searchLex).toBe("function");
    expect(typeof store.searchVector).toBe("function");
    expect(typeof store.expandQuery).toBe("function");
    expect(typeof store.get).toBe("function");
    expect(typeof store.multiGet).toBe("function");
    expect(typeof store.addCollection).toBe("function");
    expect(typeof store.removeCollection).toBe("function");
    expect(typeof store.renameCollection).toBe("function");
    expect(typeof store.listCollections).toBe("function");
    expect(typeof store.addContext).toBe("function");
    expect(typeof store.removeContext).toBe("function");
    expect(typeof store.setGlobalContext).toBe("function");
    expect(typeof store.getGlobalContext).toBe("function");
    expect(typeof store.listContexts).toBe("function");
    expect(typeof store.getStatus).toBe("function");
    expect(typeof store.getIndexHealth).toBe("function");
    expect(typeof store.update).toBe("function");
    expect(typeof store.embed).toBe("function");
    expect(typeof store.close).toBe("function");

    await store.close();
  });
});

// =============================================================================
// DB-Only Mode Tests (self-contained store)
// =============================================================================

describe("DB-only mode", () => {
  test("reopen store with just dbPath after config+update session", async () => {
    const dbPath = freshDbPath();

    // Session 1: create store with config, update, close
    const store1 = await createStore({
      dbPath,
      config: {
        collections: {
          docs: { path: docsDir, pattern: "**/*.md" },
          notes: { path: notesDir, pattern: "**/*.md" },
        },
        global_context: "Test knowledge base",
      },
    });

    await store1.update();

    // Verify documents indexed
    const status1 = await store1.getStatus();
    expect(status1.totalDocuments).toBe(6);
    await store1.close();

    // Session 2: reopen with just dbPath — no config
    const store2 = await createStore({ dbPath } as StoreOptions);

    // Collections should still be available
    const collections = await store2.listCollections();
    expect(collections.map(c => c.name).sort()).toEqual(["docs", "notes"]);

    // Search should still work
    const results = await store2.searchLex("authentication");
    expect(results.length).toBeGreaterThan(0);

    // Global context should still be available
    const globalCtx = await store2.getGlobalContext();
    expect(globalCtx).toBe("Test knowledge base");

    // Contexts from collections should persist
    const status2 = await store2.getStatus();
    expect(status2.totalDocuments).toBe(6);

    await store2.close();
  });

  test("config sync populates store_collections table", async () => {
    const dbPath = freshDbPath();
    const store = await createStore({
      dbPath,
      config: {
        collections: {
          docs: {
            path: docsDir,
            pattern: "**/*.md",
            context: { "/auth": "Auth documentation" },
          },
        },
      },
    });

    // Verify collections are in the DB via listCollections
    const collections = await store.listCollections();
    expect(collections).toHaveLength(1);
    expect(collections[0]!.name).toBe("docs");
    expect(collections[0]!.pwd).toBe(docsDir);

    // Verify contexts are accessible
    const contexts = await store.listContexts();
    expect(contexts).toContainEqual({
      collection: "docs",
      path: "/auth",
      context: "Auth documentation",
    });

    await store.close();
  });

  test("config hash skip: second init with same config skips sync", async () => {
    const dbPath = freshDbPath();
    const config = {
      collections: {
        docs: { path: docsDir, pattern: "**/*.md" },
      },
    };

    // First init — syncs config
    const store1 = await createStore({ dbPath, config });
    await store1.close();

    // Second init with same config — should skip sync (no-op, but should not error)
    const store2 = await createStore({ dbPath, config });
    const collections = await store2.listCollections();
    expect(collections).toHaveLength(1);
    expect(collections[0]!.name).toBe("docs");
    await store2.close();
  });

  test("DB-only mode supports collection mutations", async () => {
    const dbPath = freshDbPath();

    // Session 1: create with config
    const store1 = await createStore({
      dbPath,
      config: {
        collections: {
          docs: { path: docsDir, pattern: "**/*.md" },
        },
      },
    });
    await store1.close();

    // Session 2: reopen DB-only, add a collection
    const store2 = await createStore({ dbPath } as StoreOptions);
    await store2.addCollection("notes", { path: notesDir, pattern: "**/*.md" });

    const names = (await store2.listCollections()).map(c => c.name).sort();
    expect(names).toEqual(["docs", "notes"]);

    await store2.close();

    // Session 3: reopen DB-only again, verify both collections persist
    const store3 = await createStore({ dbPath } as StoreOptions);
    const names3 = (await store3.listCollections()).map(c => c.name).sort();
    expect(names3).toEqual(["docs", "notes"]);
    await store3.close();
  });

  test("DB-only mode supports context mutations", async () => {
    const dbPath = freshDbPath();

    // Session 1: create with config
    const store1 = await createStore({
      dbPath,
      config: {
        collections: {
          docs: { path: docsDir, pattern: "**/*.md" },
        },
      },
    });
    await store1.addContext("docs", "/api", "API docs");
    await store1.setGlobalContext("Global context");
    await store1.close();

    // Session 2: reopen DB-only
    const store2 = await createStore({ dbPath } as StoreOptions);

    const contexts = await store2.listContexts();
    expect(contexts).toContainEqual({
      collection: "docs",
      path: "/api",
      context: "API docs",
    });
    expect(contexts).toContainEqual({
      collection: "*",
      path: "/",
      context: "Global context",
    });

    await store2.close();
  });
});
