/**
 * metadata.test.ts - Frontmatter metadata extraction and normalization.
 */

import { describe, test, expect } from "vitest";
import {
  askMetadataSource,
  extractDocumentMetadata,
  normalizeSourceMetadata,
  METADATA_EXTRACTION_VERSION,
  METADATA_LIMITS,
  type DocumentMetadata,
  type DocumentMetadataSource,
} from "../src/metadata.js";

function buildDoc(frontmatterYaml: string, body: string = "# Title\n\nBody text.\n"): string {
  return `---\n${frontmatterYaml}---\n\n${body}`;
}

describe("extractDocumentMetadata", () => {
  test("document without frontmatter yields empty metadata", () => {
    const extraction = extractDocumentMetadata("# Title\n\nBody.\n", "doc.md");
    expect(extraction).toEqual({ metadata: {}, extractionVersion: METADATA_EXTRACTION_VERSION });
  });

  test("frontmatter without qmd namespace yields empty metadata", () => {
    const extraction = extractDocumentMetadata(buildDoc("title: Hello\ntags: [a, b]\n"), "doc.md");
    expect(extraction.metadata).toEqual({});
    expect(extraction.error).toBeUndefined();
  });

  test("qmd namespace without metadata yields empty metadata", () => {
    const extraction = extractDocumentMetadata(buildDoc("qmd:\n  other: true\n"), "doc.md");
    expect(extraction.metadata).toEqual({});
    expect(extraction.error).toBeUndefined();
  });

  test("empty qmd.metadata mapping yields empty metadata", () => {
    const extraction = extractDocumentMetadata(buildDoc("qmd:\n  metadata: {}\n"), "doc.md");
    expect(extraction.metadata).toEqual({});
    expect(extraction.error).toBeUndefined();
  });

  test("extracts string, number, and boolean scalars", () => {
    const extraction = extractDocumentMetadata(
      buildDoc("qmd:\n  metadata:\n    status: published\n    priority: 3\n    reviewed: true\n"),
      "doc.md",
    );
    expect(extraction.metadata).toEqual({ status: "published", priority: 3, reviewed: true });
    expect(extraction.error).toBeUndefined();
  });

  test("extracts homogeneous arrays of every scalar type", () => {
    const extraction = extractDocumentMetadata(
      buildDoc([
        "qmd:",
        "  metadata:",
        "    topics: [typescript, programming]",
        "    scores: [1, 2.5, 3]",
        "    flags: [true, false]",
        "",
      ].join("\n")),
      "doc.md",
    );
    expect(extraction.metadata).toEqual({
      topics: ["typescript", "programming"],
      scores: [1, 2.5, 3],
      flags: [true, false],
    });
  });

  test("de-duplicates array values preserving first-seen order", () => {
    const extraction = extractDocumentMetadata(
      buildDoc("qmd:\n  metadata:\n    topics: [b, a, b, c, a]\n"),
      "doc.md",
    );
    expect(extraction.metadata["topics"]).toEqual(["b", "a", "c"]);
  });

  test("treats prototype-shaped metadata keys as ordinary data", () => {
    const extraction = extractDocumentMetadata(
      buildDoc("qmd:\n  metadata:\n    __proto__: inherited\n    constructor: built\n"),
      "doc.md",
    );
    expect(Object.keys(extraction.metadata)).toEqual(["__proto__", "constructor"]);
    expect(extraction.metadata["__proto__"]).toBe("inherited");
    expect(extraction.metadata["constructor"]).toBe("built");
    expect(JSON.stringify(extraction.metadata)).toBe(
      '{"__proto__":"inherited","constructor":"built"}',
    );
  });

  test("tolerates BOM, CRLF, and '...' closing marker", () => {
    const bomDoc = "\uFEFF---\nqmd:\n  metadata:\n    status: ok\n---\nBody\n";
    expect(extractDocumentMetadata(bomDoc, "doc.md").metadata).toEqual({ status: "ok" });

    const crlfDoc = "---\r\nqmd:\r\n  metadata:\r\n    status: ok\r\n---\r\nBody\r\n";
    expect(extractDocumentMetadata(crlfDoc, "doc.md").metadata).toEqual({ status: "ok" });

    const dotsDoc = "---\nqmd:\n  metadata:\n    status: ok\n...\nBody\n";
    expect(extractDocumentMetadata(dotsDoc, "doc.md").metadata).toEqual({ status: "ok" });
  });

  test("missing closing delimiter is treated as no frontmatter", () => {
    const extraction = extractDocumentMetadata("---\nqmd:\n  metadata:\n    status: ok\n", "doc.md");
    expect(extraction).toEqual({ metadata: {}, extractionVersion: METADATA_EXTRACTION_VERSION });
  });

  test("non-markdown extensions are not parsed as frontmatter", () => {
    const content = buildDoc("qmd:\n  metadata:\n    status: ok\n");
    expect(extractDocumentMetadata(content, "script.ts").metadata).toEqual({});
    expect(extractDocumentMetadata(content, "noext").metadata).toEqual({});
    expect(extractDocumentMetadata(content, "doc.markdown").metadata).toEqual({ status: "ok" });
    expect(extractDocumentMetadata(content, "doc.mdx").metadata).toEqual({ status: "ok" });
  });

  test("malformed YAML records an extraction error", () => {
    const extraction = extractDocumentMetadata(buildDoc("qmd: [unclosed\n"), "doc.md");
    expect(extraction.metadata).toEqual({});
    expect(extraction.error).toMatch(/invalid frontmatter YAML/);
  });

  test("non-mapping qmd or qmd.metadata records an extraction error", () => {
    const qmdScalar = extractDocumentMetadata(buildDoc("qmd: hello\n"), "doc.md");
    expect(qmdScalar.error).toMatch(/'qmd' must be a mapping/);

    const metadataScalar = extractDocumentMetadata(buildDoc("qmd:\n  metadata: hello\n"), "doc.md");
    expect(metadataScalar.error).toMatch(/'qmd.metadata' must be a mapping/);
  });

  test("rejects nested objects, null, empty arrays, nested arrays, and mixed arrays", () => {
    const cases: [string, RegExp][] = [
      ["qmd:\n  metadata:\n    nested:\n      a: 1\n", /unsupported value type/],
      ["qmd:\n  metadata:\n    empty: null\n", /null is not supported/],
      ["qmd:\n  metadata:\n    empty: []\n", /empty arrays are not supported/],
      ["qmd:\n  metadata:\n    nested: [[1, 2]]\n", /nested arrays are not supported/],
      ["qmd:\n  metadata:\n    mixed: [1, two]\n", /mixed-type arrays are not supported/],
    ];

    for (const [frontmatterYaml, expected] of cases) {
      const extraction = extractDocumentMetadata(buildDoc(frontmatterYaml), "doc.md");
      expect(extraction.metadata).toEqual({});
      expect(extraction.error).toMatch(expected);
    }
  });

  test("rejects non-finite numbers", () => {
    const extraction = extractDocumentMetadata(buildDoc("qmd:\n  metadata:\n    bad: .inf\n"), "doc.md");
    expect(extraction.metadata).toEqual({});
    expect(extraction.error).toMatch(/finite/);
  });

  test("rejects oversized keys, strings, arrays, and key counts", () => {
    const longKey = "k".repeat(METADATA_LIMITS.maxKeyBytes + 1);
    expect(extractDocumentMetadata(buildDoc(`qmd:\n  metadata:\n    ${longKey}: 1\n`), "doc.md").error)
      .toMatch(/key exceeds/);

    const longString = "v".repeat(METADATA_LIMITS.maxStringLength + 1);
    expect(extractDocumentMetadata(buildDoc(`qmd:\n  metadata:\n    long: "${longString}"\n`), "doc.md").error)
      .toMatch(/string exceeds/);

    const bigArray = `[${Array.from({ length: METADATA_LIMITS.maxArrayLength + 1 }, (_, i) => i).join(", ")}]`;
    expect(extractDocumentMetadata(buildDoc(`qmd:\n  metadata:\n    big: ${bigArray}\n`), "doc.md").error)
      .toMatch(/array exceeds/);

    const manyKeys = Array.from({ length: METADATA_LIMITS.maxKeys + 1 }, (_, i) => `    key${i}: 1`).join("\n");
    expect(extractDocumentMetadata(buildDoc(`qmd:\n  metadata:\n${manyKeys}\n`), "doc.md").error)
      .toMatch(/keys \(max/);
  });

  test("rejects oversized frontmatter blocks", () => {
    const filler = `filler: "${"x".repeat(METADATA_LIMITS.maxFrontmatterBytes)}"\n`;
    const extraction = extractDocumentMetadata(buildDoc(`${filler}qmd:\n  metadata:\n    status: ok\n`), "doc.md");
    expect(extraction.metadata).toEqual({});
    expect(extraction.error).toMatch(/frontmatter exceeds/);
  });

  test("bounds YAML alias expansion", () => {
    const aliasBomb = [
      "a: &a [x, x, x, x, x, x, x, x, x, x]",
      "b: &b [*a, *a, *a, *a, *a, *a, *a, *a, *a, *a]",
      "c: &c [*b, *b, *b, *b, *b, *b, *b, *b, *b, *b]",
      "qmd:",
      "  metadata:",
      "    status: ok",
      "",
    ].join("\n");
    const extraction = extractDocumentMetadata(buildDoc(aliasBomb), "doc.md");
    expect(extraction.metadata).toEqual({});
    expect(extraction.error).toMatch(/invalid frontmatter YAML/);
  });

  test("bounds extraction error message length", () => {
    const longKey = "k".repeat(600);
    const extraction = extractDocumentMetadata(
      buildDoc(`qmd:\n  metadata:\n    valid: 1\n    "${longKey}x": 1\n`),
      "doc.md",
    );
    expect(extraction.error).toBeDefined();
    expect(extraction.error!.length).toBeLessThanOrEqual(METADATA_LIMITS.maxErrorLength);
  });

  test("unquoted dates stay strings", () => {
    const extraction = extractDocumentMetadata(
      buildDoc("qmd:\n  metadata:\n    published: 2024-01-15\n"),
      "doc.md",
    );
    expect(extraction.metadata).toEqual({ published: "2024-01-15" });
  });
});

/** A JavaScript caller's source, whose answer the TypeScript signature cannot check. */
function untypedSource(answer: (document: Parameters<DocumentMetadataSource>[0]) => unknown): DocumentMetadataSource {
  return answer as DocumentMetadataSource;
}

/** 63 full-length values plus one padded value: canonical JSON of exactly `bytes` UTF-8 bytes. */
function answerWithJsonBytes(bytes: number): Record<string, string> {
  const answer: Record<string, string> = {};
  for (let index = 0; index < METADATA_LIMITS.maxKeys - 1; index++) {
    answer[`k${String(index).padStart(2, "0")}`] = "x".repeat(METADATA_LIMITS.maxStringLength);
  }
  answer["last"] = "";
  answer["last"] = "x".repeat(bytes - Buffer.byteLength(JSON.stringify(answer), "utf-8"));
  return answer;
}

describe("normalizeSourceMetadata", () => {
  test("returns detached canonical metadata for valid answers", () => {
    const topics = ["b", "a", "b"];
    const extraction = normalizeSourceMetadata({ topics, priority: 3, reviewed: false });
    topics.push("c");
    expect(extraction).toEqual({
      metadata: { topics: ["b", "a"], priority: 3, reviewed: false },
      extractionVersion: METADATA_EXTRACTION_VERSION,
    });

    const nullPrototype: Record<string, string> = Object.create(null);
    nullPrototype["status"] = "ok";
    const atBound = answerWithJsonBytes(METADATA_LIMITS.maxFrontmatterBytes);
    const rows: [name: string, answer: unknown, metadata: DocumentMetadata][] = [
      ["empty map", {}, {}],
      ["null prototype", nullPrototype, { status: "ok" }],
      ["frozen", Object.freeze({ tags: Object.freeze(["x", "y"]) }), { tags: ["x", "y"] }],
      ["string then", { then: "x" }, { then: "x" }],
      ["aggregate JSON at the bound", atBound, { ...atBound }],
    ];
    for (const [name, answer, metadata] of rows) {
      expect({ name, extraction: normalizeSourceMetadata(answer) })
        .toEqual({ name, extraction: { metadata, extractionVersion: METADATA_EXTRACTION_VERSION } });
    }
  });

  test("turns each invalid answer into empty metadata and one bounded error line", () => {
    const rows: [name: string, answer: unknown, error: RegExp][] = [
      ["undefined", undefined, /expected a plain object, received undefined$/],
      ["null", null, /expected a plain object, received null$/],
      ["array", [{ status: "ok" }], /expected a plain object, received array$/],
      ["Map", new Map([["author", "human"]]), /expected a plain object, received Map$/],
      ["Date", new Date(0), /expected a plain object, received Date$/],
      ["string", "author: human", /expected a plain object, received string$/],
      ["function", () => ({ status: "ok" }), /expected a plain object, received function$/],
      ["null value", { a: null }, /null is not supported/],
      ["empty array", { a: [] }, /empty arrays are not supported/],
      ["mixed array", { a: [1, "x"] }, /mixed-type arrays are not supported/],
      ["nested array", { a: [[1]] }, /nested arrays are not supported/],
      ["NaN", { a: Number.NaN }, /numbers must be finite/],
      ["nested object", { a: {} }, /unsupported value type/],
      ["function value", { a: () => 1 }, /unsupported value type/],
      ["65 keys", Object.fromEntries(Array.from({ length: METADATA_LIMITS.maxKeys + 1 }, (_, i) => [`k${i}`, 1])), /65 keys \(max 64\)/],
      ["129-byte key", { ["k".repeat(METADATA_LIMITS.maxKeyBytes + 1)]: 1 }, /key exceeds 128 bytes/],
      ["control-character key", { "a\u0007b": 1 }, /control characters/],
      ["1,025-unit string", { a: "v".repeat(METADATA_LIMITS.maxStringLength + 1) }, /string exceeds 1024 characters/],
      ["129-value array", { a: Array.from({ length: METADATA_LIMITS.maxArrayLength + 1 }, (_, i) => i) }, /array exceeds 128 values/],
      ["aggregate JSON past the bound", answerWithJsonBytes(METADATA_LIMITS.maxFrontmatterBytes + 1), /metadata JSON exceeds 65536 bytes$/],
      ["truncated message", { ["k".repeat(METADATA_LIMITS.maxKeyBytes)]: null }, /null is not supported.*\.\.\.$/],
    ];
    for (const [name, answer, error] of rows) {
      const extraction = normalizeSourceMetadata(answer);
      expect({ name, extraction }).toEqual({
        name,
        extraction: {
          metadata: {},
          error: expect.stringMatching(/^metadata source: [^\n]*$/),
          extractionVersion: METADATA_EXTRACTION_VERSION,
        },
      });
      expect({ name, error: extraction.error }).toEqual({ name, error: expect.stringMatching(error) });
      expect(extraction.error!.length).toBeLessThanOrEqual(METADATA_LIMITS.maxErrorLength);
    }
  });
});

describe("askMetadataSource", () => {
  const document = { collection: "notes", path: "sub/a.md", hash: "c0ffee" };

  test("passes collection, path and hash, and normalizes a synchronous answer", () => {
    const asked: Parameters<DocumentMetadataSource>[0][] = [];
    const extraction = askMetadataSource((sourceDocument) => {
      asked.push(sourceDocument);
      return { author: "human" };
    }, document);
    expect(asked).toEqual([document]);
    expect(extraction).toEqual({ metadata: { author: "human" }, extractionVersion: METADATA_EXTRACTION_VERSION });
  });

  test("rejects a throwing source with the collection, path and cause", () => {
    const failure = new Error("source map has no row");
    let thrown: unknown;
    try {
      askMetadataSource(() => { throw failure; }, document);
    } catch (err) {
      thrown = err;
    }
    expect(thrown).toBeInstanceOf(Error);
    expect((thrown as Error).message).toBe("Metadata source failed for notes/sub/a.md: source map has no row");
    expect((thrown as Error).cause).toBe(failure);
  });

  test("rejects thenable answers and handles their later rejections", async () => {
    const unhandled: unknown[] = [];
    const recordUnhandled = (reason: unknown) => { unhandled.push(reason); };
    process.on("unhandledRejection", recordUnhandled);
    try {
      const thenableSources = [
        untypedSource(async () => ({ status: "ok" })),
        untypedSource(() => Promise.reject(new Error("late rejection"))),
        untypedSource(() => ({ then: (_resolve: unknown, reject: (reason: unknown) => void) => reject(new Error("late thenable")) })),
      ];
      for (const source of thenableSources) {
        expect(() => askMetadataSource(source, document)).toThrow(TypeError);
        expect(() => askMetadataSource(source, document)).toThrow(
          "Metadata source returned a thenable for notes/sub/a.md. DocumentMetadataSource must return metadata synchronously.",
        );
      }
      await new Promise(resolve => setImmediate(resolve));
      expect(unhandled).toEqual([]);
    } finally {
      process.off("unhandledRejection", recordUnhandled);
    }
  });
});
