/**
 * QMD Metadata - Public metadata types, frontmatter extraction, and SDK metadata sources.
 *
 * Documents opt into metadata through a namespaced Markdown frontmatter block:
 *
 *   ---
 *   qmd:
 *     metadata:
 *       topics:
 *         - typescript
 *         - programming
 *       status: published
 *   ---
 *
 * Frontmatter extraction and SDK metadata normalization produce a canonical
 * `MetadataExtractionResult`. Storage records each row's origin; filtering
 * reads both origins.
 *
 * The raw document is never modified — frontmatter stays part of the stored,
 * indexed, chunked, and embedded content.
 */

import YAML from "yaml";

// =============================================================================
// Public types
// =============================================================================

export type MetadataScalar = string | number | boolean;

/** The stored type of one metadata value. Arrays are homogeneous, so a key holds one type per document. */
export type MetadataValueType = "string" | "number" | "boolean";

export type MetadataScalarArray =
  | readonly string[]
  | readonly number[]
  | readonly boolean[];

export type MetadataValue = MetadataScalar | MetadataScalarArray;

export type DocumentMetadata = Record<string, MetadataValue>;

/**
 * Result of extracting metadata from one document.
 *
 * Invalid `qmd.metadata` frontmatter or an invalid SDK metadata source answer
 * sets `error`. The document indexes with empty metadata and stays out of
 * filtered search until a subsequent update extracts valid metadata.
 */
export interface MetadataExtractionResult {
  metadata: DocumentMetadata;
  error?: string;
  extractionVersion: number;
}

/**
 * Supplies one document's complete metadata during `update()`, in place of
 * its `qmd.metadata` frontmatter. QMD calls it synchronously for every
 * nonblank file the scan admits, unchanged files included, with the path and
 * content hash the scan already holds. `{}` clears the document's metadata.
 */
export type DocumentMetadataSource = (document: {
  collection: string;
  /** Normalized, collection-relative document path. */
  path: string;
  /** QMD content hash of the indexed text (`documents.hash`). */
  hash: string;
}) => DocumentMetadata;

// =============================================================================
// Limits
// =============================================================================

/**
 * Bump when extraction or normalization semantics change so existing rows are
 * re-extracted on the next `qmd update`.
 */
export const METADATA_EXTRACTION_VERSION = 1;

/** Defensive limits for metadata from untrusted repositories. */
export const METADATA_LIMITS = {
  maxFrontmatterBytes: 64 * 1024,
  maxKeys: 64,
  maxKeyBytes: 128,
  maxStringLength: 1024,
  maxArrayLength: 128,
  maxYamlAliasCount: 100,
  maxErrorLength: 200,
} as const;

/** File extensions parsed for frontmatter metadata. */
const FRONTMATTER_FILE_EXTENSIONS = new Set([".md", ".markdown", ".mdx"]);

// =============================================================================
// Extraction
// =============================================================================

/**
 * Extract `qmd.metadata` from a document's leading YAML frontmatter.
 *
 * Never throws. A document without frontmatter, without a `qmd` namespace, or
 * with a non-frontmatter extension yields empty metadata with no error.
 * Invalid frontmatter or invalid metadata yields empty metadata plus a bounded
 * extraction error.
 */
export function extractDocumentMetadata(content: string, path: string): MetadataExtractionResult {
  const success = (metadata: DocumentMetadata): MetadataExtractionResult =>
    ({ metadata, extractionVersion: METADATA_EXTRACTION_VERSION });

  const failure = (message: string): MetadataExtractionResult => ({
    metadata: {},
    error: truncateErrorMessage(message),
    extractionVersion: METADATA_EXTRACTION_VERSION,
  });

  if (!hasFrontmatterFileExtension(path)) return success({});

  const frontmatterYaml = getFrontmatterYaml(content);
  if (frontmatterYaml === null) return success({});

  if (Buffer.byteLength(frontmatterYaml, "utf-8") > METADATA_LIMITS.maxFrontmatterBytes) {
    return failure(`frontmatter exceeds ${METADATA_LIMITS.maxFrontmatterBytes} bytes`);
  }

  let frontmatter: unknown;
  try {
    frontmatter = YAML.parse(frontmatterYaml, { maxAliasCount: METADATA_LIMITS.maxYamlAliasCount });
  } catch (err) {
    return failure(`invalid frontmatter YAML: ${err instanceof Error ? err.message : String(err)}`);
  }

  if (!isPlainObject(frontmatter)) return success({});

  const qmdNamespace = frontmatter["qmd"];
  if (qmdNamespace === undefined) return success({});
  if (!isPlainObject(qmdNamespace)) {
    return failure("frontmatter 'qmd' must be a mapping");
  }

  const rawMetadata = qmdNamespace["metadata"];
  if (rawMetadata === undefined) return success({});
  if (!isPlainObject(rawMetadata)) {
    return failure("frontmatter 'qmd.metadata' must be a mapping");
  }

  try {
    return success(normalizeMetadata(rawMetadata));
  } catch (err) {
    return failure(err instanceof Error ? err.message : String(err));
  }
}

function hasFrontmatterFileExtension(path: string): boolean {
  const dotIndex = path.lastIndexOf(".");
  if (dotIndex < 0) return false;
  return FRONTMATTER_FILE_EXTENSIONS.has(path.slice(dotIndex).toLowerCase());
}

/**
 * Slice the YAML between a leading `---` line and a closing `---` or `...`
 * line. Tolerates a UTF-8 BOM and CRLF line endings. Returns null when the
 * document has no complete leading frontmatter block.
 */
function getFrontmatterYaml(content: string): string | null {
  const body = content.charCodeAt(0) === 0xfeff ? content.slice(1) : content;

  const openMatch = body.match(/^---[ \t]*\r?\n/);
  if (!openMatch) return null;

  const yamlStart = openMatch[0].length;
  const closePattern = /^(?:---|\.\.\.)[ \t]*(?:\r?\n|$)/m;
  const closeMatch = body.slice(yamlStart).match(closePattern);
  if (!closeMatch || closeMatch.index === undefined) return null;

  return body.slice(yamlStart, yamlStart + closeMatch.index);
}

// =============================================================================
// Metadata sources
// =============================================================================

/**
 * Ask a metadata source for one document's metadata. A throwing source or a
 * thenable answer rejects the scan, because an error row would replace the
 * document's last good metadata. Every other answer is normalized.
 */
export function askMetadataSource(
  source: DocumentMetadataSource,
  document: Parameters<DocumentMetadataSource>[0],
): MetadataExtractionResult {
  const { collection, path, hash } = document;
  let answer: unknown;
  try {
    answer = source({ collection, path, hash });
  } catch (cause) {
    throw new Error(`Metadata source failed for ${collection}/${path}: ${describeCause(cause)}`, { cause });
  }
  if (isThenable(answer)) {
    // The abandoned promise may reject later; a handler keeps that rejection
    // from surfacing as an unhandled rejection.
    try {
      Promise.resolve(answer).catch(() => {});
    } catch {}
    throw new TypeError(
      `Metadata source returned a thenable for ${collection}/${path}. DocumentMetadataSource must return metadata synchronously.`,
    );
  }
  return normalizeSourceMetadata(answer);
}

/**
 * Normalize one metadata source answer under the frontmatter limits, plus a
 * `maxFrontmatterBytes` bound on the canonical JSON that QMD persists. Never
 * throws: an answer that is not a plain object, or that normalization
 * rejects, yields empty metadata plus a bounded error.
 */
export function normalizeSourceMetadata(answer: unknown): MetadataExtractionResult {
  try {
    if (!isPlainRecord(answer)) {
      throw new Error(`expected a plain object, received ${describeAnswerKind(answer)}`);
    }
    const metadata = normalizeMetadata(answer);
    if (Buffer.byteLength(JSON.stringify(metadata), "utf-8") > METADATA_LIMITS.maxFrontmatterBytes) {
      throw new Error(`metadata JSON exceeds ${METADATA_LIMITS.maxFrontmatterBytes} bytes`);
    }
    return { metadata, extractionVersion: METADATA_EXTRACTION_VERSION };
  } catch (err) {
    return {
      metadata: {},
      error: truncateErrorMessage(`metadata source: ${describeCause(err)}`),
      extractionVersion: METADATA_EXTRACTION_VERSION,
    };
  }
}

function isThenable(value: unknown): boolean {
  if ((typeof value !== "object" || value === null) && typeof value !== "function") return false;
  try {
    return typeof (value as { then?: unknown }).then === "function";
  } catch {
    return false;
  }
}

/** Prototype `Object.prototype` or null. A Map or a Date answer is invalid: `Object.keys` would read it as `{}`. */
function isPlainRecord(value: unknown): value is Record<string, unknown> {
  if (!isPlainObject(value)) return false;
  const prototype: unknown = Object.getPrototypeOf(value);
  return prototype === Object.prototype || prototype === null;
}

function describeAnswerKind(answer: unknown): string {
  if (answer === null) return "null";
  if (Array.isArray(answer)) return "array";
  if (typeof answer !== "object") return typeof answer;
  const constructorName: unknown = Object.getPrototypeOf(answer)?.constructor?.name;
  return typeof constructorName === "string" && constructorName ? constructorName : "object";
}

function describeCause(cause: unknown): string {
  try {
    return cause instanceof Error ? cause.message : String(cause);
  } catch {
    return "unprintable error";
  }
}

// =============================================================================
// Normalization
// =============================================================================

/**
 * Normalize raw `qmd.metadata` into canonical `DocumentMetadata`.
 * Throws on any unsupported key or value — extraction is all-or-nothing per
 * document so stale partial metadata can never persist.
 */
function normalizeMetadata(rawMetadata: Record<string, unknown>): DocumentMetadata {
  const keys = Object.keys(rawMetadata);
  if (keys.length > METADATA_LIMITS.maxKeys) {
    throw new Error(`metadata has ${keys.length} keys (max ${METADATA_LIMITS.maxKeys})`);
  }

  const entries: [string, MetadataValue][] = [];
  for (const key of keys) {
    validateMetadataKey(key);
    entries.push([key, normalizeMetadataValue(key, rawMetadata[key])]);
  }
  // Object.fromEntries defines "__proto__" as ordinary data instead of
  // invoking Object.prototype's legacy setter. Metadata keys are unrestricted
  // user data, so prototype-shaped names must round-trip like every other key.
  return Object.fromEntries(entries);
}

function validateMetadataKey(key: string): void {
  if (key.length === 0) {
    throw new Error("metadata keys must be non-empty strings");
  }
  if (Buffer.byteLength(key, "utf-8") > METADATA_LIMITS.maxKeyBytes) {
    throw new Error(`metadata key exceeds ${METADATA_LIMITS.maxKeyBytes} bytes`);
  }
  // eslint-disable-next-line no-control-regex
  if (/[\u0000-\u001f\u007f]/.test(key)) {
    throw new Error("metadata keys must not contain control characters");
  }
}

function normalizeMetadataValue(key: string, rawValue: unknown): MetadataValue {
  if (Array.isArray(rawValue)) {
    return normalizeMetadataArray(key, rawValue);
  }
  return normalizeMetadataScalar(key, rawValue);
}

function normalizeMetadataScalar(key: string, rawValue: unknown): MetadataScalar {
  if (typeof rawValue === "string") {
    if (rawValue.length > METADATA_LIMITS.maxStringLength) {
      throw new Error(`metadata key "${key}": string exceeds ${METADATA_LIMITS.maxStringLength} characters`);
    }
    return rawValue;
  }
  if (typeof rawValue === "number") {
    if (!Number.isFinite(rawValue)) {
      throw new Error(`metadata key "${key}": numbers must be finite`);
    }
    return rawValue;
  }
  if (typeof rawValue === "boolean") return rawValue;
  if (rawValue === null) {
    throw new Error(`metadata key "${key}": null is not supported — omit the key instead`);
  }
  throw new Error(`metadata key "${key}": unsupported value type — use strings, numbers, booleans, or flat arrays of one of those`);
}

function normalizeMetadataArray(key: string, rawValues: unknown[]): MetadataScalarArray {
  if (rawValues.length === 0) {
    throw new Error(`metadata key "${key}": empty arrays are not supported — omit the key instead`);
  }
  if (rawValues.length > METADATA_LIMITS.maxArrayLength) {
    throw new Error(`metadata key "${key}": array exceeds ${METADATA_LIMITS.maxArrayLength} values`);
  }

  const scalars = rawValues.map(rawValue => {
    if (Array.isArray(rawValue)) {
      throw new Error(`metadata key "${key}": nested arrays are not supported`);
    }
    return normalizeMetadataScalar(key, rawValue);
  });

  // Narrow each homogeneous case explicitly so the public array union remains
  // precise without discarding type evidence through chained assertions.
  if (scalars.every((scalar): scalar is string => typeof scalar === "string")) {
    return Array.from(new Set(scalars));
  }
  if (scalars.every((scalar): scalar is number => typeof scalar === "number")) {
    return Array.from(new Set(scalars));
  }
  if (scalars.every((scalar): scalar is boolean => typeof scalar === "boolean")) {
    return Array.from(new Set(scalars));
  }
  throw new Error(`metadata key "${key}": mixed-type arrays are not supported`);
}

function isPlainObject(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function truncateErrorMessage(message: string): string {
  const singleLine = message.replace(/\s+/g, " ").trim();
  if (singleLine.length <= METADATA_LIMITS.maxErrorLength) return singleLine;
  return singleLine.slice(0, METADATA_LIMITS.maxErrorLength - 3) + "...";
}
