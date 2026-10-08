export type DocumentLocationRef = {
  uri: string;
  contentHash: string;
};

export type Utf16Span = {
  startUtf16: number;
  endUtf16: number;
};

export type PassageBudget = {
  maxUtf8Bytes: number;
  maxUnicodeScalars?: number;
};

export type PassageWindow = Utf16Span & {
  text: string;
  utf8Bytes: number;
  unicodeScalars: number;
  anchorClipped: boolean;
};

export type PositiveLexicalAnchor = {
  kind: "literal" | "phrase";
  text: string;
  tokens: readonly string[];
  match: "prefix" | "phrase";
};

export type LexicalExactLocation = DocumentLocationRef & Utf16Span & {
  kind: "lexical_exact";
  anchorKind: "literal" | "phrase";
  matchedText: string;
};

export type LexicalApproximateLocation = DocumentLocationRef & Utf16Span & {
  kind: "lexical_approximate";
  reason: "stem_or_tokenizer";
  queryText: string;
};

export type VectorChunkStartLocation = DocumentLocationRef & {
  kind: "vector_chunk_start";
  startUtf16: number;
  endUtf16: null;
  chunkSeq: number | null;
};

export type SelectionWindowLocation = DocumentLocationRef & Utf16Span & {
  kind: "selection_window";
  origin: "keyword_intent";
};

export type LexicalLocation = LexicalExactLocation | LexicalApproximateLocation;

export type SearchLocation =
  | LexicalLocation
  | VectorChunkStartLocation
  | SelectionWindowLocation;

export type LocatedLexicalPassage = {
  location: LexicalLocation;
  passage: PassageWindow;
};

type BodyToken = Utf16Span & {
  comparable: string;
};

const FTS5_SEPARATOR_RUN = /[^\p{L}\p{N}'_]+/u;
const CJK_CHAR_PATTERN = /[\p{Script=Han}\p{Script=Hiragana}\p{Script=Katakana}\p{Script=Hangul}]/u;
const WORD_CHAR_PATTERN = /[\p{L}\p{N}]/u;
const UTF8_ENCODER = new TextEncoder();

function sanitizeFTS5Term(term: string): string {
  return term.replace(/[^\p{L}\p{N}'_]/gu, "").toLowerCase();
}

function splitFTS5CompoundTerm(term: string): string[] {
  return term.split(FTS5_SEPARATOR_RUN).map(sanitizeFTS5Term).filter(Boolean);
}

function containsCjk(text: string): boolean {
  return CJK_CHAR_PATTERN.test(text);
}

function tokensForPhrase(phrase: string): string[] {
  const tokens: string[] = [];
  for (const token of tokenizeText(phrase)) {
    tokens.push(token.comparable);
  }
  return tokens;
}

/** Parse the positive anchors accepted by QMD's lexical query grammar. */
export function parsePositiveLexicalAnchors(query: string): PositiveLexicalAnchor[] {
  const anchors: PositiveLexicalAnchor[] = [];
  const source = query.trim();
  let index = 0;

  while (index < source.length) {
    while (index < source.length && /\s/.test(source[index]!)) index++;
    if (index >= source.length) break;

    const negated = source[index] === "-";
    if (negated) index++;

    if (source[index] === '"') {
      const start = ++index;
      while (index < source.length && source[index] !== '"') index++;
      const phrase = source.slice(start, index).trim();
      if (index < source.length) index++;

      if (!negated && phrase.length > 0) {
        const tokens = tokensForPhrase(phrase);
        if (tokens.length > 0) {
          anchors.push({ kind: "phrase", text: phrase, tokens, match: "phrase" });
        }
      }
      continue;
    }

    const start = index;
    while (index < source.length && !/[\s"]/.test(source[index]!)) index++;
    const term = source.slice(start, index);
    if (negated || term.length === 0) continue;

    if (containsCjk(term)) {
      const tokens = tokensForPhrase(term);
      if (tokens.length > 0) {
        anchors.push({ kind: "phrase", text: term, tokens, match: "phrase" });
      }
      continue;
    }

    const parts = splitFTS5CompoundTerm(term);
    if (parts.length === 0) continue;

    const tokens = tokensForPhrase(term);
    if (tokens.length === 0) continue;

    if (parts.length > 1) {
      anchors.push({ kind: "phrase", text: term, tokens, match: "phrase" });
    } else {
      anchors.push({ kind: "literal", text: term, tokens, match: "prefix" });
    }
  }

  return anchors;
}

/** Return a byte-bounded passage around a UTF-16 source span. */
export function boundedPassageWindow(
  body: string,
  anchor: Utf16Span,
  budget: PassageBudget,
): PassageWindow {
  assertBodySpan(body, anchor, "anchor");
  assertBudget(budget);

  const maxScalars = budget.maxUnicodeScalars ?? Number.POSITIVE_INFINITY;
  const anchorText = body.slice(anchor.startUtf16, anchor.endUtf16);
  const anchorBytes = utf8Bytes(anchorText);
  const anchorScalars = unicodeScalars(anchorText);

  if (anchorBytes > budget.maxUtf8Bytes || anchorScalars > maxScalars) {
    let endUtf16 = anchor.startUtf16;
    let bytes = 0;
    let scalars = 0;

    while (endUtf16 < anchor.endUtf16) {
      const nextEnd = nextScalarEnd(body, endUtf16);
      const scalarBytes = utf8Bytes(body.slice(endUtf16, nextEnd));
      if (bytes + scalarBytes > budget.maxUtf8Bytes || scalars + 1 > maxScalars) break;
      bytes += scalarBytes;
      scalars++;
      endUtf16 = nextEnd;
    }

    return {
      startUtf16: anchor.startUtf16,
      endUtf16,
      text: body.slice(anchor.startUtf16, endUtf16),
      utf8Bytes: bytes,
      unicodeScalars: scalars,
      anchorClipped: endUtf16 < anchor.endUtf16,
    };
  }

  let startUtf16 = anchor.startUtf16;
  let endUtf16 = anchor.endUtf16;
  let bytes = anchorBytes;
  let scalars = anchorScalars;
  let nextSide: "left" | "right" = "left";
  let leftOpen = startUtf16 > 0;
  let rightOpen = endUtf16 < body.length;

  while (leftOpen || rightOpen) {
    let added = false;
    const sides: readonly ("left" | "right")[] =
      nextSide === "left" ? ["left", "right"] : ["right", "left"];

    for (const side of sides) {
      if (side === "left" && leftOpen) {
        const nextStart = previousScalarStart(body, startUtf16);
        const scalarBytes = utf8Bytes(body.slice(nextStart, startUtf16));
        if (bytes + scalarBytes <= budget.maxUtf8Bytes && scalars + 1 <= maxScalars) {
          startUtf16 = nextStart;
          bytes += scalarBytes;
          scalars++;
          leftOpen = startUtf16 > 0;
          nextSide = "right";
          added = true;
          break;
        }
        leftOpen = false;
      }

      if (side === "right" && rightOpen) {
        const nextEnd = nextScalarEnd(body, endUtf16);
        const scalarBytes = utf8Bytes(body.slice(endUtf16, nextEnd));
        if (bytes + scalarBytes <= budget.maxUtf8Bytes && scalars + 1 <= maxScalars) {
          endUtf16 = nextEnd;
          bytes += scalarBytes;
          scalars++;
          rightOpen = endUtf16 < body.length;
          nextSide = "left";
          added = true;
          break;
        }
        rightOpen = false;
      }
    }

    if (!added && !leftOpen && !rightOpen) break;
  }

  return {
    startUtf16,
    endUtf16,
    text: body.slice(startUtf16, endUtf16),
    utf8Bytes: bytes,
    unicodeScalars: scalars,
    anchorClipped: false,
  };
}

/** Locate the first strongest positive lexical anchor in the document body. */
export function locateLexical(
  body: string,
  query: string,
  ref: DocumentLocationRef,
  budget: PassageBudget,
): LocatedLexicalPassage | null {
  assertDocumentRef(ref);
  assertBudget(budget);

  const anchors = parsePositiveLexicalAnchors(query);
  for (const anchor of anchors) {
    const span = findExactAnchor(body, anchor);
    if (!span) continue;

    const passage = boundedPassageWindow(body, span, budget);
    return {
      location: {
        ...ref,
        ...span,
        kind: "lexical_exact",
        anchorKind: anchor.kind,
        matchedText: body.slice(span.startUtf16, span.endUtf16),
      },
      passage,
    };
  }

  const bodyTokens = tokenizeText(body);
  for (const anchor of anchors) {
    const span = findApproximateAnchor(bodyTokens, anchor);
    if (!span) continue;

    const passage = boundedPassageWindow(body, span, budget);
    return {
      location: {
        ...ref,
        ...span,
        kind: "lexical_approximate",
        reason: "stem_or_tokenizer",
        queryText: anchor.text,
      },
      passage,
    };
  }

  return null;
}

export function vectorChunkLocation(
  ref: DocumentLocationRef,
  startUtf16: number,
  chunkSeq: number | null,
): VectorChunkStartLocation {
  assertDocumentRef(ref);
  assertOffset(startUtf16, "startUtf16");
  if (chunkSeq !== null) assertOffset(chunkSeq, "chunkSeq");
  return { ...ref, kind: "vector_chunk_start", startUtf16, endUtf16: null, chunkSeq };
}

export function selectionWindowLocation(
  ref: DocumentLocationRef,
  span: Utf16Span,
): SelectionWindowLocation {
  assertDocumentRef(ref);
  assertSpan(span, "selection window");
  return { ...ref, ...span, kind: "selection_window", origin: "keyword_intent" };
}

function findExactAnchor(body: string, anchor: PositiveLexicalAnchor): Utf16Span | null {
  const pattern = new RegExp(escapeRegExp(anchor.text), "giu");
  for (const match of body.matchAll(pattern)) {
    const startUtf16 = match.index;
    const matchedText = match[0];
    if (startUtf16 === undefined || matchedText === undefined) continue;
    const endUtf16 = startUtf16 + matchedText.length;

    const firstAnchorScalar = anchor.text.slice(0, nextScalarEnd(anchor.text, 0));
    const lastAnchorStart = previousScalarStart(anchor.text, anchor.text.length);
    const lastAnchorScalar = anchor.text.slice(lastAnchorStart);
    const startsAtBoundary = containsCjk(firstAnchorScalar) || isTokenBoundary(body, startUtf16);
    const endsAtBoundary = containsCjk(lastAnchorScalar) || isTokenBoundary(body, endUtf16);
    if (startsAtBoundary && (anchor.match === "prefix" || endsAtBoundary)) {
      return { startUtf16, endUtf16 };
    }
  }
  return null;
}

function findApproximateAnchor(
  bodyTokens: readonly BodyToken[],
  anchor: PositiveLexicalAnchor,
): Utf16Span | null {
  if (anchor.tokens.length === 0) return null;

  for (let bodyIndex = 0; bodyIndex + anchor.tokens.length <= bodyTokens.length; bodyIndex++) {
    let matches = true;
    for (let queryIndex = 0; queryIndex < anchor.tokens.length; queryIndex++) {
      const queryToken = comparableToken(anchor.tokens[queryIndex]!);
      const bodyToken = bodyTokens[bodyIndex + queryIndex]!;
      const prefixAllowed = anchor.match === "prefix" && queryIndex === anchor.tokens.length - 1;
      if (!tokensApproximate(queryToken, bodyToken.comparable, prefixAllowed)) {
        matches = false;
        break;
      }
    }

    if (matches) {
      return {
        startUtf16: bodyTokens[bodyIndex]!.startUtf16,
        endUtf16: bodyTokens[bodyIndex + anchor.tokens.length - 1]!.endUtf16,
      };
    }
  }

  return null;
}

function tokensApproximate(query: string, body: string, prefixAllowed: boolean): boolean {
  if (query === body) return true;
  if (prefixAllowed && body.startsWith(query)) return true;
  return porterComparableStem(query) === porterComparableStem(body);
}

function porterComparableStem(token: string): string {
  let stem = token;
  if (stem.length < 3) return stem;

  if (stem.endsWith("ies") && stem.length > 4) {
    stem = `${stem.slice(0, -3)}y`;
  } else if (stem.endsWith("sses")) {
    stem = stem.slice(0, -2);
  } else if (stem.endsWith("s") && !stem.endsWith("ss") && stem.length > 3) {
    stem = stem.slice(0, -1);
  }

  const suffix = stem.endsWith("ing") ? "ing" : stem.endsWith("ed") ? "ed" : null;
  if (suffix && stem.length - suffix.length >= 3) {
    stem = stem.slice(0, -suffix.length);
    if (/([^aeiou])\1$/u.test(stem) && !/[lsz]{2}$/u.test(stem)) {
      stem = stem.slice(0, -1);
    }
  }

  return stem;
}

function tokenizeText(text: string): BodyToken[] {
  const tokens: BodyToken[] = [];
  let tokenStart: number | null = null;
  let index = 0;

  const flush = (endUtf16: number): void => {
    if (tokenStart === null) return;
    const tokenText = text.slice(tokenStart, endUtf16);
    tokens.push({
      startUtf16: tokenStart,
      endUtf16,
      comparable: comparableToken(tokenText),
    });
    tokenStart = null;
  };

  while (index < text.length) {
    const end = nextScalarEnd(text, index);
    const scalar = text.slice(index, end);

    if (containsCjk(scalar)) {
      flush(index);
      tokens.push({
        startUtf16: index,
        endUtf16: end,
        comparable: comparableToken(scalar),
      });
    } else if (WORD_CHAR_PATTERN.test(scalar)) {
      tokenStart ??= index;
    } else {
      flush(index);
    }

    index = end;
  }
  flush(text.length);
  return tokens;
}

function comparableToken(token: string): string {
  return token.normalize("NFD").replace(/\p{M}/gu, "").toLowerCase();
}

function isTokenBoundary(body: string, offset: number): boolean {
  if (offset === 0 || offset === body.length) return true;
  const previousStart = previousScalarStart(body, offset);
  const previous = body.slice(previousStart, offset);
  const next = body.slice(offset, nextScalarEnd(body, offset));
  return !WORD_CHAR_PATTERN.test(previous) || !WORD_CHAR_PATTERN.test(next);
}

function escapeRegExp(value: string): string {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function previousScalarStart(text: string, offset: number): number {
  if (offset <= 0) return 0;
  const last = text.charCodeAt(offset - 1);
  if (offset >= 2 && last >= 0xdc00 && last <= 0xdfff) {
    const first = text.charCodeAt(offset - 2);
    if (first >= 0xd800 && first <= 0xdbff) return offset - 2;
  }
  return offset - 1;
}

function nextScalarEnd(text: string, offset: number): number {
  if (offset >= text.length) return text.length;
  const first = text.charCodeAt(offset);
  if (first >= 0xd800 && first <= 0xdbff && offset + 1 < text.length) {
    const second = text.charCodeAt(offset + 1);
    if (second >= 0xdc00 && second <= 0xdfff) return offset + 2;
  }
  return offset + 1;
}

function isScalarBoundary(text: string, offset: number): boolean {
  if (offset <= 0 || offset >= text.length) return true;
  const previous = text.charCodeAt(offset - 1);
  const next = text.charCodeAt(offset);
  return !(previous >= 0xd800 && previous <= 0xdbff && next >= 0xdc00 && next <= 0xdfff);
}

function utf8Bytes(text: string): number {
  return UTF8_ENCODER.encode(text).byteLength;
}

function unicodeScalars(text: string): number {
  return Array.from(text).length;
}

function assertDocumentRef(ref: DocumentLocationRef): void {
  if (ref.uri.length === 0) throw new RangeError("uri must contain at least one character");
  if (ref.contentHash.length === 0) {
    throw new RangeError("contentHash must contain at least one character");
  }
}

function assertOffset(value: number, name: string): void {
  if (!Number.isSafeInteger(value) || value < 0) {
    throw new RangeError(`${name} must be a non-negative safe integer`);
  }
}

function assertSpan(span: Utf16Span, name: string): void {
  assertOffset(span.startUtf16, `${name}.startUtf16`);
  assertOffset(span.endUtf16, `${name}.endUtf16`);
  if (span.endUtf16 < span.startUtf16) {
    throw new RangeError(`${name}.endUtf16 must be greater than or equal to ${name}.startUtf16`);
  }
}

function assertBodySpan(body: string, span: Utf16Span, name: string): void {
  assertSpan(span, name);
  if (span.endUtf16 > body.length) {
    throw new RangeError(`${name}.endUtf16 must be within the body`);
  }
  if (!isScalarBoundary(body, span.startUtf16) || !isScalarBoundary(body, span.endUtf16)) {
    throw new RangeError(`${name} boundaries must preserve Unicode scalar values`);
  }
}

function assertBudget(budget: PassageBudget): void {
  assertOffset(budget.maxUtf8Bytes, "maxUtf8Bytes");
  if (budget.maxUnicodeScalars !== undefined) {
    assertOffset(budget.maxUnicodeScalars, "maxUnicodeScalars");
  }
}
