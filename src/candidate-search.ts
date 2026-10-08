import {
  chunkDocumentAsync,
  extractIntentTerms,
  reciprocalRankFusion,
  RERANK_CANDIDATE_LIMIT,
  validateLexQuery,
  validateSemanticQuery,
  type ChunkStrategy,
  type ExpandedQuery,
  type RankedResult,
  type SearchResult,
  type Store,
  type VectorScanCoverage,
} from "./store.js";
import { getDefaultLlamaCpp } from "./llm.js";
import { parseMetadataFilter, type MetadataFilter } from "./metadata-filter.js";
import type { DocumentMetadata, MetadataScalar } from "./metadata.js";

export type CandidateSearchOptions = {
  query?: string;
  queries?: ExpandedQuery[];
  candidates: {
    rawLimitPerLeg: number;
    group: { metadataKey: string; targetGroupsPerLeg: number };
  };
  collection?: string;
  collections?: string[];
  filter?: MetadataFilter;
  limit?: number;
  candidateLimit?: number;
  minScore?: number;
  rerank?: boolean;
  intent?: string;
  chunkStrategy?: ChunkStrategy;
};

export type CandidateGroup = {
  key: string;
  value: MetadataScalar | null;
  fallback: "filepath" | null;
};

export type CandidateMatch = {
  leg: number;
  query: string;
  queryType: "original" | "lex" | "vec" | "hyde";
  source: "fts" | "vec";
  file: string;
  contentHash: string;
  rawRank: number;
  groupRank: number;
  backendScore: number;
  weight: number;
  rrfContribution: number;
  vectorStartUtf16?: number;
};

export type CandidateHit = {
  group: CandidateGroup;
  file: string;
  contentHash: string;
  displayPath: string;
  title: string;
  metadata: DocumentMetadata;
  context: string | null;
  score: number;
  rrfScore: number;
  rrfRank: number;
  rrfTopRankBonus: number;
  matches: CandidateMatch[];
};

export type CandidateLegCoverage = {
  leg: number;
  query: string;
  source: "fts" | "vec";
  rawLimit: number;
  rawReturned: number;
  rawLimitReached: boolean;
  groupsReturned: number;
  targetGroups: number;
  groupShortfall: number;
  vectorScans: VectorScanCoverage[];
};

export type CandidateSearchResult = {
  results: CandidateHit[];
  coverage: {
    legs: CandidateLegCoverage[];
    fusedGroups: number;
    admittedGroups: number;
    unavailableGroups: number;
    returnedGroups: number;
  };
};

type RetrievalLeg = ExpandedQuery & { queryType: CandidateMatch["queryType"]; weight: number };
type GroupEvidence = { group: CandidateGroup; representative: SearchResult; representativeContribution: number; matches: CandidateMatch[] };

function positiveInteger(value: number, name: string): void {
  if (!Number.isSafeInteger(value) || value < 1) throw new Error(`${name} must be a positive safe integer`);
}

function groupFor(hit: SearchResult, metadataKey: string): CandidateGroup {
  const value = hit.metadata[metadataKey];
  if (typeof value === "string" || typeof value === "number" || typeof value === "boolean") {
    return { key: JSON.stringify(["metadata", metadataKey, typeof value, value]), value, fallback: null };
  }
  return { key: JSON.stringify(["filepath", hit.filepath]), value: null, fallback: "filepath" };
}

function validateQuery(search: ExpandedQuery): void {
  if (!search || !["lex", "vec", "hyde"].includes(search.type) || typeof search.query !== "string" || !search.query.trim()) {
    throw new Error("Each candidate query requires a lex, vec, or hyde type and nonempty query text");
  }
  if (/[\r\n]/.test(search.query)) throw new Error("Candidate queries must be a single line");
  const error = search.type === "lex" ? validateLexQuery(search.query) : validateSemanticQuery(search.query);
  if (error) throw new Error(error);
}

async function retrievalLegs(store: Store, options: CandidateSearchOptions): Promise<RetrievalLeg[]> {
  if (options.queries) {
    return options.queries.map(search => ({ ...search, queryType: search.type, weight: 1 }));
  }
  const query = options.query!;
  const expanded = await store.expandQuery(query);
  expanded.forEach(validateQuery);
  return [
    { type: "lex", query, queryType: "original", weight: 2 },
    { type: "vec", query, queryType: "original", weight: 2 },
    ...expanded.map(search => ({ ...search, queryType: search.type, weight: 1 })),
  ];
}

/** Group each eligible retrieval leg before fusion and the candidate cutoff. */
export async function searchCandidates(store: Store, options: CandidateSearchOptions): Promise<CandidateSearchResult> {
  if ((options.query !== undefined) === (options.queries !== undefined)) {
    throw new Error("searchCandidates() requires exactly one of query or queries");
  }
  if (options.query !== undefined) {
    validateQuery({ type: "lex", query: options.query });
    validateQuery({ type: "vec", query: options.query });
  }
  if (options.queries !== undefined) {
    if (!Array.isArray(options.queries) || options.queries.length === 0) throw new Error("queries must be a nonempty array");
    options.queries.forEach(validateQuery);
  }
  positiveInteger(options.candidates?.rawLimitPerLeg, "rawLimitPerLeg");
  positiveInteger(options.candidates?.group?.targetGroupsPerLeg, "targetGroupsPerLeg");
  const metadataKey = options.candidates.group.metadataKey;
  parseMetadataFilter({ field: metadataKey, operator: "exists", value: true });
  const filter = options.filter === undefined ? undefined : parseMetadataFilter(options.filter);
  const limit = options.limit ?? 10;
  const candidateLimit = options.candidateLimit ?? RERANK_CANDIDATE_LIMIT;
  positiveInteger(limit, "limit");
  positiveInteger(candidateLimit, "candidateLimit");
  const minScore = options.minScore ?? 0;
  if (!Number.isFinite(minScore) || minScore < 0 || minScore > 1) throw new Error("minScore must be between 0 and 1");
  const collections = [...(options.collection ? [options.collection] : []), ...(options.collections ?? [])];
  const scope = collections.length ? collections : undefined;
  const legs = await retrievalLegs(store, options);
  const lists: RankedResult[][] = [];
  const coverage: CandidateLegCoverage[] = [];
  const evidence = new Map<string, GroupEvidence>();
  const rawLimit = options.candidates.rawLimitPerLeg;
  const targetGroups = options.candidates.group.targetGroupsPerLeg;

  for (const [legIndex, leg] of legs.entries()) {
    const vectorScans: VectorScanCoverage[] = [];
    const hits = leg.type === "lex"
      ? store.searchFTS(leg.query, rawLimit, scope, filter, { includeBody: false })
      : await store.searchVec(leg.query, (store.llm ?? getDefaultLlamaCpp()).embedModelName, rawLimit,
        scope, undefined, undefined, filter, { includeBody: false, onVectorScan: scan => vectorScans.push(scan) });
    if (options.queries && hits.length && lists.every(list => list.length === 0)) leg.weight = 2;
    const groups = new Set<string>();
    const ranked: RankedResult[] = [];
    for (const [rawIndex, hit] of hits.entries()) {
      const group = groupFor(hit, metadataKey);
      if (groups.has(group.key)) continue;
      if (groups.size === targetGroups) break;
      groups.add(group.key);
      const groupRank = groups.size;
      const match: CandidateMatch = {
        leg: legIndex, query: leg.query, queryType: leg.queryType, source: hit.source,
        file: hit.filepath, contentHash: hit.hash, rawRank: rawIndex + 1, groupRank,
        backendScore: hit.score, weight: leg.weight, rrfContribution: leg.weight / (60 + groupRank),
        ...(hit.chunkPos === undefined ? {} : { vectorStartUtf16: hit.chunkPos }),
      };
      const existing = evidence.get(group.key);
      if (existing) {
        existing.matches.push(match);
        // The largest RRF contribution wins; input leg order breaks ties.
        if (match.rrfContribution > existing.representativeContribution) {
          existing.representative = hit;
          existing.representativeContribution = match.rrfContribution;
        }
      } else {
        evidence.set(group.key, { group, representative: hit, representativeContribution: match.rrfContribution, matches: [match] });
      }
      ranked.push({ file: group.key, displayPath: hit.displayPath, title: hit.title, body: "", score: hit.score });
    }
    lists.push(ranked);
    coverage.push({
      leg: legIndex, query: leg.query, source: leg.type === "lex" ? "fts" : "vec",
      rawLimit, rawReturned: hits.length, rawLimitReached: hits.length === rawLimit,
      groupsReturned: groups.size, targetGroups, groupShortfall: Math.max(0, targetGroups - groups.size), vectorScans,
    });
  }

  const fused = reciprocalRankFusion(lists, legs.map(leg => leg.weight));
  const admitted = fused.slice(0, candidateLimit);
  const primaryQuery = options.query ?? options.queries!.find(search => search.type === "lex")?.query ?? options.queries![0]!.query;
  const queryTerms = primaryQuery.toLowerCase().split(/\s+/).filter(term => term.length > 2);
  const intentTerms = options.intent ? extractIntentTerms(options.intent) : [];
  const candidates: CandidateHit[] = [];
  const rerankInputs: { file: string; text: string }[] = [];
  const rankByGroup = new Map(admitted.map((hit, rank) => [hit.file, rank]));
  const bodyOf = store.db.prepare(`
    SELECT c.doc AS body FROM content c JOIN documents d ON d.hash = c.hash
    WHERE d.active = 1 AND c.hash = ? AND 'qmd://' || d.collection || '/' || d.path = ? LIMIT 1
  `);
  for (const [rank, fusedHit] of admitted.entries()) {
    const groupEvidence = evidence.get(fusedHit.file)!;
    const representative = groupEvidence.representative;
    const content = bodyOf.get<{ body: string }>(representative.hash, representative.filepath);
    if (!content) continue;
    const body = content.body;
    const chunks = await chunkDocumentAsync(body, undefined, undefined, undefined, representative.filepath, options.chunkStrategy);
    let best = chunks[0];
    let bestScore = -1;
    for (const chunk of chunks) {
      const text = chunk.text.toLowerCase();
      const score = queryTerms.reduce((sum, term) => sum + Number(text.includes(term)), 0)
        + intentTerms.reduce((sum, term) => sum + 0.5 * Number(text.includes(term)), 0);
      if (score > bestScore) { best = chunk; bestScore = score; }
    }
    rerankInputs.push({ file: fusedHit.file, text: best?.text ?? body });
    candidates.push({
      group: groupEvidence.group, file: representative.filepath, contentHash: representative.hash,
      displayPath: representative.displayPath, title: representative.title, metadata: representative.metadata,
      context: representative.context, score: 1 / (rank + 1), rrfScore: fusedHit.score, rrfRank: rank + 1,
      rrfTopRankBonus: groupEvidence.matches.some(match => match.groupRank === 1) ? 0.05
        : groupEvidence.matches.some(match => match.groupRank <= 3) ? 0.02 : 0,
      matches: groupEvidence.matches,
    });
  }
  if (options.rerank !== false && rerankInputs.length) {
    const reranked = await store.rerank(primaryQuery, rerankInputs, undefined, options.intent);
    const scores = new Map(reranked.map(hit => [hit.file, hit.score]));
    for (const hit of candidates) {
      const rank = rankByGroup.get(hit.group.key)!;
      const weight = rank < 3 ? 0.75 : rank < 10 ? 0.6 : 0.4;
      hit.score = weight * hit.score + (1 - weight) * (scores.get(hit.group.key) ?? 0);
    }
    candidates.sort((a, b) => b.score - a.score);
  }
  const results = candidates.filter(hit => hit.score >= minScore).slice(0, limit);
  return { results, coverage: { legs: coverage, fusedGroups: fused.length, admittedGroups: admitted.length,
    unavailableGroups: admitted.length - candidates.length, returnedGroups: results.length } };
}
