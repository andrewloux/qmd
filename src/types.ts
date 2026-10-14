export type EmbedFailure = {
  path: string;
  hash: string;
  seq: number;
  attempts: number;
  reason: string;
};

export type EmbedProgress = {
  chunksEmbedded: number;
  totalChunks: number;
  bytesProcessed: number;
  totalBytes: number;
  /** Active failed chunks still awaiting a successful retry. */
  errors: number;
  failures?: EmbedFailure[];
};
