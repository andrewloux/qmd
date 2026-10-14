/**
 * metadata-coverage.test-d.ts - Compile-time shape of the public coverage API,
 * imported the way SDK consumers import it. Checked by vitest's typecheck pass.
 */

import { describe, test, expectTypeOf } from "vitest";
import type {
  MetadataCoverageOptions,
  MetadataCoverageResult,
  MetadataFilter,
  MetadataTypeCoverage,
  MetadataValueType,
  QMDStore,
} from "../src/index.js";

describe("getMetadataCoverage", () => {
  test("takes one key with listMetadata's scope and filter", () => {
    expectTypeOf<QMDStore["getMetadataCoverage"]>().parameters.toEqualTypeOf<[MetadataCoverageOptions]>();
    expectTypeOf<MetadataCoverageOptions>().toEqualTypeOf<{ key: string; collection?: string | string[]; filter?: MetadataFilter }>();
  });

  test("resolves the eligible and covered counts with one summary per stored type", () => {
    expectTypeOf<QMDStore["getMetadataCoverage"]>().returns.toEqualTypeOf<Promise<MetadataCoverageResult>>();
    expectTypeOf<MetadataCoverageResult>().toEqualTypeOf<{ key: string; eligibleDocuments: number; coveredDocuments: number; types: MetadataTypeCoverage[] }>();
    expectTypeOf<MetadataTypeCoverage>().toEqualTypeOf<{ type: MetadataValueType; documents: number; distinctValues: number; multiValued: boolean }>();
  });
});
