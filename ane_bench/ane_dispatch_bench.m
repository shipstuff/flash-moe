// ane_dispatch_bench.m — ANE Phase 0 decision gate.
//
// Measures the wall-clock overhead of calling [MLModel predictionFromFeatures:]
// at per-layer granularity on a compiled ANE super-block, to decide whether
// the flash_moe linear-attention offload strategy is viable.
//
// Success criterion: p90 per-prediction latency is within ~1 ms of the
// anemll-qwen35 reported pure-ANE kernel time (9.28 ms for super-block 0).
// If overhead is >2 ms consistently, the ANE offload strategy is blocked.
//
// Build:  clang -O2 -fobjc-arc -framework Foundation -framework CoreML
//               ane_dispatch_bench.m -o ane_dispatch_bench
// Run:    ./ane_dispatch_bench [path_to_mlmodelc]

#import <Foundation/Foundation.h>
#import <CoreML/CoreML.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static MLMultiArray *makeArray(NSArray<NSNumber *> *shape, NSError **err) {
    MLMultiArray *a = [[MLMultiArray alloc] initWithShape:shape
                                                 dataType:MLMultiArrayDataTypeFloat16
                                                    error:err];
    if (!a) return nil;
    // Zero-fill is safe — super-block 0 handles zero input without NaN.
    memset(a.dataPointer, 0, a.count * sizeof(uint16_t));
    return a;
}

static int cmp_double(const void *a, const void *b) {
    double da = *(const double *)a, db = *(const double *)b;
    return (da > db) - (da < db);
}

int main(int argc, const char *argv[]) {
    @autoreleasepool {
        const char *path = (argc >= 2) ? argv[1]
            : "/Users/carl/projects/turbomoe/flash_moe/ane_bench/superblock0.mlmodelc";
        NSURL *url = [NSURL fileURLWithPath:@(path)];

        NSError *err = nil;
        MLModelConfiguration *cfg_ane = [[MLModelConfiguration alloc] init];
        cfg_ane.computeUnits = MLComputeUnitsCPUAndNeuralEngine;

        printf("=== ANE Phase 0 dispatch benchmark ===\n");
        printf("Model: %s\n", path);
        printf("Compute units: CPUAndNeuralEngine\n\n");

        // --- 1. Load the model ---
        double t_load0 = CFAbsoluteTimeGetCurrent();
        MLModel *model = [MLModel modelWithContentsOfURL:url
                                          configuration:cfg_ane
                                                  error:&err];
        double t_load1 = CFAbsoluteTimeGetCurrent();
        if (!model) {
            NSLog(@"load error: %@", err);
            return 1;
        }
        printf("Model load: %.1f ms (one-shot)\n", (t_load1 - t_load0) * 1000.0);

        // --- 2. Build zero-filled inputs matching the super-block signature ---
        NSMutableDictionary<NSString *, MLMultiArray *> *inputs = [NSMutableDictionary new];
        #define MK(name, ...) do { \
            MLMultiArray *__a = makeArray(@[__VA_ARGS__], &err); \
            if (!__a) { NSLog(@"makeArray %s: %@", name, err); return 1; } \
            inputs[@name] = __a; \
        } while(0)
        MK("hidden_states",   @1, @1, @4096);
        MK("cos",             @1, @1, @1, @64);
        MK("sin",             @1, @1, @1, @64);
        MK("gated_state_0",   @1, @32, @128, @128);
        MK("conv_state_0",    @1, @3, @8192);
        MK("gated_state_1",   @1, @32, @128, @128);
        MK("conv_state_1",    @1, @3, @8192);
        MK("gated_state_2",   @1, @32, @128, @128);
        MK("conv_state_2",    @1, @3, @8192);
        #undef MK

        MLDictionaryFeatureProvider *features =
            [[MLDictionaryFeatureProvider alloc] initWithDictionary:inputs error:&err];
        if (!features) {
            NSLog(@"features error: %@", err);
            return 1;
        }

        // --- 3. Warmup (ANE compilation/placement happens on first call) ---
        printf("\nWarmup (5 calls, first call may cold-compile)...\n");
        for (int i = 0; i < 5; i++) {
            double t0 = CFAbsoluteTimeGetCurrent();
            id<MLFeatureProvider> out = [model predictionFromFeatures:features error:&err];
            double t1 = CFAbsoluteTimeGetCurrent();
            if (!out) {
                NSLog(@"warmup pred %d error: %@", i, err);
                return 1;
            }
            printf("  warmup[%d]: %.3f ms\n", i, (t1 - t0) * 1000.0);
        }

        // --- 4. Benchmark loop: 200 tight back-to-back predictions ---
        const int N = 200;
        double *times = malloc(N * sizeof(double));

        printf("\nBenchmark: %d back-to-back predictions...\n", N);
        double t_bench0 = CFAbsoluteTimeGetCurrent();
        for (int i = 0; i < N; i++) {
            double t0 = CFAbsoluteTimeGetCurrent();
            id<MLFeatureProvider> out = [model predictionFromFeatures:features error:&err];
            double t1 = CFAbsoluteTimeGetCurrent();
            if (!out) {
                NSLog(@"pred %d error: %@", i, err);
                return 1;
            }
            times[i] = (t1 - t0) * 1000.0;  // ms
            (void)out;  // discard; ARC releases
        }
        double t_bench1 = CFAbsoluteTimeGetCurrent();

        // --- 5. Stats ---
        double sum = 0;
        double raw_min = times[0], raw_max = times[0];
        for (int i = 0; i < N; i++) {
            sum += times[i];
            if (times[i] < raw_min) raw_min = times[i];
            if (times[i] > raw_max) raw_max = times[i];
        }
        double mean = sum / N;

        double *sorted = malloc(N * sizeof(double));
        memcpy(sorted, times, N * sizeof(double));
        qsort(sorted, N, sizeof(double), cmp_double);

        printf("\n=== Results ===\n");
        printf("  Wall clock over loop: %.1f ms\n", (t_bench1 - t_bench0) * 1000.0);
        printf("  Throughput:           %.1f predictions/sec\n",
               (double)N / (t_bench1 - t_bench0));
        printf("  Per-prediction:\n");
        printf("    min:  %.3f ms\n", sorted[0]);
        printf("    p50:  %.3f ms\n", sorted[N / 2]);
        printf("    p90:  %.3f ms\n", sorted[(int)(N * 0.90)]);
        printf("    p99:  %.3f ms\n", sorted[(int)(N * 0.99)]);
        printf("    max:  %.3f ms\n", sorted[N - 1]);
        printf("    mean: %.3f ms\n", mean);

        printf("\n=== Decision gate ===\n");
        printf("  anemll-qwen35 reference pure-ANE time (super-block 0): 9.28 ms\n");
        printf("  p50 overhead above reference: %+.3f ms\n", sorted[N / 2] - 9.28);
        printf("  p90 overhead above reference: %+.3f ms\n", sorted[(int)(N * 0.90)] - 9.28);
        printf("\n");
        double overhead_p90 = sorted[(int)(N * 0.90)] - 9.28;
        if (overhead_p90 < 1.0) {
            printf("  VERDICT: p90 overhead < 1 ms — ANE offload is VIABLE.\n");
        } else if (overhead_p90 < 2.0) {
            printf("  VERDICT: p90 overhead 1-2 ms — ANE offload is MARGINAL.\n");
            printf("  Consider the data-transfer cost on top of this.\n");
        } else {
            printf("  VERDICT: p90 overhead > 2 ms — ANE offload is BLOCKED.\n");
            printf("  Pivot to mixed-bit per-expert quantization (priority #2).\n");
        }

        free(sorted);
        free(times);
    }
    return 0;
}
