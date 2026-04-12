// ane_transfer_bench.m — ANE Phase 0 unknown #2.
//
// Measures the cost of shuttling a hidden-state buffer between a Metal
// MTLBuffer (shared-storage, which is what flash_moe uses everywhere)
// and an MLMultiArray wrapping the same memory.
//
// The goal: confirm that on Apple Silicon unified memory, we can pass
// hidden-state data to a CoreML prediction without copying. If each
// transition costs <100 μs, the full ANE offload strategy is safe.
// If each transition is 1+ ms, we pay 45× per token in round-trips.
//
// Two strategies tested:
//   (A) initWithDataPointer — zero-copy wrap an existing aligned buffer.
//       Should be near-free.
//   (B) alloc fresh + memcpy — the naive path for comparison.
//
// Build:  clang -O2 -fobjc-arc -framework Foundation -framework CoreML
//               -framework Metal ane_transfer_bench.m -o ane_transfer_bench

#import <Foundation/Foundation.h>
#import <CoreML/CoreML.h>
#import <Metal/Metal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define HIDDEN_DIM 4096
#define NUM_TRIALS 10000

static int cmp_double(const void *a, const void *b) {
    double da = *(const double *)a, db = *(const double *)b;
    return (da > db) - (da < db);
}

static void report(const char *label, double *times, int n) {
    qsort(times, n, sizeof(double), cmp_double);
    double sum = 0;
    for (int i = 0; i < n; i++) sum += times[i];
    printf("  %-28s min=%.3f us  p50=%.3f us  p99=%.3f us  mean=%.3f us\n",
           label, times[0], times[n/2], times[(int)(n*0.99)], sum/n);
}

int main(int argc, const char *argv[]) {
    @autoreleasepool {
        printf("=== ANE Phase 0: data transfer cost ===\n");
        printf("Hidden size: %d floats (%d bytes fp16, %d bytes fp32)\n",
               HIDDEN_DIM, HIDDEN_DIM * 2, HIDDEN_DIM * 4);
        printf("Trials per op: %d\n\n", NUM_TRIALS);

        // --- Setup: Metal device + shared-storage buffer (matches flash_moe pattern) ---
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) { fprintf(stderr, "no Metal device\n"); return 1; }
        size_t bytes_f32 = HIDDEN_DIM * sizeof(float);
        size_t bytes_f16 = HIDDEN_DIM * sizeof(uint16_t);

        id<MTLBuffer> buf_f32 = [device newBufferWithLength:bytes_f32
                                                    options:MTLResourceStorageModeShared];
        id<MTLBuffer> buf_f16 = [device newBufferWithLength:bytes_f16
                                                    options:MTLResourceStorageModeShared];
        float *src_f32 = (float *)buf_f32.contents;
        uint16_t *src_f16 = (uint16_t *)buf_f16.contents;
        for (int i = 0; i < HIDDEN_DIM; i++) { src_f32[i] = (float)i * 0.001f; src_f16[i] = (uint16_t)i; }

        NSError *err = nil;

        double *times = malloc(NUM_TRIALS * sizeof(double));

        // --- Strategy A: initWithDataPointer (zero-copy wrap) ---
        // Use a single persistent array reused across trials.
        NSArray<NSNumber *> *shape = @[@1, @1, @(HIDDEN_DIM)];
        NSArray<NSNumber *> *strides = @[@(HIDDEN_DIM), @(HIDDEN_DIM), @1];

        printf("Strategy A — zero-copy wrap of existing MTLBuffer contents:\n");
        for (int i = 0; i < NUM_TRIALS; i++) {
            double t0 = CFAbsoluteTimeGetCurrent();
            MLMultiArray *a = [[MLMultiArray alloc] initWithDataPointer:src_f16
                                                                  shape:shape
                                                               dataType:MLMultiArrayDataTypeFloat16
                                                                strides:strides
                                                            deallocator:nil
                                                                  error:&err];
            double t1 = CFAbsoluteTimeGetCurrent();
            if (!a) { fprintf(stderr, "initWithDataPointer failed: %s\n", err.localizedDescription.UTF8String); return 1; }
            times[i] = (t1 - t0) * 1e6;  // us
            (void)a;  // ARC releases
        }
        report("zero-copy wrap (fp16)", times, NUM_TRIALS);

        // --- Strategy B: alloc fresh + memcpy (the naive path) ---
        printf("\nStrategy B — alloc fresh MLMultiArray then memcpy from MTLBuffer:\n");
        for (int i = 0; i < NUM_TRIALS; i++) {
            double t0 = CFAbsoluteTimeGetCurrent();
            MLMultiArray *a = [[MLMultiArray alloc] initWithShape:shape
                                                         dataType:MLMultiArrayDataTypeFloat16
                                                            error:&err];
            if (!a) { fprintf(stderr, "initWithShape failed\n"); return 1; }
            memcpy(a.dataPointer, src_f16, bytes_f16);
            double t1 = CFAbsoluteTimeGetCurrent();
            times[i] = (t1 - t0) * 1e6;
            (void)a;
        }
        report("alloc+memcpy (fp16)", times, NUM_TRIALS);

        // --- Strategy C: read back from MLMultiArray into MTLBuffer (memcpy) ---
        printf("\nStrategy C — memcpy from MLMultiArray into existing MTLBuffer:\n");
        // Create one array first (not timed)
        MLMultiArray *out_arr = [[MLMultiArray alloc] initWithShape:shape
                                                           dataType:MLMultiArrayDataTypeFloat16
                                                              error:&err];
        memcpy(out_arr.dataPointer, src_f16, bytes_f16);
        id<MTLBuffer> dst = [device newBufferWithLength:bytes_f16
                                                options:MTLResourceStorageModeShared];
        for (int i = 0; i < NUM_TRIALS; i++) {
            double t0 = CFAbsoluteTimeGetCurrent();
            memcpy(dst.contents, out_arr.dataPointer, bytes_f16);
            double t1 = CFAbsoluteTimeGetCurrent();
            times[i] = (t1 - t0) * 1e6;
        }
        report("readback memcpy (fp16)", times, NUM_TRIALS);

        // --- Strategy D: just allocating a fresh MLMultiArray (to isolate alloc cost) ---
        printf("\nStrategy D — alloc only (no memcpy, isolates MLMultiArray alloc cost):\n");
        for (int i = 0; i < NUM_TRIALS; i++) {
            double t0 = CFAbsoluteTimeGetCurrent();
            MLMultiArray *a = [[MLMultiArray alloc] initWithShape:shape
                                                         dataType:MLMultiArrayDataTypeFloat16
                                                            error:&err];
            double t1 = CFAbsoluteTimeGetCurrent();
            times[i] = (t1 - t0) * 1e6;
            (void)a;
        }
        report("alloc only (fp16)", times, NUM_TRIALS);

        printf("\n=== Interpretation ===\n");
        printf("flash_moe hidden state moves through ~45 linear layers per token.\n");
        printf("If we use Strategy A (zero-copy wrap):\n");
        printf("  - per-layer transfer cost ≈ p50 Strategy A latency\n");
        printf("  - per-token overhead ≈ 45 × p50 Strategy A\n");
        printf("  - budget is ~33 ms/token savings from ANE offload; transitions must stay well under that.\n");
        printf("\n");
        printf("If Strategy A is <100 us per call, per-token overhead is <4.5 ms — viable.\n");
        printf("If Strategy A is ~1 ms per call, per-token overhead is ~45 ms — kills the strategy.\n");

        free(times);
    }
    return 0;
}
