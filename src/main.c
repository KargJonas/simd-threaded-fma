#include <immintrin.h>  // AVX2 intrinsics
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>

#include <unistd.h>
#include <time.h>

#include "./simd.h"

// This program manages to perform a billion mulaccs in 0.42s (avg)
// Each one of the four buffers involved holds 4Gb worth of floats.
// In total, we read 12Gb and write 4Gb in 0.42s
// This means we (theoretically) read with 55Gb/s and write with 18Gb/s
// which should be impossible given my measured 10Gb/s mem bandwidth.
// Caching is out of the picture because of the workload type
// (and because i explicitly bypass caching using _mm256_stream_ps).
// Maybe there is a logical error in my calculation somewhere...


// AVX2 can process 32 Bytes of data in parallel.
// This means we'll be able to handle eight 4 Byte floats at once.
#define N_PARALLEL 8

T_SIMD_ARR_OP(arr_fma, _mm256_fmadd_ps) // this is prob the coolest shit ever; SIMD fused mulacc

typedef struct {
    float* a;
    float* b;
    float* c;
    float* result;
    size_t nelem;
} ThreadInfo;

void* add_arrays(void* arg) {
    ThreadInfo *thread_info = (ThreadInfo*)arg;
    arr_fma(thread_info->a, thread_info->b, thread_info->c, thread_info->result, thread_info->nelem);
    pthread_exit(NULL);
}

// n threads = n cores
#define num_threads 10

#define N 1000000000
size_t size = sizeof(float) * N;
float *a, *b, *c, *result;

void run() {
    pthread_t threads[num_threads];
    ThreadInfo thread_info[num_threads];
    size_t chunk_size = N / num_threads;
    size_t remaining = N % num_threads;
    size_t current_start = 0;

    for (size_t i = 0; i < num_threads; i++) {
        thread_info[i].a = a + current_start; // Offsetting the ranges that the individual threads operate on
        thread_info[i].b = b + current_start;
        thread_info[i].c = c + current_start;
        thread_info[i].result = result + current_start;
        thread_info[i].nelem = chunk_size + (i < remaining ? 1 : 0); // Distribute the remaining elements between all threads

        // todo: aligning as many threads with problem sizes that are multiples of N_PARALLEL can improve perf marginally
        current_start += thread_info[i].nelem;

        int rc = pthread_create(&threads[i], NULL, add_arrays, (void*)&thread_info[i]);
        if (rc) {
            fprintf(stderr, "Error: Unable to create thread %ld, return code %d\n", i, rc);
            exit(EXIT_FAILURE);
        }
    }

    for (size_t i = 0; i < num_threads; i++) {
        pthread_join(threads[i], NULL);
    }
}

int main() {
    // Aligned malloc
    if (posix_memalign((void**)&a, 32, size) != 0 ||
        posix_memalign((void**)&b, 32, size) != 0 ||
        posix_memalign((void**)&c, 32, size) != 0 ||
        posix_memalign((void**)&result, 32, size) != 0
    ) {
        fprintf(stderr, "Memory alignment failed\n");
        exit(EXIT_FAILURE);
    }

    for (size_t i = 0; i < N; i++) {
        a[i] = (float)i;
        b[i] = (float)i;
        c[i] = (float)1;
    }

    // Warmup
    for (int i = 0; i < 30; i++) run();

    // START EXECUTION TIME MEASUREMENT
    printf("Start\n");
    struct timespec begin, end;
    clock_gettime(CLOCK_MONOTONIC, &begin);

    // run();
    for (int i = 0; i < 20; i++) run();

    // END EXECUTION TIME MEASUREMENT
    clock_gettime(CLOCK_MONOTONIC, &end);
    double time_spent = (end.tv_sec - begin.tv_sec) + (end.tv_nsec - begin.tv_nsec) / 1e9;
    printf("Avg execution time: %f seconds\n", time_spent / 20.);

    for (size_t i = 0; i < 10; i++) {
        printf("%6.3f * %6.3f + %6.3f = %6.3f\n", a[i], b[i], c[i], result[i]);
    }

    return 0;
}
