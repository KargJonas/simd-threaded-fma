#include <immintrin.h>  // AVX2 intrinsics
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>

#include <unistd.h>
#include <time.h>


#define N 1000000000
#define WARMUP 30
#define RUNS 50


// AVX2 can process 32 Bytes of data in parallel.
// This means we'll be able to handle eight 4 Byte floats at once.
#define N_PARALLEL 8

size_t size = sizeof(float) * N;
float *a, *b, *c, *result;

void run() {
    size_t last = N - N_PARALLEL;

    // Using _mm256_stream_ps instead of _mm256_storeu_ps to bypass caching and prevent cache pollution.

    // Note: omp likely does not align chunk sizes optimally (to multiples of 8) but the overhead is negligible
    #pragma omp parallel for
    for (size_t i = 0; i <= last; i += N_PARALLEL) {
        _mm256_stream_ps(result + i, _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), _mm256_loadu_ps(c + i)));
    }

    // Always perform tail computation. A single fma op is likely comparable or faster on avg than a condition
    _mm256_stream_ps(result + last, _mm256_fmadd_ps(_mm256_loadu_ps(a + last), _mm256_loadu_ps(b + last), _mm256_loadu_ps(c + last)));
}


int main() {
    // Ensure 32bit aligned memory
    // This is required for some AVX operations like _mm256_stream_ps
    // If you want to use 
    if (posix_memalign((void**)&a, 32, size) != 0 ||
        posix_memalign((void**)&b, 32, size) != 0 ||
        posix_memalign((void**)&c, 32, size) != 0 ||
        posix_memalign((void**)&result, 32, size) != 0
    ) {
        fprintf(stderr, "Memory alignment failed\n");
        exit(EXIT_FAILURE);
    }

    #pragma omp parallel for
    for (size_t i = 0; i < N; i++) {
        a[i] = (float)i;
        b[i] = (float)i;
        c[i] = (float)1;
    }

    // Warmup
    for (int i = 0; i < WARMUP; i++) run();

    // START EXECUTION TIME MEASUREMENT
    printf("Start\n");
    struct timespec begin, end;
    clock_gettime(CLOCK_MONOTONIC, &begin);

    for (int i = 0; i < RUNS; i++) run();

    // END EXECUTION TIME MEASUREMENT
    clock_gettime(CLOCK_MONOTONIC, &end);
    double time_spent = (end.tv_sec - begin.tv_sec) + (end.tv_nsec - begin.tv_nsec) / 1e9;
    printf("Avg execution time: %f seconds\n", time_spent / (float)RUNS);

    // Print the first 10 results. Makes mistakes easier to catch.
    for (size_t i = 0; i < 10; i++) {
        printf("%6.3f * %6.3f + %6.3f = %6.3f\n", a[i], b[i], c[i], result[i]);
    }

    return 0;
}
