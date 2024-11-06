#include <stdio.h>
#include <stdlib.h>

#include <unistd.h>
#include <time.h>

#define N 1000000000
size_t size = sizeof(float) * N;
float *a, *b, *c, *result;

void run() {
    #pragma omp parallel for
    for (size_t i = 0; i < N; i++) {
        result[i] = a[i] * b[i] + c[i];
    }
}

int main() {
    a = malloc(size);
    b = malloc(size);
    c = malloc(size);
    result = malloc(size);

    for (size_t i = 0; i < N; i++) {
        a[i] = (float)i;
        b[i] = (float)i;
        c[i] = 1.;
    }

    // Warmup
    for (int i = 0; i < 30; i++) run();

    // START EXECUTION TIME MEASUREMENT
    printf("Start\n");
    struct timespec begin, end;
    clock_gettime(CLOCK_MONOTONIC, &begin);

    // run();
    for (int i = 0; i < 10; i++) run();

    // END EXECUTION TIME MEASUREMENT
    clock_gettime(CLOCK_MONOTONIC, &end);
    double time_spent = (end.tv_sec - begin.tv_sec) + (end.tv_nsec - begin.tv_nsec) / 1e9;
    printf("Avg execution time: %f seconds\n", time_spent / 10.);
    
    // for (size_t i = 0; i < 10; i++) {
    //     printf("%6.3f * %6.3f + %6.3f = %6.3f\n", a[i], b[i], c[i], result[i]);
    // }

    return 0;
}
