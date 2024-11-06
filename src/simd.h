// note: using _mm256_stream_ps instead of _mm256_storeu_ps bypasses cache.
//       which prevents cache pollution with primitive array operations like these

// todo: handle cases where array length is less than N_PARALLEL
// Binary SIMD operations on arrays
#define B_SIMD_OP(NAME, OP)  \
void NAME(float* a, float* b, float* result, size_t nelem) { \
    size_t last = nelem - N_PARALLEL; /* NOTE: this can underflow if nelem is less than N_PARALLEL */ \
    size_t i; for (i = 0; i <= nelem - N_PARALLEL; i += N_PARALLEL) \
        _mm256_stream_ps(result + i, OP(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i))); \
    if (i != nelem) _mm256_storeu_ps(result + last, OP(_mm256_loadu_ps(a + last), _mm256_loadu_ps(b + last))); \
} \

// Ternary SIMD operations on arrays
#define T_SIMD_ARR_OP(NAME, OP) \
void NAME(float* a, float* b, float* c, float* result, size_t nelem) { \
    size_t last = nelem - N_PARALLEL; \
    size_t i; for (i = 0; i <= last; i += N_PARALLEL) \
        _mm256_stream_ps(result + i, OP(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), _mm256_loadu_ps(c + i))); \
    if (i != nelem) _mm256_storeu_ps(result + last, OP(_mm256_loadu_ps(a + last), _mm256_loadu_ps(b + last), _mm256_loadu_ps(c + last))); \
}
