#include "libdivide.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if __GNUC__
#define NOINLINE __attribute__((__noinline__))
#else
#define NOINLINE
#endif

#define NANOSEC_PER_SEC 1000000000ULL
#define NANOSEC_PER_USEC 1000ULL
#define NANOSEC_PER_MILLISEC 1000000ULL

#ifdef __cplusplus
using namespace libdivide;
#endif

#if defined(_WIN32) || defined(WIN32)
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN 1
#define VC_EXTRALEAN 1
#include <windows.h>
#include <mmsystem.h>
#define LIBDIVIDE_WINDOWS 1
#pragma comment(lib, "winmm")
#endif

#if ! LIBDIVIDE_WINDOWS
#include <sys/time.h> //for gettimeofday()
#endif

#if LIBDIVIDE_USE_SSE2
#define FUNC_VECTOR64(x)   NULL
#define FUNC_VECTOR128(x)  (x)
#define FUNC_VECTOR256(x)  NULL
#elif LIBDIVIDE_USE_NEON
#define FUNC_VECTOR64(x)   (x)
#define FUNC_VECTOR128(x)  (x)
#define FUNC_VECTOR256(x)  (x)
#else
#define FUNC_VECTOR64(x)   NULL
#define FUNC_VECTOR128(x)  NULL
#define FUNC_VECTOR256(x)  NULL
#endif

struct random_state {
    uint32_t hi;
    uint32_t lo;
};

#define SEED {2147483563, 2147483563 ^ 0x49616E42}
#define ITERATIONS (1 << 19)

#define GEN_ITERATIONS (1 << 16)

uint64_t sGlobalUInt64;

static uint32_t my_random(struct random_state *state) {
    state->hi = (state->hi << 16) + (state->hi >> 16);
    state->hi += state->lo;
    state->lo += state->hi;
    return state->hi;
}

#if LIBDIVIDE_WINDOWS
static LARGE_INTEGER gPerfCounterFreq;
#endif

#if ! LIBDIVIDE_WINDOWS
static uint64_t nanoseconds(void) {
    struct timeval now;
    gettimeofday(&now, NULL);
    return now.tv_sec * NANOSEC_PER_SEC + now.tv_usec * NANOSEC_PER_USEC;
}
#endif

struct FunctionParams_t {
    void *d; //a pointer to e.g. a uint32_t
    void *denomPtr; // a pointer to e.g. libdivide_u32_t
    const void *data; // a pointer to the data to be divided
};

struct time_result {
    uint64_t time;
    uint64_t result;
};

static struct time_result time_function(uint64_t (*func)(struct FunctionParams_t*), struct FunctionParams_t *params) {
    struct time_result tresult;
#if LIBDIVIDE_WINDOWS
    LARGE_INTEGER start, end;
    QueryPerformanceCounter(&start);
    uint64_t result = func(params);
    QueryPerformanceCounter(&end);
    uint64_t diff = end.QuadPart - start.QuadPart;
    sGlobalUInt64 += result;
    tresult.result = result;
    tresult.time = (diff * 1000000000) / gPerfCounterFreq.QuadPart;
#else
    uint64_t start = nanoseconds();
    uint64_t result = func(params);
    uint64_t end = nanoseconds();
    uint64_t diff = end - start;
    sGlobalUInt64 += result;
    tresult.result = result;
    tresult.time = diff;
#endif
    return tresult;
}

//U32

NOINLINE static uint64_t mine_u32(struct FunctionParams_t *params) {
    unsigned iter;
    const struct libdivide_u32_t denom = *(struct libdivide_u32_t *)params->denomPtr;
    const uint32_t *data = (const uint32_t *)params->data;
    uint32_t sum = 0;
    for (iter = 0; iter < ITERATIONS; iter++) {
        uint32_t numer = data[iter];
        sum += libdivide_u32_do(numer, &denom);
    }
    return sum;
}

#if LIBDIVIDE_USE_SSE2
NOINLINE static uint64_t mine_4u32_vector(struct FunctionParams_t *params) {
    unsigned iter;
    const struct libdivide_u32_t denom = *(struct libdivide_u32_t *)params->denomPtr;
    const uint32_t *data = (const uint32_t *)params->data;
    __m128i sumX = _mm_setzero_si128();
    for (iter = 0; iter < ITERATIONS; iter+=4) {
        __m128i numers = *((const __m128i*)(data + iter));
        sumX = _mm_add_epi32(sumX, libdivide_u32_do_vector(numers, &denom));
    }
    const uint32_t *comps = (const uint32_t *)&sumX;
    return comps[0] + comps[1] + comps[2] + comps[3];
}

NOINLINE static uint64_t mine_4u32_vector_unswitched(struct FunctionParams_t *params) {
    unsigned iter;
    const struct libdivide_u32_t denom = *(struct libdivide_u32_t *)params->denomPtr;
    const uint32_t *data = (const uint32_t *)params->data;
    __m128i sumX = _mm_setzero_si128();
    int algo = libdivide_u32_get_algorithm(&denom);
    if (algo == 0) {
        for (iter = 0; iter < ITERATIONS; iter+=4) {
            __m128i numers = *((const __m128i*)(data + iter));
            sumX = _mm_add_epi32(sumX, libdivide_u32_do_vector_alg0(numers, &denom));
        }
    }
    else if (algo == 1) {
        for (iter = 0; iter < ITERATIONS; iter+=4) {
            __m128i numers = *((const __m128i*)(data + iter));
            sumX = _mm_add_epi32(sumX, libdivide_u32_do_vector_alg1(numers, &denom));
        }
    }
    else if (algo == 2) {
        for (iter = 0; iter < ITERATIONS; iter+=4) {
            __m128i numers = *((const __m128i*)(data + iter));
            sumX = _mm_add_epi32(sumX, libdivide_u32_do_vector_alg2(numers, &denom));
        }
    }
    const uint32_t *comps = (const uint32_t *)&sumX;
    return comps[0] + comps[1] + comps[2] + comps[3];
}
#elif LIBDIVIDE_USE_NEON
NOINLINE static uint64_t mine_2u32_vector(struct FunctionParams_t *params) {
    unsigned iter;
    const struct libdivide_u32_t denom = *(struct libdivide_u32_t *)params->denomPtr;
    const uint32_t *data = (const uint32_t *)params->data;
    uint32x2_t sumX = vdup_n_u32(0);
    for (iter = 0; iter < ITERATIONS; iter+=2) {
        uint32x2_t numers = *((const uint32x2_t*)(data + iter));
        sumX = vadd_u32(sumX, libdivide_2u32_do_vector(numers, &denom));
    }
    const uint32_t *comps = (const uint32_t *)&sumX;
    return comps[0] + comps[1];
}
NOINLINE static uint64_t mine_4u32_vector(struct FunctionParams_t *params) {
    unsigned iter;
    const struct libdivide_u32_t denom = *(struct libdivide_u32_t *)params->denomPtr;
    const uint32_t *data = (const uint32_t *)params->data;
    uint32x4_t sumX = vdupq_n_u32(0);
    for (iter = 0; iter < ITERATIONS; iter+=4) {
        uint32x4_t numers = *((const uint32x4_t*)(data + iter));
        sumX = vaddq_u32(sumX, libdivide_4u32_do_vector(numers, &denom));
    }
    const uint32_t *comps = (const uint32_t *)&sumX;
    return comps[0] + comps[1] + comps[2] + comps[3];
}
NOINLINE static uint64_t mine_8u32_vector(struct FunctionParams_t *params) {
    unsigned iter;
    const struct libdivide_u32_t denom = *(struct libdivide_u32_t *)params->denomPtr;
    const uint32_t *data = (const uint32_t *)params->data;
    uint32x4_t sumX = vdupq_n_u32(0);
    for (iter = 0; iter < ITERATIONS; iter+=8) {
        uint32x4x2_t numers = *((const uint32x4x2_t*)(data + iter));
        uint32x4x2_t result = libdivide_8u32_do_vector(numers, &denom);
        sumX = vaddq_u32(sumX, result.val[0]);
        sumX = vaddq_u32(sumX, result.val[1]);
    }
    const uint32_t *comps = (const uint32_t *)&sumX;
    return comps[0] + comps[1] + comps[2] + comps[3];
}

NOINLINE static uint64_t mine_4u32_vector_unswitched(struct FunctionParams_t *params) {
    unsigned iter;
    const struct libdivide_u32_t denom = *(struct libdivide_u32_t *)params->denomPtr;
    const uint32_t *data = (const uint32_t *)params->data;
    uint32x4_t sumX = vdupq_n_u32(0);
    int algo = libdivide_u32_get_algorithm(&denom);
    if (algo == 0) {
        for (iter = 0; iter < ITERATIONS; iter+=4) {
            uint32x4_t numers = *((const uint32x4_t*)(data + iter));
            sumX = vaddq_u32(sumX, libdivide_4u32_do_vector_alg0(numers, &denom));
        }
    }
    else if (algo == 1) {
        for (iter = 0; iter < ITERATIONS; iter+=4) {
            uint32x4_t numers = *((const uint32x4_t*)(data + iter));
            sumX = vaddq_u32(sumX, libdivide_4u32_do_vector_alg1(numers, &denom));
        }
    }
    else if (algo == 2) {
        for (iter = 0; iter < ITERATIONS; iter+=4) {
            uint32x4_t numers = *((const uint32x4_t*)(data + iter));
            sumX = vaddq_u32(sumX, libdivide_4u32_do_vector_alg2(numers, &denom));
        }
    }
    const uint32_t *comps = (const uint32_t *)&sumX;
    return comps[0] + comps[1] + comps[2] + comps[3];
}
#endif

NOINLINE static uint64_t mine_u32_unswitched(struct FunctionParams_t *params) {
    unsigned iter;
    const struct libdivide_u32_t denom = *(struct libdivide_u32_t *)params->denomPtr;
    const uint32_t *data = (const uint32_t *)params->data;
    uint32_t sum = 0;
    int algo = libdivide_u32_get_algorithm(&denom);
    if (algo == 0) {
        for (iter = 0; iter < ITERATIONS; iter++) {
            uint32_t numer = data[iter];
            sum += libdivide_u32_do_alg0(numer, &denom);
        }
    }
    else if (algo == 1) {
        for (iter = 0; iter < ITERATIONS; iter++) {
            uint32_t numer = data[iter];
            sum += libdivide_u32_do_alg1(numer, &denom);
        }
    }
    else if (algo == 2) {
        for (iter = 0; iter < ITERATIONS; iter++) {
            uint32_t numer = data[iter];
            sum += libdivide_u32_do_alg2(numer, &denom);
        }
    }

    return sum;
}

NOINLINE static uint64_t his_u32(struct FunctionParams_t *params) {
    unsigned iter;
    const uint32_t *data = (const uint32_t *)params->data;
    const uint32_t d = *(uint32_t *)params->d;
    uint32_t sum = 0;
    for (iter = 0; iter < ITERATIONS; iter++) {
        uint32_t numer = data[iter];
        sum += numer / d;
    }
    return sum;
}

NOINLINE static uint64_t mine_u32_generate(struct FunctionParams_t *params) {
    uint32_t *dPtr = (uint32_t *)params->d;
    struct libdivide_u32_t *denomPtr = (struct libdivide_u32_t *)params->denomPtr;
    unsigned iter;
    for (iter = 0; iter < GEN_ITERATIONS; iter++) {
        *denomPtr = libdivide_u32_gen(*dPtr);
    }
    return *dPtr;
}

//S32

NOINLINE static uint64_t mine_s32(struct FunctionParams_t *params) {
    unsigned iter;
    const struct libdivide_s32_t denom = *(struct libdivide_s32_t *)params->denomPtr;
    const int32_t *data = (const int32_t *)params->data;
    int32_t sum = 0;
    for (iter = 0; iter < ITERATIONS; iter++) {
        int32_t numer = data[iter];
        sum += libdivide_s32_do(numer, &denom);
    }
    return sum;
}

#if LIBDIVIDE_USE_SSE2
NOINLINE static uint64_t mine_4s32_vector(struct FunctionParams_t *params) {
    unsigned iter;
    __m128i sumX = _mm_setzero_si128();
    const struct libdivide_s32_t denom = *(struct libdivide_s32_t *)params->denomPtr;
    const int32_t *data = (const int32_t *)params->data;
    for (iter = 0; iter < ITERATIONS; iter+=4) {
        __m128i numers = *((const __m128i*)(data + iter));
        sumX = _mm_add_epi32(sumX, libdivide_s32_do_vector(numers, &denom));
    }
    const int32_t *comps = (const int32_t *)&sumX;
    int32_t sum = comps[0] + comps[1] + comps[2] + comps[3];
    return sum;
}

NOINLINE static uint64_t mine_4s32_vector_unswitched(struct FunctionParams_t *params) {
    unsigned iter;
    __m128i sumX = _mm_setzero_si128();
    const struct libdivide_s32_t denom = *(struct libdivide_s32_t *)params->denomPtr;
    const int32_t *data = (const int32_t *)params->data;
    int algo = libdivide_s32_get_algorithm(&denom);
    if (algo == 0) {
        for (iter = 0; iter < ITERATIONS; iter+=4) {
            __m128i numers = *((const __m128i*)(data + iter));
            sumX = _mm_add_epi32(sumX, libdivide_s32_do_vector_alg0(numers, &denom));
        }
    }
    else if (algo == 1) {
        for (iter = 0; iter < ITERATIONS; iter+=4) {
            __m128i numers = *((const __m128i*)(data + iter));
            sumX = _mm_add_epi32(sumX, libdivide_s32_do_vector_alg1(numers, &denom));
        }
    }
    else if (algo == 2) {
        for (iter = 0; iter < ITERATIONS; iter+=4) {
            __m128i numers = *((const __m128i*)(data + iter));
            sumX = _mm_add_epi32(sumX, libdivide_s32_do_vector_alg2(numers, &denom));
        }
    }
    else if (algo == 3) {
        for (iter = 0; iter < ITERATIONS; iter+=4) {
            __m128i numers = *((const __m128i*)(data + iter));
            sumX = _mm_add_epi32(sumX, libdivide_s32_do_vector_alg3(numers, &denom));
        }
    }
    else if (algo == 4) {
        for (iter = 0; iter < ITERATIONS; iter+=4) {
            __m128i numers = *((const __m128i*)(data + iter));
            sumX = _mm_add_epi32(sumX, libdivide_s32_do_vector_alg4(numers, &denom));
        }
    }
    const int32_t *comps = (const int32_t *)&sumX;
    int32_t sum = comps[0] + comps[1] + comps[2] + comps[3];
    return sum;
}
#elif LIBDIVIDE_USE_NEON
NOINLINE static uint64_t mine_2s32_vector(struct FunctionParams_t *params) {
    unsigned iter;
    int32x2_t sumX = vdup_n_s32(0);
    const struct libdivide_s32_t denom = *(struct libdivide_s32_t *)params->denomPtr;
    const int32_t *data = (const int32_t *)params->data;
    for (iter = 0; iter < ITERATIONS; iter+=2) {
        int32x2_t numers = *((const int32x2_t*)(data + iter));
        sumX = vadd_s32(sumX, libdivide_2s32_do_vector(numers, &denom));
    }
    const int32_t *comps = (const int32_t *)&sumX;
    int32_t sum = comps[0] + comps[1];
    return sum;
}
NOINLINE static uint64_t mine_4s32_vector(struct FunctionParams_t *params) {
    unsigned iter;
    int32x4_t sumX = vdupq_n_s32(0);
    const struct libdivide_s32_t denom = *(struct libdivide_s32_t *)params->denomPtr;
    const int32_t *data = (const int32_t *)params->data;
    for (iter = 0; iter < ITERATIONS; iter+=4) {
        int32x4_t numers = *((const int32x4_t*)(data + iter));
        sumX = vaddq_s32(sumX, libdivide_4s32_do_vector(numers, &denom));
    }
    const int32_t *comps = (const int32_t *)&sumX;
    int32_t sum = comps[0] + comps[1] + comps[2] + comps[3];
    return sum;
}
NOINLINE static uint64_t mine_8s32_vector(struct FunctionParams_t *params) {
    unsigned iter;
    int32x4_t sumX = vdupq_n_s32(0);
    const struct libdivide_s32_t denom = *(struct libdivide_s32_t *)params->denomPtr;
    const int32_t *data = (const int32_t *)params->data;
    for (iter = 0; iter < ITERATIONS; iter+=8) {
        int32x4x2_t numers = *((const int32x4x2_t*)(data + iter));
        int32x4x2_t result = libdivide_8s32_do_vector(numers, &denom);
        sumX = vaddq_s32(sumX, result.val[0]);
        sumX = vaddq_s32(sumX, result.val[1]);
    }
    const int32_t *comps = (const int32_t *)&sumX;
    int32_t sum = comps[0] + comps[1] + comps[2] + comps[3];
    return sum;
}

NOINLINE static uint64_t mine_4s32_vector_unswitched(struct FunctionParams_t *params) {
    unsigned iter;
    int32x4_t sumX = vdupq_n_s32(0);
    const struct libdivide_s32_t denom = *(struct libdivide_s32_t *)params->denomPtr;
    const int32_t *data = (const int32_t *)params->data;
    int algo = libdivide_s32_get_algorithm(&denom);
    if (algo == 0) {
        for (iter = 0; iter < ITERATIONS; iter+=4) {
            int32x4_t numers = *((const int32x4_t*)(data + iter));
            sumX = vaddq_s32(sumX, libdivide_4s32_do_vector_alg0(numers, &denom));
        }
    }
    else if (algo == 1) {
        for (iter = 0; iter < ITERATIONS; iter+=4) {
            int32x4_t numers = *((const int32x4_t*)(data + iter));
            sumX = vaddq_s32(sumX, libdivide_4s32_do_vector_alg1(numers, &denom));
        }
    }
    else if (algo == 2) {
        for (iter = 0; iter < ITERATIONS; iter+=4) {
            int32x4_t numers = *((const int32x4_t*)(data + iter));
            sumX = vaddq_s32(sumX, libdivide_4s32_do_vector_alg2(numers, &denom));
        }
    }
    else if (algo == 3) {
        for (iter = 0; iter < ITERATIONS; iter+=4) {
            int32x4_t numers = *((const int32x4_t*)(data + iter));
            sumX = vaddq_s32(sumX, libdivide_4s32_do_vector_alg3(numers, &denom));
        }
    }
    else if (algo == 4) {
        for (iter = 0; iter < ITERATIONS; iter+=4) {
            int32x4_t numers = *((const int32x4_t*)(data + iter));
            sumX = vaddq_s32(sumX, libdivide_4s32_do_vector_alg4(numers, &denom));
        }
    }
    const int32_t *comps = (const int32_t *)&sumX;
    int32_t sum = comps[0] + comps[1] + comps[2] + comps[3];
    return sum;
}
#endif

NOINLINE static uint64_t mine_s32_unswitched(struct FunctionParams_t *params) {
    unsigned iter;
    int32_t sum = 0;
    const struct libdivide_s32_t denom = *(struct libdivide_s32_t *)params->denomPtr;
    const int32_t *data = (const int32_t *)params->data;
    int algo = libdivide_s32_get_algorithm(&denom);
    if (algo == 0) {
        for (iter = 0; iter < ITERATIONS; iter++) {
            int32_t numer = data[iter];
            sum += libdivide_s32_do_alg0(numer, &denom);
        }
    }
    else if (algo == 1) {
        for (iter = 0; iter < ITERATIONS; iter++) {
            int32_t numer = data[iter];
            sum += libdivide_s32_do_alg1(numer, &denom);
        }
    }
    else if (algo == 2) {
        for (iter = 0; iter < ITERATIONS; iter++) {
            int32_t numer = data[iter];
            sum += libdivide_s32_do_alg2(numer, &denom);
        }
    }
    else if (algo == 3) {
        for (iter = 0; iter < ITERATIONS; iter++) {
            int32_t numer = data[iter];
            sum += libdivide_s32_do_alg3(numer, &denom);
        }
    }
    else if (algo == 4) {
        for (iter = 0; iter < ITERATIONS; iter++) {
            int32_t numer = data[iter];
            sum += libdivide_s32_do_alg4(numer, &denom);
        }
    }

    return (uint64_t)sum;
}

NOINLINE static uint64_t his_s32(struct FunctionParams_t *params) {
    unsigned iter;
    int32_t sum = 0;
    const int32_t d = *(int32_t *)params->d;
    const int32_t *data = (const int32_t *)params->data;
    for (iter = 0; iter < ITERATIONS; iter++) {
        int32_t numer = data[iter];
        sum += numer / d;
    }
    return sum;
}

NOINLINE static uint64_t mine_s32_generate(struct FunctionParams_t *params) {
    unsigned iter;
    int32_t *dPtr = (int32_t *)params->d;
    struct libdivide_s32_t *denomPtr = (struct libdivide_s32_t *)params->denomPtr;
    for (iter = 0; iter < GEN_ITERATIONS; iter++) {
        *denomPtr = libdivide_s32_gen(*dPtr);
    }
    return *dPtr;
}


//U64

NOINLINE static uint64_t mine_u64(struct FunctionParams_t *params) {
    unsigned iter;
    const struct libdivide_u64_t denom = *(struct libdivide_u64_t *)params->denomPtr;
    const uint64_t *data = (const uint64_t *)params->data;
    uint64_t sum = 0;
    for (iter = 0; iter < ITERATIONS; iter++) {
        uint64_t numer = data[iter];
        sum += libdivide_u64_do(numer, &denom);
    }
    return sum;
}

NOINLINE static uint64_t mine_u64_unswitched(struct FunctionParams_t *params) {
    unsigned iter;
    uint64_t sum = 0;
    const struct libdivide_u64_t denom = *(struct libdivide_u64_t *)params->denomPtr;
    const uint64_t *data = (const uint64_t *)params->data;
    int algo = libdivide_u64_get_algorithm(&denom);
    if (algo == 0) {
        for (iter = 0; iter < ITERATIONS; iter++) {
            uint64_t numer = data[iter];
            sum += libdivide_u64_do_alg0(numer, &denom);
        }
    }
    else if (algo == 1) {
        for (iter = 0; iter < ITERATIONS; iter++) {
            uint64_t numer = data[iter];
            sum += libdivide_u64_do_alg1(numer, &denom);
        }
    }
    else if (algo == 2) {
        for (iter = 0; iter < ITERATIONS; iter++) {
            uint64_t numer = data[iter];
            sum += libdivide_u64_do_alg2(numer, &denom);
        }
    }

    return sum;
}

#if LIBDIVIDE_USE_SSE2
NOINLINE static uint64_t mine_2u64_vector(struct FunctionParams_t *params) {
    unsigned iter;
    __m128i sumX = _mm_setzero_si128();
    const struct libdivide_u64_t denom = *(struct libdivide_u64_t *)params->denomPtr;
    const uint64_t *data = (const uint64_t *)params->data;
    for (iter = 0; iter < ITERATIONS; iter+=2) {
        __m128i numers = *((const __m128i*)(data + iter));
        sumX = _mm_add_epi64(sumX, libdivide_u64_do_vector(numers, &denom));
    }
    const uint64_t *comps = (const uint64_t *)&sumX;
    return comps[0] + comps[1];
}

NOINLINE static uint64_t mine_2u64_vector_unswitched(struct FunctionParams_t *params) {
    unsigned iter;
    __m128i sumX = _mm_setzero_si128();
    const struct libdivide_u64_t denom = *(struct libdivide_u64_t *)params->denomPtr;
    const uint64_t *data = (const uint64_t *)params->data;
    int algo = libdivide_u64_get_algorithm(&denom);
    if (algo == 0) {
        for (iter = 0; iter < ITERATIONS; iter+=2) {
            __m128i numers = *((const __m128i*)(data + iter));
            sumX = _mm_add_epi64(sumX, libdivide_u64_do_vector_alg0(numers, &denom));
        }
    }
    else if (algo == 1) {
        for (iter = 0; iter < ITERATIONS; iter+=2) {
            __m128i numers = *((const __m128i*)(data + iter));
            sumX = _mm_add_epi64(sumX, libdivide_u64_do_vector_alg1(numers, &denom));
        }
    }
    else if (algo == 2) {
        for (iter = 0; iter < ITERATIONS; iter+=2) {
            __m128i numers = *((const __m128i*)(data + iter));
            sumX = _mm_add_epi64(sumX, libdivide_u64_do_vector_alg2(numers, &denom));
        }
    }
    const uint64_t *comps = (const uint64_t *)&sumX;
    return comps[0] + comps[1];
}
#elif LIBDIVIDE_USE_NEON
NOINLINE static uint64_t mine_1u64_vector(struct FunctionParams_t *params) {
    unsigned iter;
    uint64x1_t sumX = vdup_n_u64(0);
    const struct libdivide_u64_t denom = *(struct libdivide_u64_t *)params->denomPtr;
    const uint64_t *data = (const uint64_t *)params->data;
    for (iter = 0; iter < ITERATIONS; iter+=1) {
        uint64x1_t numers = *((const uint64x1_t*)(data + iter));
        sumX = vadd_u64(sumX, libdivide_1u64_do_vector(numers, &denom));
    }
    const uint64_t *comps = (const uint64_t *)&sumX;
    return comps[0];
}
NOINLINE static uint64_t mine_2u64_vector(struct FunctionParams_t *params) {
    unsigned iter;
    uint64x2_t sumX = vdupq_n_u64(0);
    const struct libdivide_u64_t denom = *(struct libdivide_u64_t *)params->denomPtr;
    const uint64_t *data = (const uint64_t *)params->data;
    for (iter = 0; iter < ITERATIONS; iter+=2) {
        uint64x2_t numers = *((const uint64x2_t*)(data + iter));
        sumX = vaddq_u64(sumX, libdivide_2u64_do_vector(numers, &denom));
    }
    const uint64_t *comps = (const uint64_t *)&sumX;
    return comps[0] + comps[1];
}
NOINLINE static uint64_t mine_4u64_vector(struct FunctionParams_t *params) {
    unsigned iter;
    uint64x2_t sumX = vdupq_n_u64(0);
    const struct libdivide_u64_t denom = *(struct libdivide_u64_t *)params->denomPtr;
    const uint64_t *data = (const uint64_t *)params->data;
    for (iter = 0; iter < ITERATIONS; iter+=4) {
        uint64x2x2_t numers = *((const uint64x2x2_t*)(data + iter));
        uint64x2x2_t result = libdivide_u64_do_vector(numers.val[0], &denom);
        sumX = vaddq_u64(sumX, result.val[0]);
        sumX = vaddq_u64(sumX, result.val[1]);
    }
    const uint64_t *comps = (const uint64_t *)&sumX;
    return comps[0] + comps[1];
}

NOINLINE static uint64_t mine_2u64_vector_unswitched(struct FunctionParams_t *params) {
    unsigned iter;
    uint64x2_t sumX = vdupq_n_u64(0);
    const struct libdivide_u64_t denom = *(struct libdivide_u64_t *)params->denomPtr;
    const uint64_t *data = (const uint64_t *)params->data;
    int algo = libdivide_u64_get_algorithm(&denom);
    if (algo == 0) {
        for (iter = 0; iter < ITERATIONS; iter+=2) {
            uint64x2_t numers = *((const uint64x2_t*)(data + iter));
            sumX = vaddq_u64(sumX, libdivide_2u64_do_vector_alg0(numers, &denom));
        }
    }
    else if (algo == 1) {
        for (iter = 0; iter < ITERATIONS; iter+=2) {
            uint64x2_t numers = *((const uint64x2_t*)(data + iter));
            sumX = vaddq_u64(sumX, libdivide_2u64_do_vector_alg1(numers, &denom));
        }
    }
    else if (algo == 2) {
        for (iter = 0; iter < ITERATIONS; iter+=2) {
            uint64x2_t numers = *((const uint64x2_t*)(data + iter));
            sumX = vaddq_u64(sumX, libdivide_2u64_do_vector_alg2(numers, &denom));
        }
    }
    const uint64_t *comps = (const uint64_t *)&sumX;
    return comps[0] + comps[1];
}
#endif

NOINLINE static uint64_t his_u64(struct FunctionParams_t *params) {
    unsigned iter;
    uint64_t sum = 0;
    const uint64_t d = *(uint64_t *)params->d;
    const uint64_t *data = (const uint64_t *)params->data;
    for (iter = 0; iter < ITERATIONS; iter++) {
        uint64_t numer = data[iter];
        sum += numer / d;
    }
    return sum;
}

NOINLINE static uint64_t mine_u64_generate(struct FunctionParams_t *params) {
    unsigned iter;
    uint64_t *dPtr = (uint64_t *)params->d;
    struct libdivide_u64_t *denomPtr = (struct libdivide_u64_t *)params->denomPtr;
    for (iter = 0; iter < GEN_ITERATIONS; iter++) {
        *denomPtr = libdivide_u64_gen(*dPtr);
    }
    return *dPtr;
}

NOINLINE static uint64_t mine_s64(struct FunctionParams_t *params) {
    unsigned iter;
    const struct libdivide_s64_t denom = *(struct libdivide_s64_t *)params->denomPtr;
    const int64_t *data = (const int64_t *)params->data;
    int64_t sum = 0;
    for (iter = 0; iter < ITERATIONS; iter++) {
        int64_t numer = data[iter];
        sum += libdivide_s64_do(numer, &denom);
    }
    return sum;
}

#if LIBDIVIDE_USE_SSE2
NOINLINE static uint64_t mine_2s64_vector(struct FunctionParams_t *params) {
    const struct libdivide_s64_t denom = *(struct libdivide_s64_t *)params->denomPtr;
    const int64_t *data = (const int64_t *)params->data;

    unsigned iter;
    __m128i sumX = _mm_setzero_si128();
    for (iter = 0; iter < ITERATIONS; iter+=2) {
        __m128i numers = *((const __m128i*)(data + iter));
        sumX = _mm_add_epi64(sumX, libdivide_s64_do_vector(numers, &denom));
    }
    const int64_t *comps = (const int64_t *)&sumX;
    int64_t sum = comps[0] + comps[1];
    return sum;
}

NOINLINE static uint64_t mine_2s64_vector_unswitched(struct FunctionParams_t *params) {
    const struct libdivide_s64_t denom = *(struct libdivide_s64_t *)params->denomPtr;
    const int64_t *data = (const int64_t *)params->data;

    unsigned iter;
    __m128i sumX = _mm_setzero_si128();
    int algo = libdivide_s64_get_algorithm(&denom);
    if (algo == 0) {
        for (iter = 0; iter < ITERATIONS; iter+=2) {
            __m128i numers = *((const __m128i*)(data + iter));
            sumX = _mm_add_epi64(sumX, libdivide_s64_do_vector_alg0(numers, &denom));
        }
    }
    else if (algo == 1) {
        for (iter = 0; iter < ITERATIONS; iter+=2) {
            __m128i numers = *((const __m128i*)(data + iter));
            sumX = _mm_add_epi64(sumX, libdivide_s64_do_vector_alg1(numers, &denom));
        }
    }
    else if (algo == 2) {
        for (iter = 0; iter < ITERATIONS; iter+=2) {
            __m128i numers = *((const __m128i*)(data + iter));
            sumX = _mm_add_epi64(sumX, libdivide_s64_do_vector_alg2(numers, &denom));
        }
    }
    else if (algo == 3) {
        for (iter = 0; iter < ITERATIONS; iter+=2) {
            __m128i numers = *((const __m128i*)(data + iter));
            sumX = _mm_add_epi64(sumX, libdivide_s64_do_vector_alg3(numers, &denom));
        }
    }
    else if (algo == 4) {
        for (iter = 0; iter < ITERATIONS; iter+=2) {
            __m128i numers = *((const __m128i*)(data + iter));
            sumX = _mm_add_epi64(sumX, libdivide_s64_do_vector_alg4(numers, &denom));
        }
    }
    const int64_t *comps = (const int64_t *)&sumX;
    int64_t sum = comps[0] + comps[1];
    return sum;
}
#elif LIBDIVIDE_USE_NEON
NOINLINE static uint64_t mine_1s64_vector(struct FunctionParams_t *params) {
    const struct libdivide_s64_t denom = *(struct libdivide_s64_t *)params->denomPtr;
    const int64_t *data = (const int64_t *)params->data;

    unsigned iter;
    int64x1_t sumX = vdup_n_s64(0);
    for (iter = 0; iter < ITERATIONS; iter+=1) {
        int64x1_t numers = *((const int64x1_t*)(data + iter));
        sumX = vadd_s64(sumX, libdivide_1s64_do_vector(numers, &denom));
    }
    const int64_t *comps = (const int64_t *)&sumX;
    int64_t sum = comps[0];
    return sum;
}
NOINLINE static uint64_t mine_2s64_vector(struct FunctionParams_t *params) {
    const struct libdivide_s64_t denom = *(struct libdivide_s64_t *)params->denomPtr;
    const int64_t *data = (const int64_t *)params->data;

    unsigned iter;
    int64x2_t sumX = vdupq_n_s64(0);
    for (iter = 0; iter < ITERATIONS; iter+=2) {
        int64x2_t numers = *((const int64x2_t*)(data + iter));
        sumX = vaddq_s64(sumX, libdivide_2s64_do_vector(numers, &denom));
    }
    const int64_t *comps = (const int64_t *)&sumX;
    int64_t sum = comps[0] + comps[1];
    return sum;
}
NOINLINE static uint64_t mine_4s64_vector(struct FunctionParams_t *params) {
    const struct libdivide_s64_t denom = *(struct libdivide_s64_t *)params->denomPtr;
    const int64_t *data = (const int64_t *)params->data;

    unsigned iter;
    int64x2_t sumX = vdupq_n_s64(0);
    for (iter = 0; iter < ITERATIONS; iter+=4) {
        int64x2x2_t numers = *((const int64x2x2_t*)(data + iter));
        int64x2x2_t result = libdivide_4s64_do_vector(numers, &denom);
        sumX = vaddq_s64(sumX, result.val[0]);
        sumX = vaddq_s64(sumX, result.val[1]);
    }
    const int64_t *comps = (const int64_t *)&sumX;
    int64_t sum = comps[0] + comps[1];
    return sum;
}

NOINLINE static uint64_t mine_2s64_vector_unswitched(struct FunctionParams_t *params) {
    const struct libdivide_s64_t denom = *(struct libdivide_s64_t *)params->denomPtr;
    const int64_t *data = (const int64_t *)params->data;

    unsigned iter;
    int64x2_t sumX = vdupq_n_s64(0);
    int algo = libdivide_s64_get_algorithm(&denom);
    if (algo == 0) {
        for (iter = 0; iter < ITERATIONS; iter+=2) {
            int64x2_t numers = *((const int64x2_t*)(data + iter));
            sumX = vaddq_s64(sumX, libdivide_2s64_do_vector_alg0(numers, &denom));
        }
    }
    else if (algo == 1) {
        for (iter = 0; iter < ITERATIONS; iter+=2) {
            int64x2_t numers = *((const int64x2_t*)(data + iter));
            sumX = vaddq_s64(sumX, libdivide_2s64_do_vector_alg1(numers, &denom));
        }
    }
    else if (algo == 2) {
        for (iter = 0; iter < ITERATIONS; iter+=2) {
            int64x2_t numers = *((const int64x2_t*)(data + iter));
            sumX = vaddq_s64(sumX, libdivide_2s64_do_vector_alg2(numers, &denom));
        }
    }
    else if (algo == 3) {
        for (iter = 0; iter < ITERATIONS; iter+=2) {
            int64x2_t numers = *((const int64x2_t*)(data + iter));
            sumX = vaddq_s64(sumX, libdivide_2s64_do_vector_alg3(numers, &denom));
        }
    }
    else if (algo == 4) {
        for (iter = 0; iter < ITERATIONS; iter+=2) {
            int64x2_t numers = *((const int64x2_t*)(data + iter));
            sumX = vaddq_s64(sumX, libdivide_2s64_do_vector_alg4(numers, &denom));
        }
    }
    const int64_t *comps = (const int64_t *)&sumX;
    int64_t sum = comps[0] + comps[1];
    return sum;
}
#endif

NOINLINE static uint64_t mine_s64_unswitched(struct FunctionParams_t *params) {
    const struct libdivide_s64_t denom = *(struct libdivide_s64_t *)params->denomPtr;
    const int64_t *data = (const int64_t *)params->data;

    unsigned iter;
    int64_t sum = 0;
    int algo = libdivide_s64_get_algorithm(&denom);
    if (algo == 0) {
        for (iter = 0; iter < ITERATIONS; iter++) {
            int64_t numer = data[iter];
            sum += libdivide_s64_do_alg0(numer, &denom);
        }
    }
    else if (algo == 1) {
        for (iter = 0; iter < ITERATIONS; iter++) {
            int64_t numer = data[iter];
            sum += libdivide_s64_do_alg1(numer, &denom);
        }
    }
    else if (algo == 2) {
        for (iter = 0; iter < ITERATIONS; iter++) {
            int64_t numer = data[iter];
            sum += libdivide_s64_do_alg2(numer, &denom);
        }
    }
    else if (algo == 3) {
        for (iter = 0; iter < ITERATIONS; iter++) {
            int64_t numer = data[iter];
            sum += libdivide_s64_do_alg3(numer, &denom);
        }
    }
    else if (algo == 4) {
        for (iter = 0; iter < ITERATIONS; iter++) {
            int64_t numer = data[iter];
            sum += libdivide_s64_do_alg4(numer, &denom);
        }
    }

    return sum;
}

NOINLINE static uint64_t his_s64(struct FunctionParams_t *params) {
    const int64_t *data = (const int64_t *)params->data;
    const int64_t d = *(int64_t *)params->d;

    unsigned iter;
    int64_t sum = 0;
    for (iter = 0; iter < ITERATIONS; iter++) {
        int64_t numer = data[iter];
        sum += numer / d;
    }
    return sum;
}

NOINLINE static uint64_t mine_s64_generate(struct FunctionParams_t *params) {
    int64_t *dPtr = (int64_t *)params->d;
    struct libdivide_s64_t *denomPtr = (struct libdivide_s64_t *)params->denomPtr;
    unsigned iter;
    for (iter = 0; iter < GEN_ITERATIONS; iter++) {
        *denomPtr = libdivide_s64_gen(*dPtr);
    }
    return *dPtr;
}

enum Tests {
    kBaseTest,
    kUnswitchedBaseTest,
    kVec64Test,
    kVec128Test,
    kVec256Test,
    kUnswitchedVec64Test,
    kUnswitchedVec128Test,
    kUnswitchedVec256Test,
    kNumTests
};

const char *strTests[kNumTests] = {
    "scalar",
    "scl_us",
    "v64",
    "v128",
    "v256",
    "v64us"
    "v128us"
    "v256us"
};

struct TestResult {
    double times[kNumTests];
    double gen_time;
    double his_time;
    int algo;
};

static uint64_t find_min(const uint64_t *vals, size_t cnt) {
    uint64_t result = vals[0];
    size_t i;
    for (i=1; i < cnt; i++) {
        if (vals[i] < result) result = vals[i];
    }
    return result;
}

typedef uint64_t (*TestFunc_t)(struct FunctionParams_t *params);

struct TestFuncs {
    TestFunc_t funcs[kNumTests];
    TestFunc_t generate;
    TestFunc_t his;
};

NOINLINE struct TestResult test_one(struct TestFuncs *funcs, struct FunctionParams_t *params) {
#define TEST_COUNT 3
    struct TestResult result;
    memset(&result, 0, sizeof result);

#define CHECK(actual, expected) do { if (1 && actual != expected) printf("Failure on line %lu\n", (unsigned long)__LINE__); } while (0)

    uint64_t my_times[kNumTests][TEST_COUNT], his_times[TEST_COUNT], gen_times[TEST_COUNT];
    unsigned iter, test;
    struct time_result tresult;
    for (iter = 0; iter < TEST_COUNT; iter++) {
        tresult = time_function(funcs->his, params); his_times[iter] = tresult.time; const uint64_t expected = tresult.result;
        for (test = 0; test < kNumTests; test++) {
            if (funcs->funcs[test]) {
                tresult = time_function(funcs->funcs[test], params); my_times[test][iter] = tresult.time; CHECK(tresult.result, expected);
            } else {
                my_times[test][iter] = 0;
            }
        }
        tresult = time_function(funcs->generate, params); gen_times[iter] = tresult.time;
    }

    result.his_time = find_min(his_times, TEST_COUNT) / (double)ITERATIONS;
    result.gen_time = find_min(gen_times, TEST_COUNT) / (double)GEN_ITERATIONS;

    for (test = 0; test < kNumTests; test++) {
        result.times[test] = find_min(my_times[test], TEST_COUNT) / (double)ITERATIONS;
    }
    return result;
#undef TEST_COUNT
}

NOINLINE struct TestResult test_one_u32(uint32_t d, const uint32_t *data) {
    struct libdivide_u32_t div_struct = libdivide_u32_gen(d);
    struct FunctionParams_t params;
    params.d = &d;
    params.denomPtr = &div_struct;
    params.data = data;

    struct TestFuncs funcs;
    funcs.funcs[kBaseTest] = mine_u32;
    funcs.funcs[kVec64Test] = FUNC_VECTOR64(mine_2u32_vector);
    funcs.funcs[kVec128Test] = FUNC_VECTOR128(mine_4u32_vector);
    funcs.funcs[kVec256Test] = FUNC_VECTOR256(mine_8u32_vector);
    funcs.funcs[kUnswitchedBaseTest] = mine_u32_unswitched;
    funcs.funcs[kUnswitchedVec64Test] = NULL;
    funcs.funcs[kUnswitchedVec128Test] = FUNC_VECTOR128(mine_4u32_vector_unswitched);
    funcs.funcs[kUnswitchedVec256Test] = NULL;
    funcs.his = his_u32;
    funcs.generate = mine_u32_generate;

    struct TestResult result = test_one(&funcs, &params);
    result.algo = libdivide_u32_get_algorithm(&div_struct);
    return result;
}

NOINLINE struct TestResult test_one_s32(int32_t d, const int32_t *data) {
    struct libdivide_s32_t div_struct = libdivide_s32_gen(d);
    struct FunctionParams_t params;
    params.d = &d;
    params.denomPtr = &div_struct;
    params.data = data;

    struct TestFuncs funcs;
    funcs.funcs[kBaseTest] = mine_s32;
    funcs.funcs[kVec64Test] = FUNC_VECTOR64(mine_2s32_vector);
    funcs.funcs[kVec128Test] = FUNC_VECTOR128(mine_4s32_vector);
    funcs.funcs[kVec256Test] = FUNC_VECTOR256(mine_8s32_vector);
    funcs.funcs[kUnswitchedBaseTest] = mine_s32_unswitched;
    funcs.funcs[kUnswitchedVec64Test] = NULL;
    funcs.funcs[kUnswitchedVec128Test] = FUNC_VECTOR128(mine_4s32_vector_unswitched);
    funcs.funcs[kUnswitchedVec256Test] = NULL;
    funcs.his = his_s32;
    funcs.generate = mine_s32_generate;

    struct TestResult result = test_one(&funcs, &params);
    result.algo = libdivide_s32_get_algorithm(&div_struct);
    return result;
}

NOINLINE struct TestResult test_one_u64(uint64_t d, const uint64_t *data) {
    struct libdivide_u64_t div_struct = libdivide_u64_gen(d);
    struct FunctionParams_t params;
    params.d = &d;
    params.denomPtr = &div_struct;
    params.data = data;

    struct TestFuncs funcs;
    funcs.funcs[kBaseTest] = mine_u64;
    funcs.funcs[kVec64Test] = FUNC_VECTOR64(mine_1u64_vector);
    funcs.funcs[kVec128Test] = FUNC_VECTOR128(mine_2u64_vector);
    funcs.funcs[kVec256Test] = FUNC_VECTOR256(mine_4u64_vector);
    funcs.funcs[kUnswitchedBaseTest] = mine_u64_unswitched;
    funcs.funcs[kUnswitchedVec64Test] = NULL;
    funcs.funcs[kUnswitchedVec128Test] = FUNC_VECTOR128(mine_2u64_vector_unswitched);
    funcs.funcs[kUnswitchedVec256Test] = NULL;
    funcs.his = his_u64;
    funcs.generate = mine_u64_generate;

    struct TestResult result = test_one(&funcs, &params);
    result.algo = libdivide_u64_get_algorithm(&div_struct);
    return result;
}

NOINLINE struct TestResult test_one_s64(int64_t d, const int64_t *data) {
    struct libdivide_s64_t div_struct = libdivide_s64_gen(d);
    struct FunctionParams_t params;
    params.d = &d;
    params.denomPtr = &div_struct;
    params.data = data;

    struct TestFuncs funcs;
    funcs.funcs[kBaseTest] = mine_s64;
    funcs.funcs[kVec64Test] = FUNC_VECTOR64(mine_1s64_vector);
    funcs.funcs[kVec128Test] = FUNC_VECTOR128(mine_2s64_vector);
    funcs.funcs[kVec256Test] = FUNC_VECTOR256(mine_4s64_vector);
    funcs.funcs[kUnswitchedBaseTest] = mine_s64_unswitched;
    funcs.funcs[kUnswitchedVec64Test] = NULL;
    funcs.funcs[kUnswitchedVec128Test] = FUNC_VECTOR128(mine_2s64_vector_unswitched);
    funcs.funcs[kUnswitchedVec256Test] = NULL;
    funcs.his = his_s64;
    funcs.generate = mine_s64_generate;

    struct TestResult result = test_one(&funcs, &params);
    result.algo = libdivide_s64_get_algorithm(&div_struct);
    return result;
}

static void report_header(void) {
    unsigned test;
    printf("%6s%8s", "#", "system");
    for (test = 0; test < kNumTests; test++) {
        printf("%8s", strTests[test]);
    }
    printf("%8s%6s\n", "gener", "algo");
}

static void report_result(const char *input, struct TestResult result) {
    unsigned test;
    printf("%6s%8.3f", input, result.his_time);
    for (test = 0; test < kNumTests; test++) {
        printf("%8.3f", result.times[test]);
    }
    printf("%8.3f%6d\n", result.gen_time, result.algo);
}

static void test_many_u32(const uint32_t *data) {
    report_header();
    uint32_t d;
    for (d=1; d > 0; d++) {
        struct TestResult result = test_one_u32(d, data);
        char input_buff[32];
        sprintf(input_buff, "%u", d);
        report_result(input_buff, result);
    }
}

static void test_many_s32(const int32_t *data) {
    report_header();
    int32_t d;
    for (d=1; d != 0;) {
        struct TestResult result = test_one_s32(d, data);
        char input_buff[32];
        sprintf(input_buff, "%d", d);
        report_result(input_buff, result);

        d = -d;
        if (d > 0) d++;
    }
}

static void test_many_u64(const uint64_t *data) {
    report_header();
    uint64_t d;
    for (d=1; d > 0; d++) {
        struct TestResult result = test_one_u64(d, data);
        char input_buff[32];
        sprintf(input_buff, "%llu", d);
        report_result(input_buff, result);
    }
}

static void test_many_s64(const int64_t *data) {
    report_header();
    int64_t d;
    for (d=1; d != 0;) {
        struct TestResult result = test_one_s64(d, data);
        char input_buff[32];
        sprintf(input_buff, "%lld", d);
        report_result(input_buff, result);

        d = -d;
        if (d > 0) d++;
    }
}

static const uint32_t *random_data(unsigned multiple) {
#if LIBDIVIDE_WINDOWS
    uint32_t *data = (uint32_t *)malloc(multiple * ITERATIONS * sizeof *data);
#else
    /* Linux doesn't always give us data sufficiently aligned for SSE, so we can't use malloc(). */
    void *ptr = NULL;
    posix_memalign(&ptr, 16, multiple * ITERATIONS * sizeof(uint32_t));
    uint32_t *data = (uint32_t *)ptr;
#endif
    uint32_t i;
    struct random_state state = SEED;
    for (i=0; i < ITERATIONS * multiple; i++) {
        data[i] = my_random(&state);
    }
    return data;
}

int main(int argc, char* argv[]) {
#if LIBDIVIDE_WINDOWS
    QueryPerformanceFrequency(&gPerfCounterFreq);
#endif
    int i, u32 = 0, u64 = 0, s32 = 0, s64 = 0;
    if (argc == 1) {
        /* Test all */
        u32 = u64 = s32 = s64 = 1;
    }
    else {
        for (i=1; i < argc; i++) {
            if (! strcmp(argv[i], "u32")) u32 = 1;
            else if (! strcmp(argv[i], "u64")) u64 = 1;
            else if (! strcmp(argv[i], "s32")) s32 = 1;
            else if (! strcmp(argv[i], "s64")) s64 = 1;
            else printf("Unknown test '%s'\n", argv[i]), exit(0);
        }
    }
    const uint32_t *data = NULL;
    data = random_data(1);
    if (u32) test_many_u32(data);
    if (s32) test_many_s32((const int32_t *)data);
    free((void *)data);

    data = random_data(2);
    if (u64) test_many_u64((const uint64_t *)data);
    if (s64) test_many_s64((const int64_t *)data);
    free((void *)data);
    return 0;
}
