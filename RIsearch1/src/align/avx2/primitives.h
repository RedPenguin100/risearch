#pragma once

#include <climits>
#include <cstdlib>

/* The AVX2 operations the kernels are written in.
 *
 * Each one is a single scalar operation of the recurrence done eight times, or a
 * movement of lanes the recurrence needs. They are here rather than beside a
 * kernel because both the sweep and the traceback fill are written in them, and
 * because a kernel reads better when the only thing in it is the recurrence.
 *
 * Nothing here knows what a score is.
 */

/* The vector kernels are compiled in only where the compiler understands both
   the x86 intrinsics and the target attribute that keeps them out of the rest of
   the program. Everywhere else the scalar versions are the whole implementation. */
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
#define RISEARCH1_HAS_AVX2 1
#include <immintrin.h>
#else
#define RISEARCH1_HAS_AVX2 0
#endif

/* AVX2 is picked at run time, so one binary still starts on a CPU without it and
   the build needs no -march. RISEARCH_NO_AVX2 in the environment forces the
   scalar version, which is how the two are compared from a single build.
   Decided once at startup: a function-local static would put its thread-safe
   initialisation guard inside the caller, which is the hot loop. */
static const bool CPU_HAS_AVX2 = [] {
#if RISEARCH1_HAS_AVX2
    if (getenv("RISEARCH_NO_AVX2") != nullptr) {
        return false;
    }
    __builtin_cpu_init();
    return __builtin_cpu_supports("avx2") != 0;
#else
    return false;
#endif
}();


#if RISEARCH1_HAS_AVX2

/* One value in all eight lanes. */
__attribute__((target("avx2"), always_inline)) static inline __m256i v_int_to_avx2(int v)
{
    return _mm256_set1_epi32(v);
}

/* Eight ints of a row or of a term run. */
__attribute__((target("avx2"), always_inline)) static inline __m256i v_vec_load(const int* p)
{
    return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p));
}

__attribute__((target("avx2"), always_inline)) static inline void v_vec_store(int* p, __m256i v)
{
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(p), v);
}

/* The scalar operations of the recurrence, eight lanes at a time. vmax4 mirrors
   max4, and add_unless_zero_or_neg1 is the `x != 0 ? x + term : -1` test -- no
   lane can branch, so both arms are computed and one is selected per lane. */
__attribute__((target("avx2"), always_inline)) static inline __m256i v_add(__m256i a, __m256i b)
{
    return _mm256_add_epi32(a, b);
}

__attribute__((target("avx2"), always_inline)) static inline __m256i v_sub(__m256i a, __m256i b)
{
    return _mm256_sub_epi32(a, b);
}

__attribute__((target("avx2"), always_inline)) static inline __m256i v_max(__m256i a, __m256i b)
{
    return _mm256_max_epi32(a, b);
}

__attribute__((target("avx2"), always_inline)) static inline __m256i v_max3(__m256i a, __m256i b,
                                                                            __m256i c)
{
    return v_max(v_max(a, b), c);
}

__attribute__((target("avx2"), always_inline)) static inline __m256i v_max4(__m256i a, __m256i b,
                                                                            __m256i c, __m256i d)
{
    return v_max(v_max(a, b), v_max(c, d));
}


__attribute__((target("avx2"), always_inline)) static inline __m256i
v_add_unless_zero_or_neg1(__m256i base, __m256i term)
{
    // base != 0 ? base + term : -1; (per lane)
    return _mm256_or_si256(v_add(base, term), _mm256_cmpeq_epi32(base, _mm256_setzero_si256()));
}

/* Largest of the eight lanes, by folding in half three times: eight values cost
   three maxes rather than seven. Lane 0 ends up holding the answer. */
__attribute__((target("avx2"), always_inline)) static inline int v_hmax(__m256i v)
{
    __m128i best = _mm_max_epi32(_mm256_castsi256_si128(v), _mm256_extracti128_si256(v, 1));
    best = _mm_max_epi32(best, _mm_shuffle_epi32(best, _MM_SHUFFLE(1, 0, 3, 2)));
    best = _mm_max_epi32(best, _mm_shuffle_epi32(best, _MM_SHUFFLE(2, 3, 0, 1)));
    return _mm_cvtsi128_si32(best);
}

/* Lane k becomes the largest of lanes 0..k, in three doubling steps: each takes
   the value 1, then 2, then 4 lanes below and maxes it in. Three suffice for
   eight lanes because 1 + 2 + 4 = 7 is the furthest any lane must see, and the
   regrouping is legal only because max is associative -- which is what the
   ix_prefix rewrite bought. permutevar crosses the register's two 128-bit
   halves, which a byte shift cannot; index 0 is filler for the low lanes with no
   source, and the blend overwrites exactly those with INT_MIN so filler never
   wins. */
__attribute__((target("avx2"), always_inline)) static inline __m256i v_prefix_max(__m256i v)
{
    const __m256i none = v_int_to_avx2(INT_MIN);
    const __m256i down1 = _mm256_setr_epi32(0, 0, 1, 2, 3, 4, 5, 6);
    const __m256i down2 = _mm256_setr_epi32(0, 0, 0, 1, 2, 3, 4, 5);
    const __m256i down4 = _mm256_setr_epi32(0, 0, 0, 0, 0, 1, 2, 3);

    v = v_max(v, _mm256_blend_epi32(_mm256_permutevar8x32_epi32(v, down1), none, 0x01));
    v = v_max(v, _mm256_blend_epi32(_mm256_permutevar8x32_epi32(v, down2), none, 0x03));
    return v_max(v, _mm256_blend_epi32(_mm256_permutevar8x32_epi32(v, down4), none, 0x0f));
}

/* The block's eight columns of M moved one column to the right, so that lane k
   holds M at the column just left of the block's lane k. Lane 7 of prev is the
   value that moves in. A four-byte shift of a whole register has to cross its
   two 128-bit halves, which alignr does not do on its own, so the permute puts
   the two halves alignr needs side by side first. */
__attribute__((target("avx2"), always_inline)) static inline __m256i
v_shifted_left_one(__m256i prev, __m256i cur)
{
    return _mm256_alignr_epi8(cur, _mm256_permute2x128_si256(prev, cur, 0x21), 12);
}

#endif /* RISEARCH1_HAS_AVX2 */
