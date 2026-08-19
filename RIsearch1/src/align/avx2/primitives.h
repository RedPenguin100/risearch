#pragma once

#include <cstdint>

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

/* Declared here and specialised for each element type below, so that a kernel
   written against them runs at whichever width its scores allow. Only the ones
   whose instruction or shape differs are spelled out per type. */
template<typename int_type>
__attribute__((target("avx2"), always_inline)) inline __m256i v_int_to_avx2(int_type v);
template<typename int_type>
__attribute__((target("avx2"), always_inline)) inline __m256i v_add(__m256i a, __m256i b);
template<typename int_type>
__attribute__((target("avx2"), always_inline)) inline __m256i v_sub(__m256i a, __m256i b);
template<typename int_type>
__attribute__((target("avx2"), always_inline)) inline __m256i v_max(__m256i a, __m256i b);
template<typename int_type>
__attribute__((target("avx2"), always_inline)) inline __m256i v_add_unless_zero_or_neg1(__m256i base, __m256i term);
template<typename int_type>
__attribute__((target("avx2"), always_inline)) inline int_type v_hmax(__m256i v);
template<typename int_type>
__attribute__((target("avx2"), always_inline)) inline __m256i v_prefix_max(__m256i v);
template<typename int_type>
__attribute__((target("avx2"), always_inline)) inline __m256i v_shifted_left_one(__m256i prev, __m256i cur);
template<typename int_type>
__attribute__((target("avx2"), always_inline)) inline __m256i v_broadcast_last(__m256i v);

template<typename int_type>
constexpr unsigned v_lanes()
{
    return sizeof(__m256i) / sizeof(int_type);
}

/* One value in every lane. */
template<>
__attribute__((target("avx2"), always_inline)) inline __m256i v_int_to_avx2<std::int32_t>(int v)
{
    return _mm256_set1_epi32(v);
}

/* A register's worth of a row or of a term run. */
template<typename int_type>
__attribute__((target("avx2"), always_inline)) static inline __m256i v_vec_load(const int_type* p)
{
    return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p));
}

template<typename int_type>
__attribute__((target("avx2"), always_inline)) static inline void v_vec_store(int_type* p, __m256i v)
{
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(p), v);
}

/* The scalar operations of the recurrence, eight lanes at a time. vmax4 mirrors
   max4, and add_unless_zero_or_neg1 is the `x != 0 ? x + term : -1` test -- no
   lane can branch, so both arms are computed and one is selected per lane. */
/* int32 stays inside its range for any score the bound allows, so a plain add
   serves. int16 saturates instead, which is what keeps a sentinel a sentinel. */
template<>
__attribute__((target("avx2"), always_inline)) inline __m256i v_add<std::int32_t>(__m256i a, __m256i b)
{
    return _mm256_add_epi32(a, b);
}

template<>
__attribute__((target("avx2"), always_inline)) inline __m256i v_sub<std::int32_t>(__m256i a, __m256i b)
{
    return _mm256_sub_epi32(a, b);
}

template<>
__attribute__((target("avx2"), always_inline)) inline __m256i v_max<std::int32_t>(__m256i a, __m256i b)
{
    return _mm256_max_epi32(a, b);
}

template<typename int_type>
__attribute__((target("avx2"), always_inline)) static inline __m256i v_max3(__m256i a, __m256i b,
                                                                            __m256i c)
{
    return v_max<int_type>(v_max<int_type>(a, b), c);
}

template<typename int_type>
__attribute__((target("avx2"), always_inline)) static inline __m256i v_max4(__m256i a, __m256i b,
                                                                            __m256i c, __m256i d)
{
    return v_max<int_type>(v_max<int_type>(a, b), v_max<int_type>(c, d));
}


template<>
__attribute__((target("avx2"), always_inline)) inline __m256i
v_add_unless_zero_or_neg1<std::int32_t>(__m256i base, __m256i term)
{
    // base != 0 ? base + term : -1; (per lane)
    return _mm256_or_si256(v_add<std::int32_t>(base, term),
                           _mm256_cmpeq_epi32(base, _mm256_setzero_si256()));
}

/* Largest of the eight lanes, by folding in half three times: eight values cost
   three maxes rather than seven. Lane 0 ends up holding the answer. */
template<>
__attribute__((target("avx2"), always_inline)) inline int v_hmax<std::int32_t>(__m256i v)
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
template<>
__attribute__((target("avx2"), always_inline)) inline __m256i v_prefix_max<std::int32_t>(__m256i v)
{
    const __m256i none = v_int_to_avx2<std::int32_t>(INT_MIN);
    const __m256i down1 = _mm256_setr_epi32(0, 0, 1, 2, 3, 4, 5, 6);
    const __m256i down2 = _mm256_setr_epi32(0, 0, 0, 1, 2, 3, 4, 5);
    const __m256i down4 = _mm256_setr_epi32(0, 0, 0, 0, 0, 1, 2, 3);

    v = v_max<std::int32_t>(v, _mm256_blend_epi32(_mm256_permutevar8x32_epi32(v, down1), none, 0x01));
    v = v_max<std::int32_t>(v, _mm256_blend_epi32(_mm256_permutevar8x32_epi32(v, down2), none, 0x03));
    return v_max<std::int32_t>(v, _mm256_blend_epi32(_mm256_permutevar8x32_epi32(v, down4), none, 0x0f));
}

/* The block's eight columns of M moved one column to the right, so that lane k
   holds M at the column just left of the block's lane k. Lane 7 of prev is the
   value that moves in. A four-byte shift of a whole register has to cross its
   two 128-bit halves, which alignr does not do on its own, so the permute puts
   the two halves alignr needs side by side first. */
template<>
__attribute__((target("avx2"), always_inline)) inline __m256i
v_shifted_left_one<std::int32_t>(__m256i prev, __m256i cur)
{
    return _mm256_alignr_epi8(cur, _mm256_permute2x128_si256(prev, cur, 0x21), 12);
}

/* The top lane broadcast, which is the carry the next block starts from. */
template<>
__attribute__((target("avx2"), always_inline)) inline __m256i v_broadcast_last<std::int32_t>(__m256i v)
{
    return _mm256_permutevar8x32_epi32(v, _mm256_set1_epi32(v_lanes<std::int32_t>() - 1));
}


/* The same operations sixteen lanes at a time. The recurrence is identical; what
   differs is the instruction, and the two shuffles AVX2 has no 16-bit form of. */

template<>
__attribute__((target("avx2"), always_inline)) inline __m256i v_int_to_avx2<std::int16_t>(std::int16_t v)
{
    return _mm256_set1_epi16(v);
}

/* Saturating, not wrapping: a sentinel plus a negative term must stay at the
   bottom of the range rather than wrap round to the top and win everything. On a
   real value the two agree, because int16_bound keeps every value clear of the
   ends. */
template<>
__attribute__((target("avx2"), always_inline)) inline __m256i v_add<std::int16_t>(__m256i a, __m256i b)
{
    return _mm256_adds_epi16(a, b);
}

template<>
__attribute__((target("avx2"), always_inline)) inline __m256i v_sub<std::int16_t>(__m256i a, __m256i b)
{
    return _mm256_subs_epi16(a, b);
}

template<>
__attribute__((target("avx2"), always_inline)) inline __m256i v_max<std::int16_t>(__m256i a, __m256i b)
{
    return _mm256_max_epi16(a, b);
}

template<>
__attribute__((target("avx2"), always_inline)) inline __m256i
v_add_unless_zero_or_neg1<std::int16_t>(__m256i base, __m256i term)
{
    return _mm256_or_si256(v_add<std::int16_t>(base, term),
                           _mm256_cmpeq_epi16(base, _mm256_setzero_si256()));
}

template<>
__attribute__((target("avx2"), always_inline)) inline std::int16_t v_hmax<std::int16_t>(__m256i v)
{
    __m128i best = _mm_max_epi16(_mm256_castsi256_si128(v), _mm256_extracti128_si256(v, 1));
    best = _mm_max_epi16(best, _mm_shuffle_epi32(best, _MM_SHUFFLE(1, 0, 3, 2)));
    best = _mm_max_epi16(best, _mm_shuffle_epi32(best, _MM_SHUFFLE(2, 3, 0, 1)));
    best = _mm_max_epi16(best, _mm_shufflelo_epi16(best, _MM_SHUFFLE(2, 3, 0, 1)));
    return static_cast<std::int16_t>(_mm_extract_epi16(best, 0));
}

/* AVX2 has no cross-lane 16-bit permute, so the climb is byte shifts inside each
   128-bit half, and the low half's running max is carried into the high half by
   hand at the end. */
template<>
__attribute__((target("avx2"), always_inline)) inline __m256i v_prefix_max<std::int16_t>(__m256i v)
{
    const __m256i none = _mm256_set1_epi16(SHRT_MIN);

    v = _mm256_max_epi16(v, _mm256_alignr_epi8(v, none, 14));
    v = _mm256_max_epi16(v, _mm256_alignr_epi8(v, none, 12));
    v = _mm256_max_epi16(v, _mm256_alignr_epi8(v, none, 8));

    /* Each half now holds its own prefix max; the high half has seen nothing of
       the low one, so lane 7's value is broadcast into it and folded in. */
    const __m128i low = _mm256_castsi256_si128(v);
    const __m128i low_last = _mm_shuffle_epi8(
        low, _mm_setr_epi8(14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15));
    /* The low half must be left alone, so what it is maxed against is the
       sentinel rather than zero -- zero would raise every negative prefix. */
    const __m256i carry = _mm256_inserti128_si256(none, low_last, 1);
    return _mm256_max_epi16(v, carry);
}

template<>
__attribute__((target("avx2"), always_inline)) inline __m256i
v_shifted_left_one<std::int16_t>(__m256i prev, __m256i cur)
{
    return _mm256_alignr_epi8(cur, _mm256_permute2x128_si256(prev, cur, 0x21), 14);
}

template<>
__attribute__((target("avx2"), always_inline)) inline __m256i v_broadcast_last<std::int16_t>(__m256i v)
{
    const __m128i high = _mm256_extracti128_si256(v, 1);
    const __m128i last = _mm_shuffle_epi8(
        high, _mm_setr_epi8(14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15));
    return _mm256_broadcastsi128_si256(last);
}

#endif /* RISEARCH1_HAS_AVX2 */
