#pragma once

#include <climits>
#include <cstdint>
#include <cstdlib>
#include <utility>

#include "RunningMax.h"
#include "nucleotide.h"
#include "operations.h"
#include "optimization/QueryProfile.h"

/* Scoring the target: the whole j loop over target positions, run once per
 * alignment.
 *
 * For every target position it records the best score of an alignment ending
 * there and the query position it ended at -- hs and hp below -- and keeps the
 * single best across the whole target. It reconstructs nothing; that is RIs.
 *
 * score_target_scalar is the reference and the definition of what the output
 * must be. score_target_avx2 computes the same recurrence eight query positions
 * at a time and has to agree with it bit for bit, so the two are meant to be
 * read side by side.
 */

/* The vector kernel is compiled in only where the compiler understands both the
   x86 intrinsics and the target attribute that keeps them out of the rest of the
   program. Everywhere else the scalar version is the whole implementation. */
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
#define RISEARCH1_HAS_AVX2 1
#include <immintrin.h>
#else
#define RISEARCH1_HAS_AVX2 0
#endif


/* Inlined into the caller on purpose: there the rows and the profile's tables
   are visibly separate allocations, and out of line the compiler has to assume a
   store through one could land in the other and re-load everything per row. */
__attribute__((always_inline)) static inline void
score_target_scalar(const unsigned char* target_sequence, const QueryProfile& profile,
                    int* const* M, int* const* Ix, int* const* Iy, int* hs, int* hp, int n,
                    int threshold, RunningMax& running_max)
{
    const auto m = profile.m();


    for (auto j = 2u; j <= n; j++) {
        /* Begin init of i=1 case */
        const auto currentRow = j % 2;
        const auto lastRow = 1 - currentRow;

        const auto target_current = target_sequence[n - j];
        const auto target_prev = target_sequence[n - j + 1];

        /* DSM caching optimization */
        const auto context = QueryProfile::context(target_prev, target_current);
        const auto T = profile.row(context);
        const auto iy_ext = T.iy_extend;
        /* DSM caching optimization */


        /* Column 1 is the query's first nt, nothing can precede it */
        M[currentRow][1] = MAX(0, T.m_open[1]);

        // Track only the best value, the position is recovered later (OPTIMIZATION)
        auto row_max = M[currentRow][1] + T.close[1];

        // Ix bulges a query nt, impossible in 1st nucleotide
        Ix[currentRow][1] = NEGINF;

        // Iy bulges a target nt, j>=2 so bulge possible.
        // Only M can win row max, so we don't take this as a candidate
        Iy[currentRow][1] = MAX(
            // We are now opening the bulge
            M[lastRow][1] + T.iy_from_m[1],
            // We are extending a bulge
            Iy[lastRow][1] + iy_ext);

        /* finished init of i=1 col */


        for (auto i = 2u; i <= m; i++) {
            // Assign M with a 4-way max
            M[currentRow][i] = max4(
                /* coming from a match */
                M[lastRow][i - 1] != 0 ? M[lastRow][i - 1] + T.m_from_m[i] : -1,
                /* coming from gap in target */
                Ix[lastRow][i - 1] + T.m_from_ix[i],
                /* coming from gap in query */
                Iy[lastRow][i - 1] + T.m_from_iy[i],
                /* start fresh */
                T.m_open[i]);

            // Set max now, position is recovered later
            row_max = MAX(row_max, M[currentRow][i] + T.close[i]);

            // Iy: target nt against a gap
            Iy[currentRow][i] = MAX(
                // pair at previous row, now bulge
                M[lastRow][i] + T.iy_from_m[i],
                // already bulging, add one more
                Iy[lastRow][i] + iy_ext);
        }

        // Split the loop for performance.
        for (auto i = 2u; i <= m; ++i) {
            // Ix: query nt against a gap
            Ix[currentRow][i] = MAX(
                // pair at i - 1, now bulge
                M[currentRow][i - 1] + T.ix_from_m[i],
                // already bulging, add one more
                Ix[currentRow][i - 1] + T.ix_extend[i]);
        }

        // Re-infer the row max's position
        auto row_pos = 1u;
        if (row_max > threshold || row_max > running_max.score) {
            for (auto i = 1u; i <= m; ++i) {
                if (M[currentRow][i] + T.close[i] == row_max) {
                    row_pos = i;
                    break;
                }
            }
        }

        hs[j - 1] = row_max;
        hp[j - 1] = static_cast<int>(row_pos);

        running_max.set_if_better(row_max, static_cast<int>(row_pos), j);

    } /*next row j */
}


#if RISEARCH1_HAS_AVX2

/* One value in all eight lanes. */
__attribute__((target("avx2"), always_inline)) static inline __m256i all_lanes(int v)
{
    return _mm256_set1_epi32(v);
}

/* The scalar operations of the recurrence, eight lanes at a time. vmax4 mirrors
   max4, and add_unless_zero is the `x != 0 ? x + term : -1` test -- no lane can
   branch, so both arms are computed and one is selected per lane. */
__attribute__((target("avx2"), always_inline)) static inline __m256i vadd(__m256i a, __m256i b)
{
    return _mm256_add_epi32(a, b);
}

__attribute__((target("avx2"), always_inline)) static inline __m256i vsub(__m256i a, __m256i b)
{
    return _mm256_sub_epi32(a, b);
}

__attribute__((target("avx2"), always_inline)) static inline __m256i vmax(__m256i a, __m256i b)
{
    return _mm256_max_epi32(a, b);
}

__attribute__((target("avx2"), always_inline)) static inline __m256i vmax3(__m256i a, __m256i b,
                                                                           __m256i c)
{
    return vmax(vmax(a, b), c);
}

__attribute__((target("avx2"), always_inline)) static inline __m256i vmax4(__m256i a, __m256i b,
                                                                           __m256i c, __m256i d)
{
    return vmax(vmax(a, b), vmax(c, d));
}

__attribute__((target("avx2"), always_inline)) static inline __m256i
add_unless_zero(__m256i base, __m256i term, int fallback)
{
    return _mm256_blendv_epi8(vadd(base, term), all_lanes(fallback),
                              _mm256_cmpeq_epi32(base, _mm256_setzero_si256()));
}


/* Eight ints of a row or of a term run. */
__attribute__((target("avx2"), always_inline)) static inline __m256i vec_load(const int* p)
{
    return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p));
}

__attribute__((target("avx2"), always_inline)) static inline void vec_store(int* p, __m256i v)
{
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(p), v);
}

/* Largest of the eight lanes, by folding in half three times: eight values cost
   three maxes rather than seven. Lane 0 ends up holding the answer. */
__attribute__((target("avx2"), always_inline)) static inline int vec_hmax(__m256i v)
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
__attribute__((target("avx2"), always_inline)) static inline __m256i vec_prefix_max(__m256i v)
{
    const __m256i none = all_lanes(INT_MIN);
    const __m256i down1 = _mm256_setr_epi32(0, 0, 1, 2, 3, 4, 5, 6);
    const __m256i down2 = _mm256_setr_epi32(0, 0, 0, 1, 2, 3, 4, 5);
    const __m256i down4 = _mm256_setr_epi32(0, 0, 0, 0, 0, 1, 2, 3);

    v = vmax(v, _mm256_blend_epi32(_mm256_permutevar8x32_epi32(v, down1), none, 0x01));
    v = vmax(v, _mm256_blend_epi32(_mm256_permutevar8x32_epi32(v, down2), none, 0x03));
    return vmax(v, _mm256_blend_epi32(_mm256_permutevar8x32_epi32(v, down4), none, 0x0f));
}

/* M and Iy for the eight columns starting at i, and their contribution to the
   row max. Returns M, which the Ix scan reads. */
__attribute__((target("avx2"), always_inline)) static inline __m256i
main_dp_loop_avx2(unsigned i, const QueryProfile::RowView& T, int* m_cur, int* iy_cur,
                  const int* m_last, const int* ix_last, const int* iy_last, __m256i v_iy_ext,
                  __m256i* v_row_max)
{
    /* For each column this block writes, its diagonal predecessor: one target
       position back and one query position back. Adjacent because the columns
       are adjacent; the diagonal is only the -1. */
    const __m256i m_diag = vec_load(m_last + i - 1);


    // Assign M with a 4-way max
    const __m256i m_new = vmax4(
        /* coming from a match */
        add_unless_zero(m_diag, vec_load(T.m_from_m + i), -1),
        /* coming from gap in target */
        vadd(vec_load(ix_last + i - 1), vec_load(T.m_from_ix + i)),
        /* coming from gap in query */
        vadd(vec_load(iy_last + i - 1), vec_load(T.m_from_iy + i)),
        /* start fresh */
        vec_load(T.m_open + i));
    vec_store(m_cur + i, m_new);

    // Set max now, position is recovered later
    *v_row_max = vmax(*v_row_max, vadd(m_new, vec_load(T.close + i)));

    // Iy: target nt against a gap. Predecessors are vertical -- previous row,
    // same column -- so no -1 on the address, unlike M's diagonal above.
    vec_store(iy_cur + i, vmax(
                              /* pair at previous row, now bulge */
                              vadd(vec_load(m_last + i), vec_load(T.iy_from_m + i)),
                              /* already bulging, add one more */
                              vadd(vec_load(iy_last + i), v_iy_ext)));

    return m_new;
}

/* The block's eight columns of M moved one column to the right, so that lane k
   holds M at the column just left of the block's lane k. Lane 7 of prev is the
   value that moves in. A four-byte shift of a whole register has to cross its
   two 128-bit halves, which alignr does not do on its own, so the permute puts
   the two halves alignr needs side by side first. */
__attribute__((target("avx2"), always_inline)) static inline __m256i shifted_left_one(__m256i prev,
                                                                                      __m256i cur)
{
    return _mm256_alignr_epi8(cur, _mm256_permute2x128_si256(prev, cur, 0x21), 12);
}

/* The Ix scan for a block's eight columns, starting at column i.
 *
 * m_left holds M at the column just left of each of the eight, which is how the
 * loop hands the scan the M it has in a register instead of putting the row
 * back through memory. The value returned is the carry the next block starts
 * from: lane 7 of the running max, which is its maximum over the whole block,
 * broadcast so that the max above needs no shuffling. */
__attribute__((target("avx2"), always_inline)) static inline __m256i
ix_dp_loop_avx2(int* ix_out, const int* ix_from_m_scan, const int* ix_prefix, __m256i m_left,
                __m256i ix_carry)
{
    // Ix: query nt against a gap
    const __m256i candidates = vadd(m_left, vec_load(ix_from_m_scan));
    const __m256i best = vmax(vec_prefix_max(candidates), ix_carry);
    /* Adding ix_prefix back turns the carried quantity into the real Ix, which
       is what the next row and the traceback read. */
    vec_store(ix_out, vadd(best, vec_load(ix_prefix)));
    return _mm256_permutevar8x32_epi32(best, all_lanes(7));
}

/* The same recurrence, eight query positions at a time. Differences from the
 * scalar version above, and nothing else:
 *
 *  - the two rows are swapped pointers instead of j % 2 indices;
 *  - M and Iy read only the previous row, so eight of their columns are
 *    independent and a block computes them outright. The blocks cover columns
 *    2..m, the last backed up to end at m and recomputing the columns it
 *    overlaps -- to the same values, since it reads only the previous row --
 *    which is cheaper than a scalar epilogue. That needs m >= 9, which
 *    score_target_is_vectorized has already checked;
 *  - Ix reads the row it writes, so it is rewritten as a plain running max over
 *    candidates with ix_prefix taken out (see QueryProfile). A running max
 *    cannot revisit a column already folded into its carry, so only whole
 *    blocks go through it and the last few columns finish serially;
 *  - the row max is accumulated in eight lanes and reduced once per row.
 *
 * Column 1 stays scalar throughout: the neighbouring column every block reads
 * does not exist there.
 *
 * THE SCAN TAKES M FROM THE REGISTER THE BLOCK BUILT IT IN, one column to the
 * right, rather than loading the row it has just been stored to. A load that
 * reads back what the same row has just written cannot be served from the store
 * that wrote it, and the scan's candidates are the first thing on the row's
 * serial chain, so that one load sits in front of everything else. Removing it
 * is worth around 30% of this kernel at m = 12 or 16, where a row is one or two
 * blocks wide. At m >= 80 a row is ten blocks and there is enough independent
 * work to cover the load anyway, and the shift that replaces it costs about as
 * much as it saves.
 */
__attribute__((target("avx2"))) static void
score_target_avx2(const unsigned char* target_sequence, const QueryProfile& profile, int* const* M,
                  int* const* Ix, int* const* Iy, int* hs, int* hp, int n, int threshold,
                  RunningMax& running_max)
{
    const auto m = profile.m();

    /* Row j is written to row j % 2, so the first row here, j = 2, writes row 0;
       swapping the two pointers at the end of each step is the same thing and
       keeps the row addresses in registers. */
    int* m_cur = M[0];
    int* m_last = M[1];
    int* ix_cur = Ix[0];
    int* ix_last = Ix[1];
    int* iy_cur = Iy[0];
    int* iy_last = Iy[1];

    for (auto j = 2u; j <= n; j++) {
        const auto target_current = target_sequence[n - j];
        const auto target_prev = target_sequence[n - j + 1];

        const auto context = QueryProfile::context(target_prev, target_current);
        const auto T = profile.row(context);
        const auto iy_ext = T.iy_extend;

        /* Column 1 is the query's first nt, nothing can precede it */
        m_cur[1] = MAX(0, T.m_open[1]);
        auto row_max = m_cur[1] + T.close[1];
        ix_cur[1] = NEGINF;
        iy_cur[1] = MAX(m_last[1] + T.iy_from_m[1], iy_last[1] + iy_ext);
        /* finished init of i=1 col */

        const __m256i v_iy_ext = all_lanes(iy_ext);
        __m256i v_row_max = all_lanes(row_max);

        /* M at the column just left of the scan's first block, and the carry it
           enters with -- what Ix[1] holds, so the first block sees exactly what
           the serial recurrence would have carried into it. */
        __m256i m_left = all_lanes(m_cur[1]);
        __m256i v_ix_carry = all_lanes(NEGINF);

        /* Begin main DP */
        auto start = 2u;
        for (; start + 7 <= m; start += 8) {
            const __m256i m_block = main_dp_loop_avx2(start, T, m_cur, iy_cur, m_last, ix_last,
                                                      iy_last, v_iy_ext, &v_row_max);
            // The separate ix loop in a separate function
            v_ix_carry =
                ix_dp_loop_avx2(ix_cur + start, T.ix_from_m_scan + start, T.ix_prefix + start,
                                shifted_left_one(m_left, m_block), v_ix_carry);
            m_left = m_block;
        }

        // Fewer than 8 columns need special case for main DP
        if (start <= m) {
            main_dp_loop_avx2(m - 7, T, m_cur, iy_cur, m_last, ix_last, iy_last, v_iy_ext,
                              &v_row_max);
        }
        row_max = vec_hmax(v_row_max);

        // Fewer than 8 columns need special case for main IX DP
        // Ix can't re-use the function like main_dp because an overlap behaves differently.
        for (auto i = start; i <= m; ++i) {
            ix_cur[i] = MAX(m_cur[i - 1] + T.ix_from_m[i], ix_cur[i - 1] + T.ix_extend[i]);
        }
        /* End main DP */


        // Re-infer the row max's position
        auto row_pos = 1u;
        if (row_max > threshold || row_max > running_max.score) {
            for (auto i = 1u; i <= m; ++i) {
                if (m_cur[i] + T.close[i] == row_max) {
                    row_pos = i;
                    break;
                }
            }
        }

        hs[j - 1] = row_max;
        hp[j - 1] = static_cast<int>(row_pos);

        running_max.set_if_better(row_max, static_cast<int>(row_pos), j);

        /* The row just written becomes the row read. */
        std::swap(m_cur, m_last);
        std::swap(ix_cur, ix_last);
        std::swap(iy_cur, iy_last);
    }
}

#endif /* RISEARCH1_HAS_AVX2 */


/* AVX2 is picked at run time, so one binary still starts on a CPU without it and
   the build needs no -march. RISEARCH_NO_AVX2 in the environment forces the
   scalar version, which is how the two are compared from a single build.
   Decided once at startup: a function-local static would put its thread-safe
   initialisation guard inside the caller, which is the hot loop. */
static const bool SCORE_TARGET_CPU_HAS_AVX2 = [] {
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


/* The entry point the alignment calls. Inlined for the same reason
   score_target_scalar is. */
__attribute__((always_inline)) static inline void
score_target(const unsigned char* target_sequence, const QueryProfile& profile, int* const* M,
             int* const* Ix, int* const* Iy, int* hs, int* hp, int n, int threshold,
             RunningMax& running_max)
{
#if RISEARCH1_HAS_AVX2
    // No upside from using AVX2 for small target sizes
    if (profile.m() <= 8) {
        score_target_scalar(target_sequence, profile, M, Ix, Iy, hs, hp, n, threshold, running_max);
        return;
    }
    // Only use AVX2 if the CPU supports
    if (SCORE_TARGET_CPU_HAS_AVX2) {
        score_target_avx2(target_sequence, profile, M, Ix, Iy, hs, hp, n, threshold, running_max);
        return;
    }
#endif
    score_target_scalar(target_sequence, profile, M, Ix, Iy, hs, hp, n, threshold, running_max);
}
