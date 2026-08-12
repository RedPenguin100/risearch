#pragma once

#include <cstring>

#include "InteractionAlignment.h"
#include "RunningMax.h"
#include "cli.h"
#include "memory/ByteBuffer.hpp"
#include "nucleotide.h" /* GAP, NEGINF */
#include "operations.h"
#include "avx2/primitives.h"
#include "optimization/QueryProfile.h"
#include "string_util.h"

enum class TraceState { TRACE_M = 0, TRACE_IX = 1, TRACE_IY = 2, TRACE_DONE = 3 };

/* '|' for a Watson-Crick pair, '.' for a G:U wobble, ' ' for a mismatch.
   The indices are chosen so a pair sums to 3 and a wobble to 5. */
static char pair_symbol(unsigned char q, unsigned char t)
{
    if (q + t == 3) {
        return '|';
    }
    if (q + t == 5) {
        return '.';
    }
    return ' ';
}

static void emit_pair(IA* hit, int l, const unsigned char* q, const unsigned char* t, int i, int j)
{
    const auto qn = q[i];
    const auto tn = t[j];
    hit->ali_seq1[l] = index2nt(qn);
    hit->ali_ia[l] = pair_symbol(qn, tn);
    hit->ali_seq2[l] = index2nt(tn);
}

static void emit_query_bulge(IA* hit, int l, const unsigned char* q, int i)
{
    hit->ali_seq1[l] = index2nt(q[i]);
    hit->ali_ia[l] = ' ';
    hit->ali_seq2[l] = '-';
}

static void emit_target_bulge(IA* hit, int l, const unsigned char* t, int j)
{
    hit->ali_seq1[l] = '-';
    hit->ali_ia[l] = ' ';
    hit->ali_seq2[l] = index2nt(t[j]);
}


/**
 * The old version was transposed. When transposing we unlocked performance, but changed
 * the order of the "best hits", so here we transpose it back without losing major performance.
 */
static RunningMax transpose_best_cell(const unsigned char* target_seq, int m, int n, int** M,
                                      const QueryProfile& profile, int q_offset, const int* best,
                                      const RunningVectorMax& first_row)
{
    RunningVectorMax inverted_best{};
    inverted_best.set(best[1], 1);
    for (auto i = 2; i <= m; i++) {
        inverted_best.set_if_better(best[i], i);
    }

    /* Target position 1 came first, so it keeps ties. */
    RunningMax running_max{first_row.score, first_row.pos_i, 1};

    // if inverted_best.score not better, return the running max of the first row
    if (!running_max.test(inverted_best.score)) {
        return running_max;
    }

    const auto qp = q_offset + inverted_best.pos_i;
    for (auto j = 2; j <= n; j++) {
        const auto ctx = QueryProfile::context(target_seq[j - 2], target_seq[j - 1]);
        if (M[j][inverted_best.pos_i] + profile.row(ctx).close[qp] == inverted_best.score) {
            running_max.set(inverted_best.score, inverted_best.pos_i, j);
            break;
        }
    }
    return running_max;
}

/* Fills the three matrices and, per query column, the best M + close seen in any
 * target position. RIs backtracks through what this leaves behind, so every cell
 * is kept rather than two rows as the sweep keeps.
 */
static void ris_fill_scalar(const unsigned char* target_seq, int m, int n, int** M, int** Ix,
                            int** Iy, const QueryProfile& profile, int q_offset, int* best,
                            RunningVectorMax& first_row)
{
    const int* const ix_ext = profile.ix_extend();


    M[0][0] = Ix[0][0] = Iy[0][0] = 0;

    // Target position 0, every query position
    for (auto i = 1; i <= m; i++) {
        Iy[0][i] = M[0][i] = NEGINF; /* not possible before beginning of target seq */
        Ix[0][i] = 0;
    }

    // Query position 0, every target position
    for (auto j = 1; j <= n; j++) {
        Ix[j][0] = M[j][0] = NEGINF;
        Iy[j][0] = 0;
    }

    /*
     * Query position 1 and target position 1 have to be handled explicitly since
     * at this point we do not have two residues to use.
     *
     * Handle the (1,1) cell explicitly since the boundary recursion includes (i-2) or (j-2) cases.
     */

    // Use this  QueryProfile to fetch terms that ignore the previous target nt (GAP)
    // and relate to first target nt (target_seq[0])
    const auto T = profile.row(QueryProfile::context(GAP, target_seq[0]));
    const int* const ix_from_m_1 = T.ix_from_m;

    M[1][1] = T.m_open[q_offset + 1];
    best[1] = INT_MIN;
    /* Target position 1 only: its j is always 1, so the position is a query
       column and nothing more. */
    first_row.set(M[1][1] + T.close[q_offset + 1], 1);

    /* (1,1) cell can not be in Ix or Iy state. */
    Ix[1][1] = Iy[1][1] = NEGINF;


    /* init target position 1 */
    for (auto i = 2; i <= m; i++) {
        const auto qp = q_offset + i;
        M[1][i] = T.m_open[qp];
        best[i] = INT_MIN; // must start empty, will be filled later.
        first_row.set_if_better(M[1][i] + T.close[qp], i);

        Ix[1][i] = max3(M[1][i - 1] != 0 ? M[1][i - 1] + ix_from_m_1[qp] : -1,
                        Ix[1][i - 1] != 0 ? Ix[1][i - 1] + ix_ext[qp] : -1, 0);

        Iy[1][i] = NEGINF;
    }

    const auto qp_first = q_offset + 1;

    for (auto j = 2; j <= n; j++) {
        const auto t = profile.row(QueryProfile::context(target_seq[j - 2], target_seq[j - 1]));
        const auto iy_ext = t.iy_extend;

        M[j][1] = t.m_open[qp_first];
        best[1] = MAX(best[1], M[j][1] + t.close[qp_first]);

        Ix[j][1] = NEGINF;

        Iy[j][1] = max3(M[j - 1][1] != 0 ? M[j - 1][1] + t.iy_from_m[qp_first] : -1,
                         Iy[j - 1][1] != 0 ? Iy[j - 1][1] + iy_ext : -1, 0);

        for (auto i = 2; i <= m; i++) {
            const auto qp = q_offset + i;

            M[j][i] = max4(
                // continue from a pair
                M[j - 1][i - 1] != 0 ? M[j - 1][i - 1] + t.m_from_m[qp] : -1,
                // close a bulge in query
                Ix[j - 1][i - 1] != 0 ? Ix[j - 1][i - 1] + t.m_from_ix[qp] : -1,
                // close a bulge in target
                Iy[j - 1][i - 1] != 0 ? Iy[j - 1][i - 1] + t.m_from_iy[qp] : -1,
                // start fresh
                t.m_open[qp]);

            best[i] = MAX(best[i], M[j][i] + t.close[qp]);


            Ix[j][i] = max3(
                // Open a bulge in query
                M[j][i - 1] != 0 ? M[j][i - 1] + t.ix_from_m[qp] : -1,
                // Continue a bulge in query
                Ix[j][i - 1] != 0 ? Ix[j][i - 1] + ix_ext[qp] : -1, 0);

            Iy[j][i] = max3(
                // Open a bulge in target
                M[j - 1][i] != 0 ? M[j - 1][i] + t.iy_from_m[qp] : -1,
                // Continue a bulge in target
                Iy[j - 1][i] != 0 ? Iy[j - 1][i] + iy_ext : -1, 0);
        }
    }
}


#if RISEARCH1_HAS_AVX2

/* The same fill, eight query positions at a time. It is meant to be read beside
 * ris_fill_scalar: every statement below is one of that function's with each
 * scalar replaced by eight, and the differences are only these.
 *
 *  - Target position 1 and query position 1 stay scalar. The first has no
 *    previous target nt for its terms, the second no column to its left for a
 *    block to read.
 *
 *  - M and Iy read only the previous target position, so eight of their columns
 *    are independent and a block computes them outright. Blocks cover columns
 *    2..m with the last backed up to end at m; it recomputes the columns it
 *    overlaps to the same values, and best only ever takes a max, so overlapping
 *    changes nothing.
 *
 *  - Ix and Iy drop their tests for a predecessor of exactly 0. Both end in a max
 *    with 0, so the -1 a failed test yields and the bulge it avoids are both <= 0
 *    and neither can win. That holds only while no bulge term is positive, which
 *    is what has_positive_gap() rules out and what the dispatch below requires. M
 *    keeps its tests: its fourth candidate is an opening score rather than 0, so
 *    -1 can still win there.
 *
 *  - Ix reads the column to its left in the row it writes, so it is rewritten as
 *    a running max over candidates with ix_prefix taken out -- the same rewrite
 *    the sweep uses. A running max cannot revisit a column already folded into
 *    its carry, so only whole blocks go through it and the last columns finish
 *    serially.
 */
__attribute__((target("avx2"))) static void
ris_fill_avx2(const unsigned char* target_seq, int m, int n, int** M, int** Ix, int** Iy,
              const QueryProfile& profile, int q_offset, int* best, RunningVectorMax& first_row)
{
    const int* const ix_ext = profile.ix_extend();

    M[0][0] = Ix[0][0] = Iy[0][0] = 0;

    for (auto i = 1; i <= m; i++) {
        Iy[0][i] = M[0][i] = NEGINF;
        Ix[0][i] = 0;
    }
    for (auto j = 1; j <= n; j++) {
        Ix[j][0] = M[j][0] = NEGINF;
        Iy[j][0] = 0;
    }

    const auto T = profile.row(QueryProfile::context(GAP, target_seq[0]));
    const int* const ix_from_m_1 = T.ix_from_m;

    M[1][1] = T.m_open[q_offset + 1];
    best[1] = INT_MIN;
    first_row.set(M[1][1] + T.close[q_offset + 1], 1);
    Ix[1][1] = Iy[1][1] = NEGINF;

    for (auto i = 2; i <= m; i++) {
        const auto qp = q_offset + i;
        M[1][i] = T.m_open[qp];
        best[i] = INT_MIN;
        first_row.set_if_better(M[1][i] + T.close[qp], i);

        Ix[1][i] = max3(M[1][i - 1] != 0 ? M[1][i - 1] + ix_from_m_1[qp] : -1,
                        Ix[1][i - 1] != 0 ? Ix[1][i - 1] + ix_ext[qp] : -1, 0);

        Iy[1][i] = NEGINF;
    }

    const auto qp_first = q_offset + 1;
    const __m256i zero = _mm256_setzero_si256();

    for (auto j = 2; j <= n; j++) {
        const auto t = profile.row(QueryProfile::context(target_seq[j - 2], target_seq[j - 1]));
        const auto iy_ext = t.iy_extend;

        const int* const m_prev = M[j - 1];
        const int* const ix_prev = Ix[j - 1];
        const int* const iy_prev = Iy[j - 1];
        int* const m_cur = M[j];
        int* const ix_cur = Ix[j];
        int* const iy_cur = Iy[j];

        m_cur[1] = t.m_open[qp_first];
        best[1] = MAX(best[1], m_cur[1] + t.close[qp_first]);

        ix_cur[1] = NEGINF;

        iy_cur[1] = max3(m_prev[1] != 0 ? m_prev[1] + t.iy_from_m[qp_first] : -1,
                         iy_prev[1] != 0 ? iy_prev[1] + iy_ext : -1, 0);

        /* Offset once, so a block's terms are indexed by the same i as its rows. */
        const int* const from_m = t.m_from_m + q_offset;
        const int* const from_ix = t.m_from_ix + q_offset;
        const int* const from_iy = t.m_from_iy + q_offset;
        const int* const open = t.m_open + q_offset;
        const int* const close = t.close + q_offset;
        const int* const iy_from_m = t.iy_from_m + q_offset;
        const int* const ix_scan = t.ix_from_m_scan + q_offset;
        const int* const ix_pref = t.ix_prefix + q_offset;

        const __m256i v_iy_ext = all_lanes(iy_ext);

        for (auto start = 2; start <= m; start += 8) {
            /* The clamp is what makes the last block end exactly at m. */
            const auto i = MIN(start, m - 7);

            /* For each column the block writes, its diagonal predecessor: one
               target position back and one query position back. */
            const __m256i m_diag = vec_load(m_prev + i - 1);
            const __m256i ix_diag = vec_load(ix_prev + i - 1);
            const __m256i iy_diag = vec_load(iy_prev + i - 1);

            const __m256i m_new = vmax4(
                // continue from a pair
                add_unless_zero(m_diag, vec_load(from_m + i), -1),
                // close a bulge in query
                add_unless_zero(ix_diag, vec_load(from_ix + i), -1),
                // close a bulge in target
                add_unless_zero(iy_diag, vec_load(from_iy + i), -1),
                // start fresh
                vec_load(open + i));
            vec_store(m_cur + i, m_new);

            vec_store(best + i, vmax(vec_load(best + i), vadd(m_new, vec_load(close + i))));

            /* Iy's predecessors are vertical: previous target position, same
               column, so no -1 on the address. */
            vec_store(iy_cur + i,
                      vmax3(
                          // Open a bulge in target
                          vadd(vec_load(m_prev + i), vec_load(iy_from_m + i)),
                          // Continue a bulge in target
                          vadd(vec_load(iy_prev + i), v_iy_ext),
                          zero));
        }

        /* Ix, as a running max over the same candidates with ix_prefix taken
           out. The 0 arm becomes -ix_prefix in that space; the carry starts
           below every candidate, and Ix[j][1] is unreachable so it cannot win. */
        __m256i carry = all_lanes(NEGINF);
        auto i = 2;
        for (; i + 8 <= m + 1; i += 8) {
            const __m256i prefix = vec_load(ix_pref + i);
            const __m256i candidates =
                vmax(// Open a bulge in query
                     vadd(vec_load(m_cur + i - 1), vec_load(ix_scan + i)),
                     // or hold nothing yet, which is this column's 0
                     vsub(zero, prefix));
            const __m256i winner = vmax(vec_prefix_max(candidates), carry);
            /* Putting ix_prefix back turns the carried quantity into the real Ix,
               which is what the next column and the backtrack read. */
            vec_store(ix_cur + i, vadd(winner, prefix));
            carry = _mm256_permutevar8x32_epi32(winner, all_lanes(7));
        }
        /* The columns left over go through one more block, backed up to end at m.
           A running max cannot revisit a column already folded into its carry, so
           the carry is rebuilt from the column just below the block: taking
           ix_prefix back out of the Ix stored there is what the scan would have
           been carrying. */
        if (i <= m) {
            const auto back = m - 7;
            const __m256i prefix = vec_load(ix_pref + back);
            const __m256i candidates =
                vmax(vadd(vec_load(m_cur + back - 1), vec_load(ix_scan + back)),
                     vsub(zero, prefix));
            const __m256i winner = vmax(vec_prefix_max(candidates),
                                        all_lanes(ix_cur[back - 1] - ix_pref[back - 1]));
            vec_store(ix_cur + back, vadd(winner, prefix));
        }
    }
}

#endif

/* The vector fill needs eight columns to the right of column 1 for a block, and
   it may only drop Ix's and Iy's zero tests where no bulge term is positive. */
static bool ris_fill_is_vectorized(int m, const QueryProfile& profile)
{
#if RISEARCH1_HAS_AVX2
    return m >= 9 && !profile.has_positive_gap() && CPU_HAS_AVX2;
#else
    (void)m;
    (void)profile;
    return false;
#endif
}

static void ris_fill(const unsigned char* target_seq, int m, int n, int** M, int** Ix, int** Iy,
                     const QueryProfile& profile, int q_offset, int* best,
                     RunningVectorMax& first_row)
{
#if RISEARCH1_HAS_AVX2
    if (ris_fill_is_vectorized(m, profile)) {
        ris_fill_avx2(target_seq, m, n, M, Ix, Iy, profile, q_offset, best, first_row);
        return;
    }
#endif
    ris_fill_scalar(target_seq, m, n, M, Ix, Iy, profile, q_offset, best, first_row);
}

static void RIs(const unsigned char* query_seq,  /* query sequence - numeric representation */
                const unsigned char* target_seq, /* target sequence - reversed */
                int m,                           /* query seq length */
                int n,                           /* target seq length */
                IA* hit,                         /* pointer to struct, fill results */
                const config_st& config, int** M, int** Ix, int** Iy, const QueryProfile& profile,
                int q_offset,
                int* best // We use `best` parameter to preserve output order of old C version
                )
{
    /* The matrices are indexed [target][query] so that a row fixes the target
       context and the terms it reads are runs over consecutive query positions. */
    /* The backtrack re-derives which arm produced each cell, so it reads the
       same extension terms the fill did. */
    const int* const ix_ext = profile.ix_extend();

    RunningVectorMax first_row{};
    ris_fill(target_seq, m, n, M, Ix, Iy, profile, q_offset, best, first_row);
    const auto running_max =
        transpose_best_cell(target_seq, m, n, M, profile, q_offset, best, first_row);


    /*backtrack*/
    // we write to capacity and capacity + 1 indices, so we need -2
    // as a buffer zone.
    const auto capacity = static_cast<int>(1.5 * config.tblen) - 2;

    auto i = running_max.pos_i;
    auto j = running_max.pos_j;
    auto k = TraceState::TRACE_M;
    auto l = 0; /* alilen so far -used in backtrack */

    while ((i > 0 && j > 0) && (M[j][i] > 0 || Ix[j][i] > 0 || Iy[j][i] > 0)) {
        if (l > capacity) {
            printf("Interaction longer than max, so the following is only the end of the full "
                   "alignment:\n");
            /*alt: stop here / reallocate? prevent creation of longer alignments in first place? */
            break;
        }


        /*l++ in end of while instead of every sub? */

        const auto qp = q_offset + i;


        switch (k) {
        case TraceState::TRACE_M: {
            if (const auto open_score =
                    profile.row(QueryProfile::context(GAP, target_seq[j - 1])).m_open[qp];
                M[j][i] == open_score) {
                k = TraceState::TRACE_DONE;
            } else {
                const auto& t =
                    profile.row(QueryProfile::context(target_seq[j - 2], target_seq[j - 1]));

                if (M[j][i] == M[j - 1][i - 1] + t.m_from_m[qp]) {
                    k = TraceState::TRACE_M;
                } else if (M[j][i] == Ix[j - 1][i - 1] + t.m_from_ix[qp]) {
                    k = TraceState::TRACE_IX;
                } else if (M[j][i] == Iy[j - 1][i - 1] + t.m_from_iy[qp]) {
                    k = TraceState::TRACE_IY;
                } else {
                    printf("unexpected value in k=0.\n");
                }
            }
            emit_pair(hit, l++, query_seq, target_seq, --i, --j);
            break;
        }
        case TraceState::TRACE_IX: {
            // in this case, t_prev doesn't matter so we put it as GAP
            // avoids bugs in the j == 1 case.
            /* seq1(query) paired to a gap (in target) */
            const auto gap_row = profile.row(QueryProfile::context(GAP, target_seq[j - 1]));
            if (Ix[j][i] == M[j][i - 1] + gap_row.ix_from_m[qp]) {
                k = TraceState::TRACE_M; /* open a new gap coming from match */
            } else if (Ix[j][i] == Ix[j][i - 1] + ix_ext[qp]) {
                k = TraceState::TRACE_IX; /* extend existing gap */
            } else if (Ix[j][i] == profile.row(QueryProfile::context(GAP, GAP)).m_open[qp]) {
                fprintf(stderr, "\nErr: This alignment starts in a gap - not even an option!?\n");
                k = TraceState::TRACE_DONE; /* start new alignment with gap; not possible, prevented
                                               by scoring... */
            } else {
                printf("unexpected case in k=1 : %d\n", Ix[j][i]);
            }
            emit_query_bulge(hit, l++, query_seq, --i);
            break;
        }
        case TraceState::TRACE_IY: {
            const auto context = QueryProfile::context(target_seq[j - 2], target_seq[j - 1]);

            /* seq2(target) paired to a gap (in query) */
            if (Iy[j][i] == M[j - 1][i] + profile.row(context).iy_from_m[qp]) {
                k = TraceState::TRACE_M; /* open a new gap coming from match */
            } else if (Iy[j][i] == Iy[j - 1][i] + profile.row(context).iy_extend) {
                k = TraceState::TRACE_IY; /* extend existing gap */
            } else if (Iy[j][i] ==
                       profile.row(QueryProfile::context(GAP, target_seq[j - 1])).iy_extend) {
                k = TraceState::TRACE_DONE;
                fprintf(stderr, "\nErr: This alignment starts in a gap - not even an option!?\n");
            } else {
                printf("unexpected case in k=2 : %d\n", Iy[j][i]);
            }
            emit_target_bulge(hit, l++, target_seq, --j);
            break;
        }
        default: {
            fprintf(stderr, "\nThis should really NEVER happen!\n");
            break;
        }
        }

        if (TraceState::TRACE_DONE == k) {
            break;
        }
    }
    hit->ali_seq1[l] = '\0';
    hit->ali_ia[l] = '\0';
    hit->ali_seq2[l] = '\0';

    /* reverse sequences in the end*/

    if (l > 0) { // fixes a pre-existing bug, access to invalid memory [-1]
        reverse_inplace(hit->ali_seq1.get(), l - 1);
        reverse_inplace(hit->ali_ia.get(), l - 1);
        reverse_inplace(hit->ali_seq2.get(), l - 1);
    }

    hit->qbeg = i + 1;
    hit->qend = running_max.pos_i;
    hit->tbeg = n + 1 - running_max.pos_j;
    hit->tend = n - j;
    hit->max = running_max.score;
}
