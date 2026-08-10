#pragma once

#include "HitReporter.h"
#include "math/Matrix.h"
#include "RunningMax.h"
#include "align/optimization/QueryProfile.h"

#include <cstdio>
#include <climits>

#include "force_start.h"
#include "nucleotide.h"

#include "operations.h"
#include "memory/MallocRAII.hpp"
#include "optimization/RowTerms.h"


/* Pass 1 of the sweep: M and Iy for one target row. Both read only the previous
   row, so every i is independent. The rows and the profile lanes are distinct
   allocations, which __restrict states so no runtime alias test is emitted. */
__attribute__((always_inline)) static inline int
RIs_sweep_M_Iy(std::uint32_t m, int* __restrict m_cur, int* __restrict iy_cur,
               const int* __restrict m_last, const int* __restrict ix_last,
               const int* __restrict iy_last, const int* __restrict t_from_m,
               const int* __restrict t_from_ix, const int* __restrict t_from_iy,
               const int* __restrict t_open, const int* __restrict t_close,
               const int* __restrict t_iy_from_m, int iy_ext, int row_max)
{
    m_cur = static_cast<int*>(__builtin_assume_aligned(m_cur + 2, 32)) - 2;
    iy_cur = static_cast<int*>(__builtin_assume_aligned(iy_cur + 2, 32)) - 2;
    m_last = static_cast<const int*>(__builtin_assume_aligned(m_last + 2, 32)) - 2;
    t_from_m = static_cast<const int*>(__builtin_assume_aligned(t_from_m + 2, 32)) - 2;
    t_from_ix = static_cast<const int*>(__builtin_assume_aligned(t_from_ix + 2, 32)) - 2;
    t_from_iy = static_cast<const int*>(__builtin_assume_aligned(t_from_iy + 2, 32)) - 2;
    t_open = static_cast<const int*>(__builtin_assume_aligned(t_open + 2, 32)) - 2;
    t_close = static_cast<const int*>(__builtin_assume_aligned(t_close + 2, 32)) - 2;
    t_iy_from_m = static_cast<const int*>(__builtin_assume_aligned(t_iy_from_m + 2, 32)) - 2;

    for (auto i = 2u; i <= m; i++) {
        const int m_prev = m_last[i - 1];
        const int m_from_m = m_prev + t_from_m[i];
        m_cur[i] = max4(
            /* coming from a match */
            m_prev != 0 ? m_from_m : -1,
            /* coming from gap in target */
            ix_last[i - 1] + t_from_ix[i],
            /* coming from gap in query */
            iy_last[i - 1] + t_from_iy[i],
            /* start fresh */
            t_open[i]);

        // Set max now, position is recovered later (OPTIMIZATION)
        row_max = MAX(row_max, m_cur[i] + t_close[i]);

        /**
         * Iy: target nt against a gap
         */
        iy_cur[i] = MAX(
            // pair at previous row, now bulge
            m_last[i] + t_iy_from_m[i],
            // already bulging, add one more
            iy_last[i] + iy_ext);
    }
    return row_max;
}

__attribute__((target_clones("avx2", "default"))) static void
RIs_linSpace(const unsigned char* query_sequence,  /* query sequence - numeric representation */
             const unsigned char* target_sequence, /* target sequence */
             std::uint32_t m,                      /* query seq length */
             std::uint32_t n,                      /* target seq length */
             short dsm[6][6][6][6],                /* scoring matrix -- TODO variable length!? */
             int threshold,                        /* give out hits higher than that */
             const char* qname,                    /* query name */
             const char* tname,                    /* target name */
             const config_st& config)
{
    // Prepare for RIs calls:
    const auto window = config.tblen + 1;
    MatrixInt M_RI_RAII(window, window);  /* (Mis)Match */
    MatrixInt Ix_RI_RAII(window, window); /* Insertion in x(=query), so x paired to gap (in y) */
    MatrixInt Iy_RI_RAII(window, window); /* Insertion(=bulge) in y(=target) */
    int** M_RI = M_RI_RAII.get();
    int** Ix_RI = Ix_RI_RAII.get();
    int** Iy_RI = Iy_RI_RAII.get();


    const auto reference = reference_from_matrix(config.mat_name);
    RunningMax running_max{};


    MallocRAII<int> hits_score(n);
    MallocRAII<int> hits_pos(n);

    MallocRAII<unsigned char> tmpQseq(config.tblen);
    MallocRAII<unsigned char> tmpTseq(config.tblen);

    const auto alignment_capacity = static_cast<int>(1.5 * config.tblen);
    IA maxHit(alignment_capacity);

    /* matrices for alignment scores ending in different states */
    // Since we only need 2 rows we can optimize the memory layout.
    const std::uint32_t row_stride = (m + 8) / 8 * 8;
    MallocRAII<int> dp_rows;
    {
        void* raw = nullptr;
        if (posix_memalign(&raw, 32, (6 * row_stride + 8) * sizeof(int)) != 0) {
            raw = nullptr;
        }
        dp_rows.reset(static_cast<int*>(raw));
    }
    /* Shifted by 2 so element 2 of every row, where the sweep starts, is 32-byte aligned. */
    int* const dp_base = dp_rows.get() + 6;
    int* const M[2] = {dp_base + 0 * row_stride, dp_base + 1 * row_stride};
    int* const Ix[2] = {dp_base + 2 * row_stride, dp_base + 3 * row_stride};
    int* const Iy[2] = {dp_base + 4 * row_stride, dp_base + 5 * row_stride};


    const QueryProfile profile(query_sequence, m, dsm);
    const int* ix_extend = profile.ix_extend();

    M[0][0] = Ix[0][0] = Iy[0][0] = 0;

    /**
     * Row j=0 -- the empty prefix before the first target nt processed.
     * (target_sequence[n-1], the 3' end, since the target is walked backwards).
     *
     * M and Iy need a target nt to exist, so neither state is reachable here.
     *
     * Ix is 0 because an alignemnt must open on a base pair, and never on a bulge.
     *
     * This scheme can't score dangling ends.
     *
     */
    for (auto i = 1u; i <= m; ++i) {
        Iy[0][i] = M[0][i] = NEGINF;
        Ix[0][i] = 0;
    }


    /* init j=1 row */
    /*init first col (i=0) */
    Iy[1][0] = 0;
    Ix[1][0] = M[1][0] = NEGINF;

    // n - 1 is the last nt in target
    M[1][1] = dsm[GAP][query_sequence[0]][GAP][target_sequence[n - 1]];

    RunningRowMax running_row_max{};
    running_row_max.set(M[1][1] + dsm[query_sequence[0]][GAP][target_sequence[n - 1]][GAP], 1);

    /* (1,1) cell can not be in Ix or Iy state. */
    Ix[1][1] = Iy[1][1] = NEGINF;

    const auto t_last = target_sequence[n - 1];

    for (auto i = 2u; i <= m; i++) {
        const auto q_prev = query_sequence[i - 2];
        const auto q_cur = query_sequence[i - 1];

        const auto open_score = dsm[GAP][q_cur][GAP][t_last];
        const auto close_score = dsm[q_cur][GAP][t_last][GAP];

        M[1][i] = open_score;
        running_row_max.set_if_better(M[1][i] + close_score, i);

        /* removed one option, namely: start new alignment that starts in gap, reflected by (-, Xi;
         * -, -)  */
        Ix[1][i] =
            max3(0,
                 /* prev. match, now gap (no match possible before!?)  - add (Xi-1, Xi; Y1, -) */
                 M[1][i - 1] != 0 ? M[1][i - 1] + dsm[q_prev][q_cur][t_last][GAP] : -1,
                 /* extending existing gap  - add (Xi-1, Xi; -, -) */
                 Ix[1][i - 1] != 0 ? Ix[1][i - 1] + dsm[q_prev][q_cur][GAP][GAP] : -1);

        // There is no previous row, so there can't be a bulge.
        Iy[1][i] = NEGINF;
    }
    running_max.set(running_row_max.score, running_row_max.pos_i, 1);


    /**
     * The first row can only hold 1nt, so NN terms only now begin to apply.
     */

    // hs[j - 1] and hp[j - 1] hold the best alignment ending at pos j
    int* const hs = hits_score.get();
    int* const hp = hits_pos.get();

    hs[0] = running_row_max.score;
    hp[0] = running_row_max.pos_i;


    for (auto j = 2u; j <= n; j++) {
        /* Begin init of i=1 case */
        const auto currentRow = j % 2;
        const auto lastRow = 1 - currentRow;

        const auto target_current = target_sequence[n - j];
        const auto target_prev = target_sequence[n - j + 1];

        /* DSM caching optimization */
        const auto context = QueryProfile::context(target_prev, target_current);
        const QueryProfile::RowLanes T = profile.lanes(context);
        const int* const ix_from_m = profile.ix_from_m(context);
        const auto iy_ext = profile.iy_extend(context);
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



        row_max = RIs_sweep_M_Iy(m, M[currentRow], Iy[currentRow], M[lastRow], Ix[lastRow],
                                 Iy[lastRow], T.m_from_m, T.m_from_ix, T.m_from_iy, T.m_open,
                                 T.close, T.iy_from_m, iy_ext, row_max);

        // Split the loop for performance.
        for (auto i = 2u; i <= m; ++i) {
            /**
             * Ix: query nt against a gap
             */
            Ix[currentRow][i] = MAX(
                // pair at i - 1, now bulge
                M[currentRow][i - 1] + ix_from_m[i],
                // already bulging, add one more
                Ix[currentRow][i - 1] + ix_extend[i]);
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


    HitReporter reporter(query_sequence, target_sequence, n, dsm, profile, config, qname, tname);

    if (!(config.doSubopt && (config.filter_e || config.printShort > 1))) {
        reporter.report(running_max.pos_i, running_max.pos_j, running_max.score, false);
    }
    if (!config.doSubopt) {
        if (config.printShort == 3) {
            printf("%s\t%s\t%d\n", qname, tname, reporter.hitcount());
        }
        return;
    }

    auto j = n;   /* hits_score is 0-based over rows 1..n, so index j-1 holds row j */
    while (j--) {
        if (hits_score[j] <= threshold) {
            continue;
        }
        /* Look back up to `vicinity` rows and take the best of them. */
        auto tmp = MIN(config.vicinity, j);   /* how far back we may look   */
        const auto resume_at = j - tmp++;      /* where the scan resumes     */
        auto locMax = 0u;                       /* offset of the best so far  */
        while (--tmp) {
            if (hits_score[j - tmp] > hits_score[j - locMax]) {
                locMax = tmp;
            }
        }
        j -= locMax;                           /* move onto the window's best */

        reporter.report(hits_pos[j], j + 1, hits_score[j], true);

        /* Resume below the whole window, not just below the hit we reported. */
        j = resume_at;
    }
    if (config.printShort == 3) {
        printf("%s\t%s\t%d\n", qname, tname, reporter.hitcount());
    }

}
