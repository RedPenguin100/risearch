// Tests for the bound the int16 kernel needs. A wrong bound is a wrong answer
// and not a crash, so these pin both the arithmetic and the rejections.

#include <gtest/gtest.h>

#include <vector>

#include "align/int16_safety.h"
#include "dsm.h"
#include "nucleotide.h"

namespace {

std::vector<unsigned char> alternating(unsigned char a, unsigned char b, std::uint32_t m)
{
    std::vector<unsigned char> q(m);
    for (auto i = 0u; i < m; i++) {
        q[i] = (i % 2) ? b : a;
    }
    return q;
}

void matrix(const char* name, short (&out)[6][6][6][6])
{
    getMat(name, &out[0][0][0][0], 0, 0);
}

} // namespace

TEST(Int16Bound, HoldsForTheLengthsAsosAreWritten)
{
    short dsm[6][6][6][6];
    matrix("su95_noGU", dsm);

    for (auto m = 15u; m <= 25u; m++) {
        const auto q = alternating(0, 1, m);
        EXPECT_TRUE(fits_int16(dsm, q.data(), m)) << "m = " << m;
    }
}

TEST(Int16Bound, GrowsWithTheQueryUntilItStopsFitting)
{
    short dsm[6][6][6][6];
    matrix("su95_noGU", dsm);

    const auto short_q = alternating(0, 1, 20);
    const auto long_q = alternating(0, 1, 60);

    EXPECT_LT(int16_bound(dsm, short_q.data(), 20),
              int16_bound(dsm, long_q.data(), 60));
    EXPECT_TRUE(fits_int16(dsm, short_q.data(), 20));
    EXPECT_FALSE(fits_int16(dsm, long_q.data(), 60));
}

TEST(Int16Bound, LeavesRoomBelowWhatAShortHolds)
{
    short dsm[6][6][6][6];
    matrix("su95_noGU", dsm);

    const auto q = alternating(0, 1, 20);
    const auto bound = int16_bound(dsm, q.data(), 20);

    EXPECT_GT(bound, 0);
    EXPECT_LE(bound, 30000);
    // The cap is what keeps a saturating add off every real value.
    EXPECT_LT(bound, 32767);
}

TEST(BulgeExtension, HoldsForEveryMatrixRisearchShips)
{
    for (const char* name : {"su95", "su95_noGU", "t04", "t99", "slh04_noGU"}) {
        short dsm[6][6][6][6];
        matrix(name, dsm);
        const auto q = alternating(0, 1, 20);
        EXPECT_TRUE(!is_pos_target_bulge(dsm) && !is_pos_query_bulge(dsm)) << name;
    }
}

TEST(BulgeExtension, FailsWhenExtendingATargetBulgePays)
{
    short dsm[6][6][6][6];
    matrix("su95_noGU", dsm);
    const auto q = alternating(0, 1, 20);
    ASSERT_TRUE(!is_pos_target_bulge(dsm) && !is_pos_query_bulge(dsm));

    dsm[GAP][GAP][0][1] = 1;

    EXPECT_FALSE(!is_pos_target_bulge(dsm) && !is_pos_query_bulge(dsm));
    EXPECT_FALSE(fits_int16(dsm, q.data(), 20));
}

TEST(BulgeExtension, FailsWhenExtendingTheQuerysOwnBulgePays)
{
    short dsm[6][6][6][6];
    matrix("su95_noGU", dsm);
    const auto q = alternating(0, 1, 20);
    ASSERT_TRUE(!is_pos_target_bulge(dsm) && !is_pos_query_bulge(dsm));

    dsm[q[0]][q[1]][GAP][GAP] = 1;

    EXPECT_FALSE(!is_pos_target_bulge(dsm) && !is_pos_query_bulge(dsm));
    EXPECT_FALSE(fits_int16(dsm, q.data(), 20));
}

TEST(BulgeExtension, IsAPropertyOfTheMatrixAndNoQuery)
{
    short dsm[6][6][6][6];
    matrix("su95_noGU", dsm);
    const auto q = alternating(0, 1, 20);

    /* A dinucleotide this query cannot produce still rejects the matrix, which is
       what lets the check run once before any sequence is read. */
    dsm[1][1][GAP][GAP] = 1;

    EXPECT_TRUE(is_pos_query_bulge(dsm));
    EXPECT_FALSE(fits_int16(dsm, q.data(), 20));
}

TEST(Int16Bound, HoldsForEveryMatrixRisearchShips)
{
    for (const char* name : {"su95", "su95_noGU", "t04", "t99", "slh04_noGU"}) {
        short dsm[6][6][6][6];
        matrix(name, dsm);
        const auto q = alternating(0, 1, 20);
        EXPECT_TRUE(fits_int16(dsm, q.data(), 20)) << name;
    }
}
