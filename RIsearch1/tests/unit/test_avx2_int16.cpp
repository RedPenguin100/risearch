// The int16 lane operations against a scalar statement of what each one means.
// A wrong one here is a wrong alignment, not a crash, so each is pinned directly.

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <climits>
#include <cstdint>
#include <vector>

#include "align/avx2/primitives.h"

#if RISEARCH1_HAS_AVX2

namespace {

using Lane = std::int16_t;
constexpr unsigned kW = 16;

/* Every intrinsic stays inside a function carrying the target attribute; the
   tests themselves only ever see plain arrays. */
using Row = std::array<Lane, kW>;

__attribute__((target("avx2"))) Row apply(__m256i (*op)(__m256i), const Row& in)
{
    Row out{};
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(out.data()),
                        op(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(in.data()))));
    return out;
}

__attribute__((target("avx2"))) Row apply2(__m256i (*op)(__m256i, __m256i), const Row& a,
                                           const Row& b)
{
    Row out{};
    _mm256_storeu_si256(
        reinterpret_cast<__m256i*>(out.data()),
        op(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(a.data())),
           _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b.data()))));
    return out;
}

__attribute__((target("avx2"))) Lane reduce(const Row& in)
{
    return v_hmax<Lane>(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(in.data())));
}

Row prefix_max(const Row& in) { return apply(v_prefix_max<Lane>, in); }
Row broadcast_last(const Row& in) { return apply(v_broadcast_last<Lane>, in); }
Row saturating_add(const Row& a, const Row& b) { return apply2(v_add<Lane>, a, b); }
Row shifted(const Row& a, const Row& b) { return apply2(v_shifted_left_one<Lane>, a, b); }
Row add_unless_zero(const Row& a, const Row& b)
{
    return apply2(v_add_unless_zero_or_neg1<Lane>, a, b);
}

Row ramp(std::initializer_list<int> vals)
{
    Row a{};
    auto i = 0u;
    for (int v : vals) {
        a[i++] = static_cast<Lane>(v);
    }
    return a;
}

} // namespace

TEST(Avx2Int16, PrefixMaxIsTheRunningMaxOfEveryLaneBelow)
{
    const auto in = ramp({3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8, 9, 7, 9, 3});
    const auto got = prefix_max(in);

    Lane best = SHRT_MIN;
    for (auto i = 0u; i < kW; i++) {
        best = std::max(best, in[i]);
        EXPECT_EQ(got[i], best) << "lane " << i;
    }
}

TEST(Avx2Int16, PrefixMaxLeavesNegativeLanesNegative)
{
    // Every lane below zero: a prefix max must never raise one to zero.
    const auto in = ramp({-9, -3, -7, -2, -8, -4, -6, -1, -5, -9, -3, -7, -2, -8, -4, -6});
    const auto got = prefix_max(in);

    Lane best = SHRT_MIN;
    for (auto i = 0u; i < kW; i++) {
        best = std::max(best, in[i]);
        EXPECT_EQ(got[i], best) << "lane " << i;
    }
}

TEST(Avx2Int16, PrefixMaxCarriesAcrossTheHalfwayBoundary)
{
    // The largest value sits in the low half; every high lane must still see it.
    auto in = ramp({0, 0, 0, 0, 0, 0, 0, 100, -5, -5, -5, -5, -5, -5, -5, -5});
    const auto got = prefix_max(in);
    for (auto i = 7u; i < kW; i++) {
        EXPECT_EQ(got[i], 100) << "lane " << i;
    }
}

TEST(Avx2Int16, AddSaturatesRatherThanWraps)
{
    Row a{}, b{};
    a.fill(SHRT_MIN);
    b.fill(-1000);
    const auto got = saturating_add(a, b);
    for (auto v : got) {
        EXPECT_EQ(v, SHRT_MIN); // wrapping would put this near SHRT_MAX
    }
}

TEST(Avx2Int16, HmaxIsTheLargestLane)
{
    const auto in = ramp({3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8, 9, 7, 9, 3});
    EXPECT_EQ(reduce(in), *std::max_element(in.begin(), in.end()));

    const auto neg = ramp({-9, -3, -7, -2, -8, -4, -6, -1, -5, -9, -3, -7, -2, -8, -4, -6});
    EXPECT_EQ(reduce(neg), *std::max_element(neg.begin(), neg.end()));
}

TEST(Avx2Int16, ShiftedLeftOneMovesEveryLaneOneColumnRight)
{
    Row prev{}, cur{};
    for (auto i = 0u; i < kW; i++) {
        prev[i] = static_cast<Lane>(100 + i);
        cur[i] = static_cast<Lane>(200 + i);
    }
    const auto got = shifted(prev, cur);

    EXPECT_EQ(got[0], prev[kW - 1]); // the value that moves in
    for (auto i = 1u; i < kW; i++) {
        EXPECT_EQ(got[i], cur[i - 1]) << "lane " << i;
    }
}

TEST(Avx2Int16, BroadcastLastPutsTheTopLaneEverywhere)
{
    auto in = ramp({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 77});
    const auto got = broadcast_last(in);
    for (auto v : got) {
        EXPECT_EQ(v, 77);
    }
}

TEST(Avx2Int16, AddUnlessZeroOrNeg1SkipsExactlyTheZeroLanes)
{
    const auto base = ramp({0, 5, 0, -3, 7, 0, 2, 0, 1, 0, 4, 0, 6, 0, 8, 0});
    Row term{};
    term.fill(10);
    const auto got = add_unless_zero(base, term);
    for (auto i = 0u; i < kW; i++) {
        EXPECT_EQ(got[i], base[i] == 0 ? -1 : static_cast<Lane>(base[i] + 10)) << "lane " << i;
    }
}

#endif
