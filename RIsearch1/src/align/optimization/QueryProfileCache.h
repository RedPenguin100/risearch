#pragma once

#include <cstdlib>
#include <cstring>

#include "align/optimization/QueryProfile.h"

/* Profiles kept across the runs that reuse them.
 *
 * A profile depends on the query and the matrix, and the matrix is fixed for a
 * run, so a query's profile is the same every time it comes round. The query
 * file is read again for every target record, so a run of T target records
 * against Q queries builds Q * T of them where Q would do.
 *
 * Keyed on the query's place in the file rather than on its bytes: the file is
 * read in the same order every time, so the k'th record is the same query, and
 * the sequence is compared to confirm that before a profile is handed back.
 *
 * A profile is tens of kilobytes and a query file has no bound on how many
 * records it holds, so the cache stops taking new ones at its budget and says
 * so; a caller that is refused builds its own, as it did before.
 *
 * Note this is worth nothing against a single target record: the query file is
 * read once there, so every slot is filled and none is read back.
 */
class QueryProfileCache {
public:
    static constexpr std::size_t kBudgetBytes = 32u * 1024u * 1024u;

    QueryProfileCache() = default;

    ~QueryProfileCache()
    {
        for (std::size_t i = 0; i < m_count; i++) {
            delete m_slots[i].profile;
            free(m_slots[i].sequence);
        }
        free(m_slots);
        delete m_scratch.profile;
        free(m_scratch.sequence);
    }

    QueryProfileCache(const QueryProfileCache&) = delete;
    QueryProfileCache& operator=(const QueryProfileCache&) = delete;

    /* The profile for the query at `index`. Past the budget it is rebuilt into
       a slot of its own each time, which is what every caller did before this
       cache existed, so the reference is valid either way. */
    const QueryProfile& get(std::size_t index, const unsigned char* query, std::uint32_t m,
                            short dsm[6][6][6][6], bool has_positive_gap)
    {
        if (index < m_count && holds(m_slots[index], query, m)) {
            return *m_slots[index].profile;
        }

        const auto bytes = footprint(m);
        const bool keep = m_bytes + bytes <= kBudgetBytes && reserve(index + 1);
        Slot& s = keep ? m_slots[index] : m_scratch;

        delete s.profile;
        s.profile = new QueryProfile(query, m, dsm, has_positive_gap);
        if (keep) {
            remember(s, query, m);
            m_bytes += bytes;
        }
        return *s.profile;
    }

private:
    /* Plain data, so the run of them is one allocation the cache owns outright
       and clears in its destructor. */
    struct Slot {
        unsigned char* sequence;
        std::uint32_t length;
        QueryProfile* profile;
    };

    /* Only a kept slot needs its sequence back, to confirm the file gave the
       same query at this place as last time. */
    static void remember(Slot& s, const unsigned char* query, std::uint32_t m)
    {
        auto* const kept = static_cast<unsigned char*>(realloc(s.sequence, m));
        if (kept == nullptr) {
            return;
        }
        std::memcpy(kept, query, m);
        s.sequence = kept;
        s.length = m;
    }

    static bool holds(const Slot& s, const unsigned char* query, std::uint32_t m)
    {
        return s.profile != nullptr && s.length == m && std::memcmp(s.sequence, query, m) == 0;
    }

    /* Eight runs of stride ints per context, and two that are one per query
       position. Enough to bound the cache; a slot itself is smaller. */
    static std::size_t footprint(std::uint32_t m)
    {
        const std::size_t stride = static_cast<std::size_t>(m) + 1;
        return (8 * QueryProfile::kContexts + 2) * stride * sizeof(int);
    }

    /* Grows by doubling, so a query file of any length costs a bounded number
       of reallocations. New slots start empty. */
    bool reserve(std::size_t wanted)
    {
        if (wanted <= m_count) {
            return true;
        }
        std::size_t capacity = m_count == 0 ? 16 : m_count;
        while (capacity < wanted) {
            capacity *= 2;
        }
        auto* const grown = static_cast<Slot*>(realloc(m_slots, capacity * sizeof(Slot)));
        if (grown == nullptr) {
            return false;
        }
        std::memset(grown + m_count, 0, (capacity - m_count) * sizeof(Slot));
        m_slots = grown;
        m_count = capacity;
        return true;
    }

    Slot* m_slots = nullptr;
    Slot m_scratch{nullptr, 0, nullptr};
    std::size_t m_count = 0;
    std::size_t m_bytes = 0;
};
