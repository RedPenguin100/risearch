#pragma once

#include <cstring>
#include <memory>
#include <vector>

#include "align/optimization/QueryProfile.h"

/* Profiles kept across the runs that reuse them.
 *
 * A profile depends on the query and the matrix, and the matrix is fixed for a
 * run, so a query's profile is the same every time it comes round. The query
 * file is read again for every target record, so a run of T targets against Q
 * queries builds Q * T of them where Q would do.
 *
 * Keyed on the query's place in the file rather than on its bytes: the file is
 * read in the same order every time, so the k'th record is the same query, and
 * the sequence is compared to confirm that before a profile is handed back.
 *
 * A profile is tens of kilobytes and a query file has no bound on how many
 * records it holds, so the cache stops taking new ones at its budget and says
 * so; a caller that is refused builds its own, as it did before.
 */
class QueryProfileCache {
public:
    static constexpr std::size_t kBudgetBytes = 32u * 1024u * 1024u;

    /* The profile for the query at `index`, or nullptr when this cache is not
       holding one and will not take it. */
    const QueryProfile* get(std::size_t index, const unsigned char* query, std::uint32_t m,
                            short dsm[6][6][6][6], bool has_positive_gap)
    {
        if (index < m_entries.size() && m_entries[index].holds(query, m)) {
            return m_entries[index].profile.get();
        }

        const auto bytes = footprint(m);
        if (m_bytes + bytes > kBudgetBytes) {
            return nullptr;
        }

        if (index >= m_entries.size()) {
            m_entries.resize(index + 1);
        }
        Entry& e = m_entries[index];
        m_bytes += bytes;
        e.sequence.assign(query, query + m);
        e.profile = std::make_unique<QueryProfile>(query, m, dsm, has_positive_gap);
        return e.profile.get();
    }

private:
    /* Eight runs of stride ints per context, and two that are one per query
       position. Close enough to bound the cache; the entry itself is smaller. */
    static std::size_t footprint(std::uint32_t m)
    {
        const std::size_t stride = static_cast<std::size_t>(m) + 1;
        return (8 * QueryProfile::kContexts + 2) * stride * sizeof(int);
    }

    struct Entry {
        std::vector<unsigned char> sequence;
        std::unique_ptr<QueryProfile> profile;

        bool holds(const unsigned char* query, std::uint32_t m) const
        {
            return profile && sequence.size() == m &&
                   std::memcmp(sequence.data(), query, m) == 0;
        }
    };

    std::vector<Entry> m_entries;
    std::size_t m_bytes = 0;
};
