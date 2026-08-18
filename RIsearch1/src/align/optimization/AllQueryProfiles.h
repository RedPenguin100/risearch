#pragma once

#include <cstring>
#include <memory>
#include <vector>

#include "align/optimization/QueryProfile.h"

/* Every query's profile, built once and kept.
 *
 * A profile depends on the query and the matrix, and the matrix is fixed for a
 * run, so a query's profile is the same every time it comes round. The query
 * file is read again for every target record, so a run of T target records
 * against Q queries built Q * T of them where Q would do.
 *
 * Held by the query's place in the file rather than by its bytes: the file is
 * read in the same order every time, so the k'th record is the same query, and
 * the sequence is compared to confirm that before a profile is handed back.
 *
 * A profile is tens of kilobytes and a query file has no bound on how many
 * records it holds, so past a budget the profile is built into one slot that
 * every such query shares, which is what each of them did for itself before.
 *
 * TODO: the two vectors are one allocation each per query and a second pass
 * over the sequence; a flat run of both, sized once the query count is known,
 * would do the same job with less. Worth revisiting alongside the query file
 * being re-read per target record, which is what makes any of this necessary.
 */
class AllQueryProfiles {
public:
    static constexpr std::size_t kBudgetBytes = 32u * 1024u * 1024u;

    /* The profile for the query at `index`. Past the budget it is rebuilt into
       a shared slot each time, so the reference is valid either way. */
    const QueryProfile& get(std::size_t index, const unsigned char* query, std::uint32_t m,
                            short dsm[6][6][6][6], bool has_positive_gap)
    {
        if (index < m_entries.size() && m_entries[index].holds(query, m)) {
            return *m_entries[index].profile;
        }

        const auto bytes = footprint(m);
        const bool keep = m_bytes + bytes <= kBudgetBytes;
        if (keep && index >= m_entries.size()) {
            m_entries.resize(index + 1);
        }

        Entry& e = keep ? m_entries[index] : m_shared;
        e.profile = std::make_unique<QueryProfile>(query, m, dsm, has_positive_gap);
        if (keep) {
            e.sequence.assign(query, query + m);
            m_bytes += bytes;
        }
        return *e.profile;
    }

private:
    /* Eight runs of stride ints per context, and two that are one per query
       position. Enough to bound what is held; an entry itself is smaller. */
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
    /* Where a query past the budget builds its profile; its sequence is left
       empty so holds() never matches and it is rebuilt every time. */
    Entry m_shared;
    std::size_t m_bytes = 0;
};
