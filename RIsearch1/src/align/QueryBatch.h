#pragma once

#include <cstdint>
#include <cstring>

#include "align/dispatch.h"
#include "cli.h"
#include "memory/ByteBuffer.hpp"

/**
 * Queries held back so that several can be swept against one target together.
 *
 * A query and its header are printed from here rather than from the read loop,
 * so that collecting queries before running them cannot move either. What comes
 * out is what the same queries produce one at a time, in the same order.
 */
class QueryBatch {
public:
    static constexpr unsigned kLanes = 16;

    /**
     * The force-start modes align a fixed window instead of sweeping, and the
     * reporting here assumes the ordinary path.
     */
    static bool applies(const config_st& config)
    {
        return !uses_force_start(config);
    }

    /**
     * The sequence and the name are copied: the read loop reuses its buffers for
     * the next record before this batch runs.
     */
    void add(const ByteBuffer& query_seq, const char* name, int query_count, std::uint32_t len)
    {
        Entry& e = m_entries[m_count];
        e.seq.clear();
        e.seq.append(query_seq.data(), query_seq.size());
        e.name.clear();
        e.name.append(name, std::strlen(name));
        e.name.terminate();
        e.query_count = query_count;
        e.len = len;
        m_count++;
    }

    bool full() const
    {
        return m_count == kLanes;
    }

    bool empty() const
    {
        return m_count == 0;
    }

    void run(const ByteBuffer& target_seq, short (&dsm)[6][6][6][6], const char* tname,
             int target_count, std::uint32_t len_seq2, const config_st& config)
    {
        for (auto k = 0u; k < m_count; k++) {
            const Entry& e = m_entries[k];
            if (config.printShort < 2) {
                printf("\n\nquery %d: %s (%u nts) vs. target %d: %s (%u nts)\n\n", e.query_count,
                       e.name.data(), e.len, target_count, tname, len_seq2);
            }
            run_alignment(e.seq, target_seq, dsm, e.name.data(), tname, config);
        }
        clear();
    }

    void clear()
    {
        m_count = 0;
    }

private:
    struct Entry {
        ByteBuffer seq;
        ByteBuffer name;
        int query_count = 0;
        std::uint32_t len = 0;
    };

    Entry m_entries[kLanes];
    unsigned m_count = 0;
};
