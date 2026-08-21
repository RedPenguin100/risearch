#pragma once

#include <cctype>
#include <climits>
#include <cstdint>
#include <cstdio>

#include "memory/ByteBuffer.hpp"

#define GAP 5 /* position of '-' in alphabet, not as define if read from matrix... */

#define NEGINF INT_MIN / 2

inline unsigned char nt2index(char nt)
{
    switch (nt) {
    case 'A':
    case 'a':
        return 0;
    case 'C':
    case 'c':
        return 1;
    case 'G':
    case 'g':
        return 2;
    case 'T':
    case 't':
    case 'U':
    case 'u':
        return 3;
    case 'N':
    case 'n':
        return 4;
    case '-':
        return 5; /*any case!? */
    default:
        fprintf(stderr, "Nonstandard nucleotide code: %c\n", nt);
        return 4;
    }
}

inline char index2nt(unsigned char ix)
{
    switch (ix) {
    case 0:
        return 'A';
    case 1:
        return 'C';
    case 2:
        return 'G';
    case 3:
        return 'U';
    case 4:
        return 'N';
    case 5:
        return '-'; /*any case!? */
    default:
        fprintf(stderr, "\nUnknown symbol >>%d<< found.\n", ix);
        exit(1); /*just skip it!? not fail? */
    }
}


/* The code seq2ix gives one input character, for the characters that have one.
   Anything else -- a gap, a nonstandard code, a character that is not sequence
   at all -- is left to the switch, which is the only place that reports. */
static constexpr unsigned char kSeqCodeOther = 255;

struct SeqCodeTable {
    unsigned char code[256];

    constexpr SeqCodeTable() : code()
    {
        for (auto& c : code) {
            c = kSeqCodeOther;
        }
        code[static_cast<unsigned char>('A')] = code[static_cast<unsigned char>('a')] = 0;
        code[static_cast<unsigned char>('C')] = code[static_cast<unsigned char>('c')] = 1;
        code[static_cast<unsigned char>('G')] = code[static_cast<unsigned char>('g')] = 2;
        code[static_cast<unsigned char>('T')] = code[static_cast<unsigned char>('t')] = 3;
        code[static_cast<unsigned char>('U')] = code[static_cast<unsigned char>('u')] = 3;
        code[static_cast<unsigned char>('N')] = code[static_cast<unsigned char>('n')] = 4;
    }
};

static constexpr SeqCodeTable kSeqCodes{};


/* Encodes a sequence into matrix indices, dropping gap characters. The buffer
 * ends up exactly as long as the encoding, so there is no gap count to hand back
 * and no length for the caller to correct. False means the sequence held a
 * character that is not a nucleotide code at all, and should be skipped.
 */
inline bool seq2ix(std::uint32_t len, const char* seq, ByteBuffer& retIx, const char* name,
                   const char* type)
{
    retIx.clear();
    /* Gaps are dropped and nothing expands, so len is the most this can produce
       and no character has to check for room. */
    retIx.reserve(len);

    for (auto i = 0u; i < len; i++) {
        // kSeqCodes.code[..] is a faster way to access nucleotides than a switch
        const auto code = kSeqCodes.code[static_cast<unsigned char>(seq[i])];
        if (code != kSeqCodeOther) {
            retIx.push_back(static_cast<char>(code));
            continue;
        }
        switch (seq[i]) {
        case '-':
        case '.': /*discard gaps from input*/
            break;
        default:
            if (isalpha(seq[i])) {
                fprintf(stderr,
                        "Nonstandard nucleotide code '%c' in %s sequence '%s'. Replaced with 'N'\n",
                        seq[i], type, name);
                retIx.push_back(4);
                break;
            } else { /*skip sequence!? */
                fprintf(stderr,
                        "Unexpected character '%c' in %s sequence '%s'. Skipping sequence.\n",
                        seq[i], type, name);
                return false;
            }
        }
    }
    return true;
}
