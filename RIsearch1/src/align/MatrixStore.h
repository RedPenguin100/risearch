#pragma once

#include <cstdint>

#include "align/int16_safety.h"
#include "matrix_operations.h"
#include "nucleotide.h" /* NEGINF */


/* Row 0 and column 0 hold the same values whatever the window, so they are set
   once for the whole matrix and no fill writes them. */
static void ris_fill_bounds(std::int32_t** M, std::int32_t** Ix, std::int32_t** Iy, int rows,
                            int cols)
{
    M[0][0] = Ix[0][0] = Iy[0][0] = 0;

    // Target position 0, every query position
    for (auto i = 1; i < cols; i++) {
        Iy[0][i] = M[0][i] = NEGINF; /* not possible before beginning of target seq */
        Ix[0][i] = 0;
    }

    // Query position 0, every target position
    for (auto j = 1; j < rows; j++) {
        Ix[j][0] = M[j][0] = NEGINF;
        Iy[j][0] = 0;
    }
}


class MatrixStore {
public:
    explicit MatrixStore(int traceback_len)
        : m_M(allocMatrix<std::int32_t>(traceback_len + 1, traceback_len + 1)),
          m_Ix(allocMatrix<std::int32_t>(traceback_len + 1, traceback_len + 1)),
          m_Iy(allocMatrix<std::int32_t>(traceback_len + 1, traceback_len + 1)),
          m_rows(traceback_len + 1)
    {
        ris_fill_bounds(m_M, m_Ix, m_Iy, traceback_len + 1, traceback_len + 1);
    }

    ~MatrixStore()
    {
        freeMatrix<std::int32_t>(m_M, m_rows);
        freeMatrix<std::int32_t>(m_Ix, m_rows);
        freeMatrix<std::int32_t>(m_Iy, m_rows);
    }

    MatrixStore(const MatrixStore&) = delete;
    MatrixStore& operator=(const MatrixStore&) = delete;

    std::int32_t** M() const
    {
        return m_M;
    }
    std::int32_t** Ix() const
    {
        return m_Ix;
    }
    std::int32_t** Iy() const
    {
        return m_Iy;
    }

private:
    std::int32_t** m_M;
    std::int32_t** m_Ix;
    std::int32_t** m_Iy;
    std::uint32_t m_rows;
};
