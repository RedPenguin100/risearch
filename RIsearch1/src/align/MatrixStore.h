#pragma once

#include <cstdint>

#include "align/int16_safety.h"
#include "matrix_operations.h"
#include "nucleotide.h" /* NEGINF */


/* Row 0 and column 0 hold the same values whatever the window, so they are set
   once for the whole matrix and no fill writes them. */
template<typename int_type>
static void ris_fill_bounds(int_type** M, int_type** Ix, int_type** Iy, int rows, int cols)
{
    M[0][0] = Ix[0][0] = Iy[0][0] = 0;

    // Target position 0, every query position
    for (auto i = 1; i < cols; i++) {
        Iy[0][i] = M[0][i] = neg_inf<int_type>(); /* not possible before beginning of target seq */
        Ix[0][i] = 0;
    }

    // Query position 0, every target position
    for (auto j = 1; j < rows; j++) {
        Ix[j][0] = M[j][0] = neg_inf<int_type>();
        Iy[j][0] = 0;
    }
}


template<typename int_type>
class MatrixStore {
public:
    explicit MatrixStore(int traceback_len)
        : m_M(allocMatrix<int_type>(traceback_len + 1, traceback_len + 1)),
          m_Ix(allocMatrix<int_type>(traceback_len + 1, traceback_len + 1)),
          m_Iy(allocMatrix<int_type>(traceback_len + 1, traceback_len + 1)),
          m_rows(traceback_len + 1)
    {
        ris_fill_bounds<int_type>(m_M, m_Ix, m_Iy, traceback_len + 1, traceback_len + 1);
    }

    ~MatrixStore()
    {
        freeMatrix<int_type>(m_M, m_rows);
        freeMatrix<int_type>(m_Ix, m_rows);
        freeMatrix<int_type>(m_Iy, m_rows);
    }

    MatrixStore(const MatrixStore&) = delete;
    MatrixStore& operator=(const MatrixStore&) = delete;

    int_type** M() const
    {
        return m_M;
    }
    int_type** Ix() const
    {
        return m_Ix;
    }
    int_type** Iy() const
    {
        return m_Iy;
    }

private:
    int_type** m_M;
    int_type** m_Ix;
    int_type** m_Iy;
    std::uint32_t m_rows;
};
