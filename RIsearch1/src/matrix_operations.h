#pragma once

#include <cstdint>
#include <cstdio>
#include <cstdlib>

template<typename int_type>
static int_type** allocMatrix(std::uint32_t rows, std::uint32_t cols)
{
    const auto pointer_bytes = static_cast<std::size_t>(rows) * sizeof(int_type*);
    const auto data_bytes = static_cast<std::size_t>(rows) * cols * sizeof(int_type);

    int_type** m = reinterpret_cast<int_type**>(malloc(pointer_bytes + data_bytes));
    if (!m) {
        printf("Cannot allocate matrix with %u rows and %u cols\n", rows, cols);
        exit(1);
    }

    int_type* data = reinterpret_cast<int_type*>(m + rows);
    for (auto i = 0u; i < rows; i++) {
        m[i] = data + static_cast<std::size_t>(i) * cols;
    }
    return m;
}

static float** allocFloatMatrix(std::uint32_t rows, std::uint32_t cols)
{
    const auto pointer_bytes = static_cast<std::size_t>(rows) * sizeof(float*);
    const auto data_bytes = static_cast<std::size_t>(rows) * cols * sizeof(float);

    float** m = reinterpret_cast<float**>(malloc(pointer_bytes + data_bytes));
    if (!m) {
        printf("Cannot allocate float matrix with %u rows and %u cols\n", rows, cols);
        exit(1);
    }

    float* data = reinterpret_cast<float*>(m + rows);
    for (auto i = 0u; i < rows; i++) {
        m[i] = data + static_cast<std::size_t>(i) * cols;
    }
    return m;
}

template<typename int_type>
static void freeMatrix(int_type** m, std::uint32_t)
{
    free(m);
}

static void freeFloatMatrix(float** m, std::uint32_t)
{
    free(m);
}
