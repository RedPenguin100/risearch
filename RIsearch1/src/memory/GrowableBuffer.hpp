#pragma once

#include <cstddef>
#include <cstdlib>

/**
 * An array that grows to whatever a run needs and is reused by the next one.
 *
 * Sized rather than resized: growing throws away what the buffer held, because
 * every caller fills it before reading it. Holding the capacity here is what
 * separates this from MallocRAII, whose callers had to carry a matching
 * capacity of their own beside every buffer.
 */
template<typename T>
class GrowableBuffer {
public:
    GrowableBuffer() = default;

    ~GrowableBuffer()
    {
        std::free(m_data);
    }

    GrowableBuffer(const GrowableBuffer&) = delete;
    GrowableBuffer& operator=(const GrowableBuffer&) = delete;

    void reserve(std::size_t wanted)
    {
        if (wanted <= m_capacity) {
            return;
        }
        std::free(m_data);
        m_data = static_cast<T*>(std::malloc(wanted * sizeof(T)));
        m_capacity = wanted;
    }

    T* get() const
    {
        return m_data;
    }

    T& operator[](std::size_t i) const
    {
        return m_data[i];
    }

private:
    T* m_data = nullptr;
    std::size_t m_capacity = 0;
};
