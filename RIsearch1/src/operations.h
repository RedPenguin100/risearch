#pragma once


#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))


inline int max3(int a, int b, int c)
{
    int max = a;
    if (b > max) {
        max = b;
    }
    if (c > max) {
        max = c;
    }
    return max;
}

inline int min2(int a, int b)
{
    return b > a ? a : b;
}

inline int max4(int a, int b, int c, int d)
{
    int max = a;
    if (b > max) {
        max = b;
    }
    if (c > max) {
        max = c;
    }
    if (d > max) {
        max = d;
    }
    return max;
}

inline int max5(int a, int b, int c, int d, int e)
{
    int max = a;
    if (b > max) {
        max = b;
    }
    if (c > max) {
        max = c;
    }
    if (d > max) {
        max = d;
    }
    if (e > max) {
        max = e;
    }
    return max;
}

inline float max3f(float a, float b, float c)
{
    float max = a;
    if (b > max) {
        max = b;
    }
    if (c > max) {
        max = c;
    }
    return max;
}

inline float max4f(float a, float b, float c, float d)
{
    float max = a;
    if (b > max) {
        max = b;
    }
    if (c > max) {
        max = c;
    }
    if (d > max) {
        max = d;
    }
    return max;
}

inline float max5f(float a, float b, float c, float d, float e)
{
    float max = a;
    if (b > max) {
        max = b;
    }
    if (c > max) {
        max = c;
    }
    if (d > max) {
        max = d;
    }
    if (e > max) {
        max = e;
    }
    return max;
}
