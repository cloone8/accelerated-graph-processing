#ifndef __HIGHRESTIMER_H__
#define __HIGHRESTIMER_H__

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"

#include <time.h>
#include <stdlib.h>
#include <stdio.h>

/**
 *  This file has a header-only implementation of the HIGHRESTIMER graph utilities
 *  and structure, to circumvent weird cross-compilation issues
 */

#ifdef WIN32
    #include <windows.h>

    typedef double highrestimer_t;

    static highrestimer_t get_highrestime(void) {
        LARGE_INTEGER t, f;
        QueryPerformanceCounter(&t);
        QueryPerformanceFrequency(&f);
        return (highrestimer_t)t.QuadPart/(highrestimer_t)f.QuadPart;
    }

    static double highrestime_diff(highrestimer_t start, highrestimer_t end) {
        return end - start;
    }

#elif defined (__unix__) || (defined (__APPLE__) && defined (__MACH__))

    #include <sys/time.h>
    #include <sys/resource.h>

    typedef double highrestimer_t;

    static highrestimer_t get_highrestime(void) {
        struct timeval t;
        gettimeofday(&t, NULL);
        return t.tv_sec + (t.tv_usec * 1e-6);
    }

    static double highrestime_diff(highrestimer_t start, highrestimer_t end) {
        return end - start;
    }

#else

    #include <time.h>

    typedef time_t highrestimer_t;

    static highrestimer_t get_highrestime(void) {
        return time(NULL);
    }

    static double highrestime_diff(highrestimer_t start, highrestimer_t end) {
        return difftime(end, start);
    }

#endif

#pragma GCC diagnostic pop

#endif
