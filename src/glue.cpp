/* (C)2022 Simon Urbanek <simon.urbanek@R-project.org>

   License: choice of
     - GPL-2
     - MIT
*/

#include <cstdio>
#include <iostream>
#include <numeric>
#include <string>
#include "digraph.hpp"

#include <Rinternals.h>

/* used by digraph.hpp */
int opt_verbose = 0;

extern "C" SEXP C_lecount(SEXP sMatrix, SEXP sN) {
    bool *matrix;
    LECOptions options;
    size_t count;
    size_t n = (size_t) Rf_asReal(sN), N = n * n, i = 0;
    int *v;
    /* our R wrapper actually guarantees this, but just in case .. */
    if (XLENGTH(sMatrix) != N || TYPEOF(sMatrix) != INTSXP)
	Rf_error("Invalid matrix, must be a square integer matrix.");
    v = INTEGER(sMatrix);
    
    /* FIXME: pass options from R */
    
    /* copy R integer representation to a bool array */
    matrix = new bool[N];
    while (i < N) {
	matrix[i] = v[i] ? true : false;
	i++;
    }

    LinearExtensionCounterAuto<size_t> lec(n);
    lec.options = options;
    count = lec.count(matrix);

    delete [] matrix;
    return
	(count < (1 << 31)) ?
	Rf_ScalarInteger((int) count) :
	Rf_ScalarReal((double) count);
}
