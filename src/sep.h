#ifndef sep_h
#define sep_h

#include <RcppArmadillo.h>
#include "ranktiesR_types.h"
#include "misc.h"

// [[Rcpp::depends(RcppArmadillo)]]

uvec sep(dvec u, double delta)
{
  int n = u.n_elem;

  uvec r = rankvec(u);
  u = sort(u);

  uvec y(n);
  y.fill(1);

  for (int i = n - 1; i > 0; --i) {
    if (u(i) - u(i-1) > delta) {
      y(i-1) = y(i) + 1;
    } else {
      y(i-1) = y(i);
    }
  }                                                               
  
  return y(r - 1);
}

#endif
