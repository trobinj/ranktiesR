#ifndef seqcluster_h
#define seqcluster_h

#include <RcppArmadillo.h>
#include "ranktiesR_types.h"
#include "misc.h"

uvec seq_min(dvec u, double delta)
{
  int n = u.n_elem;

  uvec r = rankvec(u);
  u = sort(u);

  uvec y(n);
  y.fill(1);

  int a = n - 1;

  for (int i = n - 2; i >= 0; --i) {
    if (min(u.subvec(i + 1, a)) - u(i) > delta) {
      y(i) = y(i + 1) + 1;
      a = i;
    } else {
      y(i) = y(i + 1);
    }
   }

  return y(r - 1);
}

uvec seq_max(dvec u, double delta)
{
  int n = u.n_elem;

  uvec r = rankvec(u);
  u = sort(u);

  uvec y(n);
  y.fill(1);

  int a = n - 1;

  for (int i = n - 2; i >= 0; --i) {
    if (max(u.subvec(i + 1, a)) - u(i) > delta) {
      y(i) = y(i + 1) + 1;
      a = i;
    } else {
      y(i) = y(i + 1);
    }
   }

  return y(r - 1);
}

uvec seq_avg(dvec u, double delta)
{
  int n = u.n_elem;

  uvec r = rankvec(u);
  u = sort(u);

  uvec y(n);
  y.fill(1);

  int a = n - 1;

  for (int i = n - 2; i >= 0; --i) {
    if (mean(u.subvec(i + 1, a)) - u(i) > delta) {
      y(i) = y(i + 1) + 1;
      a = i;
    } else {
      y(i) = y(i + 1);
    }
   }

  return y(r - 1);
}

#endif