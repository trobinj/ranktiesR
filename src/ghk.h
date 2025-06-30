#ifndef ghk_h
#define ghk_h

#include <RcppArmadillo.h>
#include "ranktiesR_types.h"
#include "misc.h"

namespace ghkspc
{
  double pnorm(double z) {
    return R::pnorm(z, 0.0, 1.0, true, false);
  }
  double tnorm(double a, double b) {
    return rnormint(0.0, 1.0, a, b);
  }
}

double ghk(arma::vec m, arma::mat s, arma::vec low, arma::vec upp, int n)
{
  using namespace ghkspc;

  int d = m.n_elem;
  arma::vec lw = low - m;
  arma::vec up = upp - m;
  arma::vec q(d, arma::fill::zeros);
  arma::vec z(d - 1, arma::fill::zeros);
  arma::mat C = arma::chol(s, "lower");
  double v, prb = 0.0;

  arma::vec l(d);
  arma::vec u(d);

  l(0) = lw(0) / C(0,0);
  u(0) = up(0) / C(0,0);
  q(0) = pnorm(u(0)) - pnorm(l(0));

  for (int i = 0; i < n; ++i) {
    z(0) = tnorm(l(0), u(0));
    for (int j = 1; j < d; ++j) {
      v = as_scalar(C(j, arma::span(0,j-1)) * z.head(j));
      l(j) = (lw(j) - v) / C(j,j);
      u(j) = (up(j) - v) / C(j,j);
      q(j) = pnorm(u(j)) - pnorm(l(j));
      if (j < d - 1) {
        z(j) = tnorm(l(j), u(j));
      }
    }
    prb = prb + prod(q);
  }

  return prb / n;
}

#endif