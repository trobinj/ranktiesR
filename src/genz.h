#ifndef genz_h
#define genz_h

#include <RcppArmadillo.h>
#include "ranktiesR_types.h"
#include "misc.h"

/* The pmvnorm function uses a Monte Carlo algorithm due to Genz (1992, Journal of
 * Computational and Graphical Statistics). I added a parameter nmin for the
 * minimum number of iterations before checking the estimated Monte Carlo error.
 */

namespace pmvnormspc {
  
  double pnorm(double z) 
  {
    return R::pnorm(z, 0.0, 1.0, true, false);
  }
  double qnorm(double p) 
  {
    return R::qnorm(p, 0.0, 1.0, true, false);
  }
}

// [[Rcpp::export]]
double genz(arma::vec a, arma::vec b, arma::vec mu, arma::mat sigma, double epsilon, double alpha, int nmin, int nmax) {

  using namespace pmvnormspc;

  arma::mat c = chol(sigma, "lower");
  a = a - mu;
  b = b - mu;

  double intsum = 0.0;
  double varsum = 0.0;
  double interr = 0.0;
  double cy;
  int m = a.n_elem;
  int N;

  arma::vec y(m - 1);
  arma::vec w(m - 1);
  arma::vec d(m); d(0) = pnorm(a(0) / c(0,0));
  arma::vec e(m); e(0) = pnorm(b(0) / c(0,0));
  arma::vec f(m); f(0) = e(0) - d(0);

  for (int k = 0; k < nmax; ++k) {
    w.randu();
    for (int i = 1; i < m; ++i) {
      y(i - 1) = qnorm(d(i - 1) + w(i - 1) * (e(i - 1) - d(i - 1)));
      cy = 0.0;
      for (int j = 0; j < i; ++j) {
        cy = cy + c(i, j) * y(j);
      }
      d(i) = pnorm((a(i) - cy) / c(i, i));
      e(i) = pnorm((b(i) - cy) / c(i, i));
      f(i) = (e(i) - d(i)) * f(i - 1);
    }
    N = k + 1;
    intsum = intsum + f(m - 1);
    varsum = varsum + pow(f(m - 1), 2);

    if (N > nmin) {
      interr = alpha * sqrt((varsum / N - pow(intsum / N, 2)) / N);
      if (interr < epsilon) {
        break;
      }
    }
  }

  return intsum / N;
}

#endif