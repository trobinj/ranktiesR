#ifndef kmeans1d_h
#define kmeans1d_h

#include <RcppArmadillo.h>
#include "ranktiesR_types.h"
#include "misc.h"

uvec repeatvec(uvec n)
{
  uvec y(accu(n));
  int m = n.n_elem;
  int t = 0;
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n(i); ++j) {
      y(t) = i;
      ++t;
    }
  }
  return y;
}

umat kmeans1dpart(int k, int g)
{
  int n = pow(k, g);
  int s, t, z;
  umat x(n, g);

  for (int j = 1; j <= g; ++j) {
    z = pow(k, j);
    s = 1;
    t = 1;
    for (int i = 1; i <= n; ++i) {
      if (z * (s + 1) > n) {
        s = 1;
        x(i - 1, j - 1) = t;
        if (t + 1 > k) {
          t = 1;
        } else {
          t = t + 1;
        }
      } else {
        x(i - 1, j - 1) = t;
        s = s + 1;
      }
    }
  }

  uvec indx = find(sum(x, 1) == k);
  x = x.rows(indx);

  umat y(x.n_rows, k);
  for (int i = 0; i < x.n_rows; ++i) {
    y.row(i) = repeatvec(x.row(i).t()).t();
  }

  return y;
}

std::vector<umat> kmeans1dpartvec(int k)
{
  std::vector<umat> y;
  y.reserve(k - 2);
  for (int i = 0; i < k - 2; ++i) {
    y.emplace_back(kmeans1dpart(k, i + 2));
  }
  return y;
}

struct kmeansolution
{
  dvec wss;
  uvec groups;
};

kmeansolution kmeansolve(dvec u, umat groups)
{
  uvec r = rankvec(u);
  u = sort(u);

  int g = groups.max() + 1;
  int n = groups.n_rows;
  dmat wss(n, g);
  uvec idx;
  double mij;
  dvec uij;

  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < g; ++j) {
      idx = find(groups.row(i) == j);
      mij = mean(u(idx));
      uij = u(idx);
      wss(i, j) = wss(i, j) + sum(pow(uij - mij, 2));
    }
  }

  dvec wsstotal = sum(wss, 1);
  int imin = wsstotal.index_min();

  kmeansolution out {
    wss.row(imin).t(),
    groups.row(imin).t()
  };
  out.groups =  out.groups(r - 1);
  out.groups = -out.groups + max(out.groups) + 1;
  return out;
}

uvec kmean1d(dvec u, double delta, std::vector<umat>& ysets)
{
  int k = u.n_elem;

  kmeansolution out;
  uvec indx;
  dvec vwg;

  if (var(u,1)*k < delta) {
    uvec y(k, arma::fill::ones);
    return y;
  }
  for (int g = 2; g < k; ++g) {
    vwg.set_size(g);
    vwg.fill(0.0);
    out = kmeansolve(u, ysets[g - 2]);
    if (sum(out.wss) < delta) {
      return out.groups;
    }
  }
  return rankvec(-u);
}

double withinss(uvec y, dvec u)
{
  int kmax = max(y);
  dvec uj;
  int nj;
  double wws = 0.0;

  for (int j = 0; j < kmax; ++j) {
    uj = u(find(y == j + 1));
    nj = uj.n_elem;
    wws = wws + nj * var(uj, 1);
  }

  return wws;
}

// [[Rcpp::export]]
dvec kstart(uvec y, double delta)
{
  using namespace arma;

  int k = y.n_elem;
  int kmax = max(y);

  std::vector<umat> ysets;
  ysets = kmeans1dpartvec(k);

  dvec u(k);
  uvec jset;
  for (int j = 0; j < kmax; ++j) {
    jset = find(y == j + 1);
    u(jset) = randu(jset.n_elem, distr_param(-0.25, 0.25)) + kmax - j;
  }

  double wss;
  kmeansolution tmp;
  if (k == kmax) {
    tmp = kmeansolve(u, ysets[kmax - 3]);
    wss = sum(tmp.wss);
    u = u/sqrt(wss) * sqrt(delta) * 1.01;
  } else {
    wss = withinss(y, u);
    u = u/sqrt(wss) * sqrt(delta) * 0.99;
  }

  u = u - u(k - 1);
  return u.head(k - 1);
}

#endif
