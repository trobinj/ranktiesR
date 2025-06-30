#ifndef hclust_h
#define hclust_h

#include <RcppArmadillo.h>
#include "ranktiesR_types.h"
#include "misc.h"
#include <cmath>

double cdist(dvec a, dvec b, std::string linkage)
{
  int na = a.n_elem;
  int nb = b.n_elem;
  dmat d(na, nb);

  for (int i = 0; i < na; ++i) {
    for (int j = 0; j < nb; ++j) {
      d(i,j) = std::abs(a(i) - b(j));
    }
  }

  if (linkage == "single") {
    return min(vectorise(d));
  }

  if (linkage == "complete") {
    return max(vectorise(d));
  }

  if (linkage == "average") {
    return mean(vectorise(d));
  }

  Rcpp::stop("linkage not recognized");
}

dvec cfind(std::vector<uvec>& g, dvec u, double& dist, std::string linkage)
{
  int n = g.size();
  double a, b, d, y = 1000;

  for (int i = 1; i < n; ++i) {
    for (int j = 0; j < i; ++j) {
      d = cdist(u(g[i]), u(g[j]), linkage);
      if (d < y) {
        y = d;
        a = i;
        b = j;
      }
    }
  }

  dist = y;
  dvec out = {a, b};

  return out;
}

uvec hclust(dvec u, double delta, std::string linkage)
{
  int k = u.n_elem;
  uvec x(1);
  uvec r = rankvec(-u);
  u = sort(u);

  std::vector<uvec> g;
  g.reserve(k);
  for (int i = 0; i < k; ++i) {
    x.fill(i);
    g.emplace_back(x);
  }

  double d;
  dvec tmp;
  uvec out;

  for (int i = 0; i < k - 1; ++i) {
    tmp = cfind(g, u, d, linkage);
    if (d > delta) {
      if (i == 0) {
        return r;
      } else {
        out = g[0];
        for (int j = 0; j < k - i; ++j) {
          g[j].fill(k - i - j);
          out = join_vert(g[j], out);
        }
        return out(r - 1);
      }
    }
    g[min(tmp)] = join_vert(g[tmp(0)], g[tmp(1)]);
    g.erase(g.begin() + max(tmp));
  }

  out.set_size(k);
  out.fill(1);
  return out;
}

uvec hcl_min(dvec u, double delta)
{
  return hclust(u, delta, "single");
}

uvec hcl_max(dvec u, double delta)
{
  return hclust(u, delta, "complete");
}

uvec hcl_avg(dvec u, double delta)
{
  return hclust(u, delta, "average");
}

#endif
