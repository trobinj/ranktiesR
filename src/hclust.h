#ifndef hclust_h
#define hclust_h

#include <RcppArmadillo.h>
#include "ranktiesR_types.h"
#include "misc.h"
#include <cmath>

double cdist(dvec a, dvec b) // complete-linkage
{
  int na = a.n_elem;
  int nb = b.n_elem;
  double d, y = 0.0;

  for (int i = 0; i < na; ++i) {
    for (int j = 0; j < nb; ++j) {
      d = std::abs(a(i) - b(j));
      if (d > y) {
        y = d;
      }
    }
  }

  return y;
}

dvec cfind(std::vector<uvec> g, dvec u, double& dist)
{
  int n = g.size();
  double a, b, d, y = 100;

  for (int i = 1; i < n; ++i) {
    for (int j = 0; j < i; ++j) {
      d = cdist(u(g[i]), u(g[j]));
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

uvec hclust(dvec u, double delta)
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
    tmp = cfind(g, u, d);
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

#endif
