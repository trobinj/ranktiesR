#include <RcppArmadillo.h>
#include "ranktiesR_types.h"
#include "misc.h"
#include "poonxu.h"
#include "kmeans1d.h"
#include "hclust.h"
#include "sep.h"
#include "seqcluster.h"

// [[Rcpp::depends(RcppArmadillo)]]

using namespace arma;

typedef uvec (*clfunc)(dvec, double);

template <typename type>
static void dump(type x)
{
	Rcpp::Rcout << x << "\n";
}

template <typename type>
static void dump(type x, std::string y)
{
	Rcpp::Rcout << y << "\n" << x << "\n";
}

class rankvector
{
private:

  uvec y;
  dvec x;
  dmat z;
  int k, r, m, t;
  std::vector<uvec> lower;
  std::vector<uvec> upper;
  const double ninf = arma::datum::nan;
  const double pinf = arma::datum::nan;

public:

  static std::vector<umat> ysets;

  rankvector(uvec y, dvec x) : y(y), x(x)
  {
    k = y.n_elem;
    r = max(y);
    m = 1;
    t = 5*k;

    z.set_size(m, k - 1);

    dvec u(k);
    dvec v(r, fill::randn);
    v = sort(v, "descend");
    for (int j = 0; j < r; ++j) {
      u(find(y == j + 1)).fill(v(j));
    }
    z.row(0) = u.head(k - 1).t() - u(k - 1);

    lower.reserve(k);
    upper.reserve(k);
    for (int i = 0; i < k; ++i) {
      lower.emplace_back(find(y == y(i) + 1));
      upper.emplace_back(find(y == y(i) - 1));
    }
  }

  void setsize(int m, int t)
  {
    this->m = m;
    this->t = t;
    z.resize(m, k - 1);
  }

  dvec zsamp(rvec& z, cnorm& zdist)
  {
    dvec u(k);
    u.head(k - 1) = z.t();
    double mj, sj, aj, bj;
    for (int i = 0; i < t; ++i) {
      for (int j = 0; j < k - 1; ++j) {
        mj = zdist.getm(j, u.head(k - 1));
        sj = zdist.gets(j);
        aj = lower[j].is_empty() ? ninf : max(u(lower[j]));
        bj = upper[j].is_empty() ? pinf : min(u(upper[j]));
        u(j) = rtnorm(mj, sj, aj, bj);
      }
    }
    return u.head(k - 1);
  }

  void estep_ran(cnorm& zdist)
  {
    rvec z0 = z.row(0);
    for (int j = 0; j < m; ++j) {
      z.row(j) = zsamp(z0, zdist).t();
    }
  }

  void estep_tie(cnorm& zdist, double delta, std::string type, int nmatch)
  {
    clfunc cluster;
    if (type == "poon_xu") cluster = poon_xu;
    if (type == "seq_min") cluster = seq_min;
    if (type == "seq_max") cluster = seq_max;
    if (type == "seq_avg") cluster = seq_avg;

    dvec utmp(k);
    uvec ytmp(k);
    rvec z0 = z.row(0);

    int cnt;
    for (int j = 0; j < m; ++j) {
      cnt = 0;
      do {
        cnt++;
        utmp.head(k - 1) = zsamp(z0, zdist);
        ytmp = cluster(utmp, delta);
      } while (!vecmatch(y, ytmp, nmatch));
      Rcpp::Rcout << cnt << "\n";
      z.row(j) = utmp.head(k - 1).t();
    }
  }

  void estep_k1d(cnorm& zdist, double delta, int nmatch)
  {
    dvec utmp(k);
    uvec ytmp(k);
    rvec z0 = z.row(0);
  
    int cnt;
    for (int j = 0; j < m; ++j) {
      cnt = 0;
      do {
        cnt++;
        utmp.head(k - 1) = zsamp(z0, zdist);
        ytmp = kmeans1d(utmp, delta, ysets);
        if (cnt > 50000) {
          Rcpp::Rcout << "break" << "\n";
          break;
        }
      } while (!vecmatch(y, ytmp, nmatch));
      z.row(j) = utmp.head(k - 1).t();
    }
  }

  dmat getzx()
  {
    return mean(z, 0).t() * x.t();
  }

  dmat getzz()
  {
    return (z.t() * z) / m;
  }
};

std::vector<umat> rankvector::ysets;

class rankdata
{
private:

  int n, p, q, k;
  umat y;
  dmat x;
  dmat beta;
  dmat sigm;
  dmat xx, xxinv;
  std::vector<rankvector> data;
  std::vector<umat> ysets;

public:

  rankdata(umat y, dmat x) : y(y), x(x)
  {
    n = y.n_rows;
    p = x.n_cols;
    k = y.n_cols;
    q = k * (k - 1) / 2;

    beta.zeros(k - 1, p);
    sigm.eye(k - 1, k - 1);

    xx = x.t() * x;
    xxinv = inv(xx);

    ysets = kmeans1dpartvec(k);
    rankvector::ysets = ysets;

    data.reserve(n);
    for (int i = 0; i < n; ++i) {
      data.emplace_back(y.row(i).t(), x.row(i).t());
    }
  }

  dvec getparameters()
  {
    dvec theta((k - 1) * p + q);
    theta.head((k - 1) * p) = vectorise(beta);
    theta.tail(q) = vectorise(lowertri(sigm));
    return theta;
  }

  void getparameters(dmat& bhat, dmat& shat)
  {
    bhat = beta;
    shat = sigm;
  }

  void setparameters(dvec theta)
  {
    dmat temp = theta.head((k-1)*p);
    beta = temp.reshape(size(beta));
    sigm = vec2lower(theta.tail(q), true);
  }

  void setsize(int m, int t)
  {
    for (auto& obs : data) {
      obs.setsize(m, t);
    }
  }

  double loglik(double delta, std::string type, int r)
  {
    assert(type != "ran");

    dmat eta = beta * x.t();
    dmat Q = arma::chol(sigm, "lower");

    uvec yobs;
    uvec ytmp(k);

    dvec utmp(k);
    dvec etai(k - 1);

    int succ, fail;
    double l = 0.0;

    clfunc cluster;
    if (type == "poon_xu") cluster = poon_xu;
    if (type == "seq_min") cluster = seq_min;
    if (type == "seq_max") cluster = seq_max;
    if (type == "seq_avg") cluster = seq_avg;

    for (int i = 0; i < n; ++i) {
      yobs = y.row(i).t();
      etai = eta.col(i);
      succ = 0;
      fail = 0;

      Rcpp::Rcout << i + 1 << "\n";

      if (type == "k1d") {
        do {
          utmp.head(k - 1) = mvrnorm(etai, Q, true);
          ytmp = kmeans1d(utmp, delta, ysets);
          if (all(ytmp == yobs)) {
            ++succ;
          } else {
            ++fail;
          }
        } while (succ < r);
      } else {
        do {
          utmp.head(k - 1) = mvrnorm(etai, Q, true);
          ytmp = cluster(utmp, delta);
          if (all(ytmp == yobs)) {
            ++succ;
          } else {
            ++fail;
          }
        } while (succ < r);
      }

      l = l + log(succ - 1) - log(succ + fail - 1);
    }
    return l;
  }

  void estep(double delta, std::string type, int nmatch)
  {
    cnorm zdist(sigm);
    dmat eta = beta * x.t();

    if (type == "ran") {
      for (int i = 0; i < n; ++i) {
        zdist.setm(eta.col(i));
        data[i].estep_ran(zdist);
      }
      return;
    }

    if (type == "k1d") {
      for (int i = 0; i < n; ++i) {
        zdist.setm(eta.col(i));
        data[i].estep_k1d(zdist, delta, nmatch);
      }
      return;
    }

    for (int i = 0; i < n; ++i) {
      zdist.setm(eta.col(i));
      data[i].estep_tie(zdist, delta, type, nmatch);
    }
  }

  void mstep()
  {
    dmat ezx(k - 1, p);
    dmat ezz(k - 1, k - 1);

    for (int i = 0; i < n; ++i) {
      ezx = ezx + data[i].getzx();
      ezz = ezz + data[i].getzz();
    }

    beta = ezx * xxinv;
    sigm = (ezz - beta * xx * beta.t()) / n;
  }

  void xstep()
  {
    double d = 1.0 / sigm(0,0);
    sigm = sigm * d;
    beta = beta * sqrt(d);
  }
};

// [[Rcpp::export]]
Rcpp::List rankties(umat y, dmat x, uvec n, uvec m, int t,
  double delta, std::string type, bool print, int v)
{
  if (std::set<std::string>{"ran","pxu","seq_min","seq_max","seq_avg"}.count(type) == 0) {
    Rcpp::stop("unknown type");
  }

  int k = y.n_cols;
  int p = x.n_cols;
  int q = k * (k - 1) / 2;

  int m0 = m(0);
  int mf = m(1);
  int n0 = n(0);
  int nf = n(1);

  rankdata data(y, x);

  data.setsize(m0, t);

  dmat out(n0 + nf + 1, (k - 1)*p + q);
  dmat bhat, shat;

  out.row(0) = data.getparameters().t();

  for (int i = 0; i < n0 + nf; ++i) {

    if (i == n0) {
      data.setsize(mf, t);
    }

    data.estep(delta, type, std::min(i+1,k));
    data.mstep();
    if (type == "ran") data.xstep();

    out.row(i + 1) = data.getparameters().t();

    if ((i + 1) % 1 == 0 && print) {
      dump<int>(i + 1, "iteration: ");

      data.getparameters(bhat, shat);
      dump<dmat>(bhat, "beta: \n");
      dump<dmat>(shat, "sigm: \n");
    }

    Rcpp::checkUserInterrupt();
  }

  dvec theta = mean(out.tail_rows(nf/2), 0).t();
  data.setparameters(theta);

  double ll = arma::datum::nan;
  if (v > 0) {
    ll = data.loglik(delta, type, v);
  }

  return Rcpp::List::create(
    Rcpp::Named("output") = Rcpp::wrap(out),
    Rcpp::Named("loglik") = Rcpp::wrap(ll)
  );
}

