#include <RcppArmadillo.h>
#include "ranktiesR_types.h"
#include "misc.h"
#include "seqcluster.h"
#include "hclust.h"
#include "poon_xu.h"
#include "genz.h"
#include "kmeans1d.h"

// Maybe remove scale setter.

// [[Rcpp::depends(RcppArmadillo)]]

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

std::vector<umat> ysets;

class ranktiesvector
{
private:

  uvec y;
  dvec x;
  dmat zfull;
  dmat zthin;
  int k, m, t;
  std::string type;
  clfunc cluster;
  double delta, scale;

public:

  ranktiesvector(uvec y, dvec x, std::string type, double delta) : y(y), x(x), type(type), delta(delta)
  {
    if (type == "seq_min") cluster = seq_min;
    if (type == "seq_max") cluster = seq_max;
    if (type == "seq_avg") cluster = seq_avg;
    if (type == "poon_xu") cluster = poon_xu;
    if (type == "hcl_min") cluster = hcl_min;
    if (type == "hcl_max") cluster = hcl_max;
    if (type == "hcl_avg") cluster = hcl_avg;

    k = y.n_elem;
    m = 5;
    t = 100;
    scale = 0.1;

    zfull.set_size(t * (m + 1), k - 1);
    zthin.set_size(m, k - 1);

    if (type == "kmean1d") {
      zfull.row(0) = kstart(y, delta).t();
    } else {
      zfull.row(0) = zstart(y, delta).t();
    }
  }

  void setscale(double scale) {
    this->scale = scale;
  }

  void setsize(int m, int t)
  {
    dvec z0 = zfull.row(0).t();

    this->m = m;
    this->t = t;

    zthin.resize(m, k-1);
    zfull.resize(t*(m+1), k-1);
    zfull.row(0) = z0.t();
  }

  void estep(dvec mu, dmat sigma, double& rate)
  {
    using namespace arma;

    dmat Q = arma::chol(sigma, "lower");
    dmat R = sqrt(scale)*eye(k-1,k-1);

    mvndist dist(mu, sigma);

    uvec ynew(k);
    dvec uold(k-1);
    dvec unew(k);
    double lprb;
    double cnt = 0;

    for (int i = 1; i < t*(m + 1); ++i) {
      uold = zfull.row(i-1).t();
      unew.head(k-1) = uold + randn<dvec>(k-1, distr_param(0.0, scale));
      if (type == "kmean1d") {
        ynew = kmean1d(unew, delta, ysets);
      } else {
        ynew = cluster(unew, delta);
      }
      if (all(ynew == y)) {
        lprb = std::min(0.0, dist.logpdf(unew.head(k-1)) - dist.logpdf(uold));
        if (R::runif(0.0, 1.0) < exp(lprb)) {
          zfull.row(i) = unew.head(k-1).t();
          cnt++;
        } else {
          zfull.row(i) = uold.t();
        }
      } else {
        zfull.row(i) = uold.t();
      }
    }

    rate = cnt / (t * m);
    if (rate > 0.15) {
      scale = scale * 2.0;
    }
    if (rate < 0.05) {
      scale = scale / 2.0;
    }

    zfull.row(0) = zfull.row(t*(m+1)-1);
    zthin = zfull.rows(linspace<uvec>(t, t*(m+1)-1, m));
  }

  dmat getzx()
  {
    return mean(zthin, 0).t() * x.t();
  }

  dmat getzz()
  {
    return (zthin.t() * zthin) / m;
  }

  dvec gradient(dmat beta, dmat sigm)
  {
    using namespace arma;

    dmat R = inv(sigm);
    dmat I = eye(k-1, k-1);
    dvec e(k-1);
    dmat betag(size(beta));
    dmat sigmg(size(sigm));
    dvec sigmv;

    betag.fill(0.0);
    sigmg.fill(0.0);

    for (int j = 0; j < m; ++j) {
      e = (zthin.row(j).t() - beta * x);
      betag = betag + R * e * x.t();
      sigmg = sigmg - 0.5 * (2 * R - (R % I) -
        2 * R * e * e.t() * R + ((R * e * e.t() * R) % I));
    }

    betag = betag / m;
    sigmg = sigmg / m;

    return join_vert(vectorise(betag), vectorise(lowertri(sigmg)));
  }
};

class ranktiesdata
{
private:

  int n, p, q, k;
  umat y;
  dmat x;
  dmat beta;
  dmat sigm;
  dmat xx, xxinv;
  std::vector<ranktiesvector> data;
  std::string type;
  double delta;

public:

  ranktiesdata(umat y, dmat x, std::string type, double delta) : y(y), x(x), type(type), delta(delta)
  {
    n = y.n_rows;
    p = x.n_cols;
    k = y.n_cols;
    q = k * (k - 1) / 2;

    beta.zeros(k - 1, p);
    sigm.eye(k - 1, k - 1);

    xx = x.t() * x;
    xxinv = inv(xx);

    data.reserve(n);
    for (int i = 0; i < n; ++i) {
      data.emplace_back(y.row(i).t(), x.row(i).t(), type, delta);
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
    dmat temp = theta.head((k - 1) * p);
    beta = temp.reshape(size(beta));
    sigm = vec2lower(theta.tail(q), true);
  }

  void setscale(double scale)
  {
    for (auto& obs : data) {
      obs.setscale(scale);
    }
  }

  void setsize(int m, int t)
  {
    for (auto& obs : data) {
      obs.setsize(m, t);
    }
  }

  void estep()
  {
    dmat eta = beta * x.t();

    dvec rate(n);
    for (int i = 0; i < n; ++i) {
      data[i].estep(eta.col(i), sigm, rate(i));
    }

    dvec p(5);
    dvec prob = {0.25, 0.5, 0.75};

    p(0) = min(rate);
    p.subvec(1,3) = arma::quantile(rate, prob);
    p(4) = max(rate);
    dump<rvec>(p.t(), "acceptance rate:");
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

  dmat vmat()
  {
    dmat score(n, (k - 1) * p + q);
    for (int i = 0; i < n; ++i) {
      score.row(i) = data[i].gradient(beta, sigm).t();
    }
    return inv(score.t() * score);
  }
};

// [[Rcpp::export]]
Rcpp::List ranktiesmodel(umat y, dmat x, uvec n, uvec m, int t, double delta, double scale, std::string type, int h, bool print)
{
  using namespace Rcpp;

  int k = y.n_cols;
  int p = x.n_cols;
  int q = k * (k - 1)/2;

  int m0 = m(0);
  int mf = m(1);
  int n0 = n(0);
  int nf = n(1);

  ysets = kmeans1dpartvec(k);

  ranktiesdata data(y, x, type, delta);

  data.setscale(scale);
  data.setsize(m0, t);

  dmat out(n0 + nf + 1, (k-1)*p + q);
  dmat bhat, shat;

  out.row(0) = data.getparameters().t();

  for (int i = 0; i < n0 + nf; ++i) {

    if (i == n0) {
      data.setsize(mf, t);
    }

    data.estep();
    data.mstep();

    out.row(i + 1) = data.getparameters().t();

    if ((i + 1) % 1 == 0 && print) {
      data.getparameters(bhat, shat);
      dump<int>(i+1, "iterations: \n");
      dump<dmat>(bhat, "beta: \n");
      dump<dmat>(shat, "sigm: \n");
    }

    Rcpp::checkUserInterrupt();
  }

  dvec theta = mean(out.tail_rows(nf), 0).t();
  dmat vcov;
  if (h) {
    data.setparameters(theta);
    data.setsize(h, t);
    data.estep();
    vcov = data.vmat();
  } else {
    vcov.fill(arma::datum::nan);
  }

  return List::create(
    Named("out") = wrap(out),
    Named("theta") = wrap(theta),
    Named("vcov") = wrap(vcov)
  );
}

// [[Rcpp::export]]
double ranktiesloglik(umat y, dvec x, dvec w, dvec theta, std::string type, double delta, int b, int m)
{
  clfunc cluster;
  if (type == "seq_min") cluster = seq_min;
  if (type == "seq_max") cluster = seq_max;
  if (type == "seq_avg") cluster = seq_avg;
  if (type == "poon_xu") cluster = poon_xu;
  if (type == "hcl_min") cluster = hcl_min;
  if (type == "hcl_max") cluster = hcl_max;
  if (type == "hcl_avg") cluster = hcl_avg;

  int n = y.n_rows;
  int k = y.n_cols;
  int p = x.n_elem;
  int q = k * (k - 1) / 2;

  umat R = allpermutations(k);

  uvec indx_all(R.n_rows);
  uvec indx_set;
  std::vector<uvec> indx_vec;
  indx_vec.reserve(n);

  for (int i = 0; i < n; ++i) {
    indx_set = rankset(y.row(i).t(), R);
    indx_vec.emplace_back(indx_set);
    indx_all(indx_set).fill(1);
  }

  dmat beta(k-1,p);
  dmat sigm(k-1,k-1);
  dmat temp = theta.head((k-1)*p);
  beta = temp.reshape(size(beta));
  sigm = vec2lower(theta.tail(q), true);

  dvec mu = beta * x;
  dmat D;
  dvec lower(k - 1);
  dvec upper(k - 1);

  lower.fill(0.0);
  upper.fill(100);

  dmat Q = arma::chol(sigm, "lower");

  dvec prob_all(R.n_rows);
  prob_all.fill(0.0);
  for (int i = 0; i < R.n_rows; ++i) {
    if (indx_all(i)) {
      D = rankmat(R.row(i).t());
      D.shed_col(k - 1);
      prob_all(i) = genz(lower, upper, mu, Q, 0.00001, 2.5, 100, 10000);
    }
  }

  dvec prob_set(n);
  dvec prob;
  for (int i = 0; i < n; ++i) {
    prob = prob_all.rows(indx_vec[i]);
    prob_set(i) = sum(prob);
    if (prob_set(i) == 0) {
      Rcpp::stop("zero probability estimate");
    }
  }

  dmat u;
  uvec yi;
  uvec ytmp;
  dvec incl(m);
  double loglik = 0.0;
  cnorm zdist(mu, sigm);

  dvec sumprob(n);
  for (int i = 0; i < n; ++i) {
    yi = y.row(i).t();
    u = zsamp(yi, zdist, b, m);
    for (int j = 0; j < m; ++j) {
      if (type == "kmean1d") {
        ytmp = kmean1d(u.row(j).t(), delta, ysets);
      } else {
        ytmp = cluster(u.row(j).t(), delta);
      }
      incl(j) = all(yi == ytmp) ? prob_set(i) : 0.0;
    }
    sumprob(i) = arma::sum(incl);
    // loglik = loglik + w(i) * (log(arma::sum(incl)) - log(m));
  }

  double minprob = min(sumprob(find(sumprob > 0.0)));
  for (int i = 0; i < n; ++i) {
    if (sumprob(i) == 0) {
      sumprob(i) = minprob;
    }
    loglik = loglik + w(i) * (log(sumprob(i)) - log(m));
  }

  return loglik;
}

// [[Rcpp::export]]
umat rankresiduals(umat y, dvec x, dvec theta, std::string type, double delta, int n)
{
  using namespace arma;
  using namespace Rcpp;

  int p = x.n_elem;
  int k = y.n_cols;
  int q = (k - 1) * k / 2;

  clfunc cluster;
  if (type == "seq_min") cluster = seq_min;
  if (type == "seq_max") cluster = seq_max;
  if (type == "seq_avg") cluster = seq_avg;
  if (type == "poon_xu") cluster = poon_xu;
  if (type == "hcl_min") cluster = hcl_min;
  if (type == "hcl_max") cluster = hcl_max;
  if (type == "hcl_avg") cluster = hcl_avg;

  dmat beta(k - 1, p);
  dmat sigm(k - 1, k - 1);

  dmat temp = theta.head((k-1)*p);
  beta = temp.reshape(size(beta));
  sigm = vec2lower(theta.tail(q), true);

  dvec mu = beta * x;
  dmat Q = arma::chol(sigm, "lower");

  umat v(n, k);
  dvec utmp(k);

  for (int i = 0; i < n; ++i) {
    utmp.head(k-1) = mvrnorm(mu, Q, true);
    if (type == "kmean1d") {
      v.row(i) = kmean1d(utmp, delta, ysets).t();
    } else {
      v.row(i) = cluster(utmp, delta).t();
    }
  }

  return v;
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

  rankvector(uvec y, dvec x) : y(y), x(x)
  {
    k = y.n_elem;
    r = max(y);
    m = 1;
    t = 5*k;

    z.set_size(m, k - 1);

    dvec u(k);
    dvec v(r, arma::fill::randn);
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

  void estep(cnorm& zdist)
  {
    rvec z0 = z.row(0);
    for (int j = 0; j < m; ++j) {
      z.row(j) = zsamp(z0, zdist).t();
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

  dvec gradient(dmat beta, dmat sigm)
  {
    using namespace arma;

    dmat R = inv(sigm);
    dmat I = eye(k-1, k-1);
    dvec e(k-1);
    dmat betag(size(beta));
    dmat sigmg(size(sigm));
    dvec sigmv;

    betag.fill(0.0);
    sigmg.fill(0.0);

    for (int j = 0; j < m; ++j) {
      e = (z.row(j).t() - beta * x);
      betag = betag + R * e * x.t();
      sigmg = sigmg - 0.5 * (2 * R - (R % I) - 2 * R * e * e.t() * R + ((R * e * e.t() * R) % I));
    }

    betag = betag / m;
    sigmv = lowertri(sigmg);
    sigmv = sigmv.tail(k * (k - 1) / 2 - 1);

    return join_vert(vectorise(betag), sigmv);
  }
};

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

  void estep()
  {
    cnorm zdist(sigm);
    dmat eta = beta * x.t();

    for (int i = 0; i < n; ++i) {
      zdist.setm(eta.col(i));
      data[i].estep(zdist);
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

  dmat vmat()
  {
    dmat score(n, (k - 1) * p + q - 1);
    for (int i = 0; i < n; ++i) {
      score.row(i) = data[i].gradient(beta, sigm).t();
    }
    return inv(score.t() * score);
  }
};

// [[Rcpp::export]]
Rcpp::List rankmodel(umat y, dmat x, uvec n, uvec m, int t, int h, bool print)
{
  using namespace Rcpp;

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

    data.estep();
    data.mstep();
    data.xstep();

    out.row(i + 1) = data.getparameters().t();

    if ((i + 1) % 1 == 0 && print) {
      data.getparameters(bhat, shat);
      dump<int>(i+1, "iterations: \n");
      dump<dmat>(bhat, "beta: \n");
      dump<dmat>(shat, "sigm: \n");
    }

    Rcpp::checkUserInterrupt();
  }

  dvec theta = mean(out.tail_rows(nf), 0).t();
  dmat vcov;
  if (h) {
    data.setparameters(theta);
    data.setsize(h, t);
    data.estep();
    vcov = data.vmat();
  } else {
    vcov.fill(arma::datum::nan);
  }

  return List::create(
    Named("out") = wrap(out),
    Named("theta") = wrap(theta),
    Named("vcov") = wrap(vcov)
  );
}

