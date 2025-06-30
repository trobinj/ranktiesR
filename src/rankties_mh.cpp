#include <RcppArmadillo.h>
#include "ranktiesR_types.h"
#include "misc.h"
#include "seqcluster.h"
#include "hclust.h"
#include "poon_xu.h"
#include "ghk.h"

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

class ranktiesvector
{
private:

  uvec y;
  dvec x;
  dvec z;
  dmat zfull;
  dmat zthin;
  int k, m, t;
  std::string type;
  clfunc cluster;
  double delta, scale;

public:

  ranktiesvector(uvec y, dvec x, dvec z, std::string type) : y(y), x(x), z(z), type(type)
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
    delta = 0.1;
    scale = 0.1;

    zfull.set_size(t*(m+1), k-1);
    zthin.set_size(m, k-1);
    zfull.row(0) = z.t();
  }

  void setscale(double scale) {
    this->scale = scale;
  }

  void setsize(int m, int t, double delta)
  {
    z = zfull.row(0).t();

    this->m = m;
    this->t = t;
    this->delta = delta;

    zthin.resize(m, k-1);
    zfull.resize(t*(m+1), k-1);
    zfull.row(0) = z.t();
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
      ynew = cluster(unew, delta);
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
    dvec z(k-1);
    dmat betag(size(beta));
    dmat sigmg(size(sigm));
    dvec sigmv;

    betag.fill(0.0);
    sigmg.fill(0.0);

    for (int j = 0; j < m; ++j) {
      z = (zthin.row(j).t() - beta * x);
      betag = betag + R * z * x.t();
      sigmg = sigmg + 0.5 * (2 * R - (R * I) - 2.0 * 
        R * z * z.t() * R + R * z * z.t() * R * I);
    }

    betag = betag / m;
    sigmg = sigmg / m;

    sigmv = lowertri(sigmg);
    sigmv = sigmv.tail(k * (k - 1) / 2);

    return join_vert(vectorise(betag), sigmv);
  }
};

class ranktiesdata
{
private:

  int n, p, q, k;
  umat y;
  dmat x;
  dmat z;
  dmat beta;
  dmat sigm;
  dmat xx, xxinv;
  std::vector<ranktiesvector> data;
  std::string type;

public:

  ranktiesdata(umat y, dmat x, dmat z, std::string type) : y(y), x(x), z(z), type(type)
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
      data.emplace_back(y.row(i).t(), x.row(i).t(), z.row(i).t(), type);
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

  void setsize(int m, int t, double delta)
  {
    for (auto& obs : data) {
      obs.setsize(m, t, delta);
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
Rcpp::List rankties(umat y, dmat x, dmat z, uvec n, uvec m, int t, double delta, double scale, std::string type, int h, bool print)
{
  using namespace Rcpp;

  int k = y.n_cols;
  int p = x.n_cols;
  int q = k * (k - 1)/2;

  int m0 = m(0);
  int mf = m(1);
  int n0 = n(0);
  int nf = n(1);

  ranktiesdata data(y, x, z, type);

  data.setscale(scale);
  data.setsize(m0, t, delta);

  dmat out(n0 + nf + 1, (k-1)*p + q);
  dmat bhat, shat;

  out.row(0) = data.getparameters().t();

  for (int i = 0; i < n0 + nf; ++i) {

    if (i == n0) {
      data.setsize(mf, t, delta);
    }

    data.estep();
    data.mstep();

    out.row(i+1) = data.getparameters().t();

    if ((i+1) % 1 == 0 && print) {
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
    data.setsize(h, t, delta);
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
double ranktiesloglik(umat y, dmat x, dvec theta, std::string type, double delta, int m)
{
  int n = y.n_rows;
  int k = y.n_cols;
  int p = x.n_cols;
  int q = k * (k - 1)/2;

  dmat beta(k-1,p);
  dmat sigm(k-1,k-1);

  clfunc cluster;
  if (type == "seq_min") cluster = seq_min;
  if (type == "seq_max") cluster = seq_max;
  if (type == "seq_avg") cluster = seq_avg;
  if (type == "poon_xu") cluster = poon_xu;

  dmat temp = theta.head((k-1)*p);
  beta = temp.reshape(size(beta));
  sigm = vec2lower(theta.tail(q), true);
  dmat Q = arma::chol(sigm, "lower");

  dmat eta = beta * x.t();
  uvec ytmp(k);
  dvec utmp(k);
  dvec mu(k);
  uvec yi(k);

  dvec prob(n);
  for (int i = 0; i < n; ++i) {
    mu = eta.col(i);
    yi = y.row(i).t();
    for (int j = 0; j < m; ++j) {
      utmp.head(k-1) = mvrnorm(mu, Q, true);
      ytmp = cluster(utmp, delta);
      if (all(yi == ytmp)) {
        ++prob(i);
      }
    }
    prob(i) = prob(i)/m;
  }

  double minprob = min(prob(find(prob > 0.0)));
  for (int i = 0; i < n; ++i) {
    Rcpp::Rcout << prob(i) << "\n";
    if (prob(i) == 0) {
      prob(i) = minprob;
    }
  }

  return sum(log(prob));
}

// [[Rcpp::export]]
double foo(arma::vec m, arma::mat s, arma::vec low, arma::vec upp, int n)
{
  return ghk(m, s, low, upp, n);
}

// [[Rcpp::export]]
Rcpp::List residuals(umat y, dvec x, dvec theta, std::string type, double delta, int n)
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
    v.row(i) = cluster(utmp, delta).t();
  }

  return List::create(
    Named("mean") = wrap(mean(conv_to<dmat>::from(v), 0) - mean(conv_to<dmat>::from(y), 0)),
    Named("cov") = wrap(cov(conv_to<dmat>::from(v), 1) - cov(conv_to<dmat>::from(y), 1))
  );
}
