#ifndef misc_h
#define misc_h

#include <RcppArmadillo.h>
#include "ranktiesR_types.h"

// [[Rcpp::depends(RcppArmadillo)]]

unsigned int indx(uvec x, int y)
{
	return as_scalar(find(x == y));
}

dvec zstart(uvec y, double delta)
{
	using namespace arma;

	int k = y.n_elem;
	int kmax = max(y);
	dvec u = regspace<dvec>(kmax, -1, 1) * delta * 1.5;
	dvec z(k);

	for (int j = 0; j < kmax; ++j) {
		z(find(y == j + 1)).fill(u(j));
	}

	z = z - z(k - 1) + randu(k, distr_param(-0.01, 0.01));

	return z.head(k - 1);
}

int factorial(uint n)
{
  return (n == 1 || n == 0) ? 1 : factorial(n - 1) * n;
}

umat allpermutations(int k)
{
  uvec y = arma::regspace<uvec>(1, k);
  umat x(k, factorial(k));

  int i = 0;
  do {
    x.col(i++) = y;
  } while (std::next_permutation(y.begin(), y.end()));

  return x.t();
}

class mvndist
{
private:

	dvec mu;
	dmat sigma;
	double ldet;
	dmat R;

public:

	mvndist(dvec mu, dmat sigma) : mu(mu), sigma(sigma)
	{
		log_det_sympd(ldet, 2.0 * M_PI * sigma);
		R = arma::chol(inv(sigma));
	}

	double logpdf(dvec y)
	{
		dvec z = R * (y - mu);
		return -ldet/2 - as_scalar(z.t() * z)/2;
	}
};

uvec rankvec(dvec y) // change argument type to auto?
{
  int n = y.n_elem;
  uvec indx = sort_index(y);
  uvec rnks(n);
  for (int i = 0; i < n; ++i) {
    rnks(indx(i)) = i + 1;
  }
  return rnks;
}

uvec rankvec(uvec y)
{
  int n = y.n_elem;
  uvec indx = sort_index(y);
  uvec rnks(n);
  for (int i = 0; i < n; ++i) {
    rnks(indx(i)) = i + 1;
  }
  return rnks;
}

bool rankmatch(uvec y, uvec x)
{
  int n = y.n_elem;

  uvec indx = sort_index(x);
  y = y(indx);

  for (int i = 1; i < n; ++i) {
    if (y(i) < y(i - 1)) {
      return false;
    }
  }
  return true;
}

dmat rankmat(uvec r)
{
	int k = r.n_elem;
	dmat y(k - 1, k);

	for (int i = 0; i < k - 1; ++i) {
		y(i, indx(r, i + 1)) =  1;
		y(i, indx(r, i + 2)) = -1;
	}
	return y;
}

uvec rankset(uvec y, umat x)
{
	int n = x.n_rows;
	uvec inc(n);

	for (int i = 0; i < n; ++i) {
		if (rankmatch(y, x.row(i).t())) {
			inc(i) = 1;
		}
	}

	return find(inc);
}

bool vecmatch(uvec a, uvec b, int k)
{
	return sum(a == b) < k ? false : true;
}

dmat vec2lower(dvec x, bool upper)
{
	int n = (sqrt(8 * x.n_elem + 1) - 1) / 2;
	int t = 0;
	dmat y(n, n);
	for (int j = 0; j < n; ++j) {
		for (int i = j; i < n; ++i) {
			y(i, j) = x(t);
			++t;
		}
	}
	if (upper) {
		y = symmatl(y);
	}

	return y;
}

dvec mvrnorm(dvec mu, dmat sigma, bool cholesky)
{
  int p = sigma.n_cols;
  if (cholesky) {
    return mu + sigma * arma::randn(p);
  }
  return mu + arma::chol(sigma, "lower") * arma::randn(p);
}

void vswap(dvec &x, int a, int b) {
  double y = x(a);
  x(a) = x(b);
  x(b) = y;
}

dvec lowertri(dmat x, bool diag = true)
{
	if (!diag) x.shed_row(0);
	int n = x.n_rows;
	int m = x.n_cols;
	int d = std::min(n,m) * (std::min(n,m) + 1) / 2;
	if (n > m) {
		d = d + (n - m) * m;
	}
	dvec y(d);
	int t = 0;
	for (int j = 0; j < std::min(n,m); ++j) {
		for (int i = j; i < n; ++i) {
			y(t) = x(i, j);
			++t;
		}
	}
	return y;
}

int randint(int a, int b)
{
  return floor(R::runif(0.0, 1.0) * (b - a + 1)) + a;
}

dvec srs(dvec x, int n)
{
  int j;
  int l = x.n_elem;
  for (int i = 0; i < n; ++i) {
    j = randint(i, l - 1);
    vswap(x, i, j);
  }
  return x.head(n);
}

double rnormint(double m, double s, double a, double b)
{
	const double sqrt2pi = 2.506628;
	double low = (a - m) / s;
	double upp = (b - m) / s;
	double z, u, p, d;

	if (upp < 0) {
		d = pow(upp,2);
	} else if (low > 0) {
		d = pow(low,2);
	} else {
		d = 0.0;
	}

	if (d == 0.0 && b - a > sqrt2pi) {
		do {
			z = R::rnorm(0.0, 1.0);
		} while (low > z || z > upp);
	} else {
		do {
			z = R::runif(low, upp);
			u = R::runif(0.0, 1.0);
			p = exp((d - pow(z,2)) / 2.0);
		} while (u > p);
	}

	return z * s + m;
}

double rnormpos(double m, double s, bool pos)
{
	double l, a, z, p, u;
	l = pos ? -m/s : m/s;
	a = (l + sqrt(pow(l, 2) + 4.0)) / 2.0;

	do {
		z = R::rexp(1.0) / a + l;
		u = R::runif(0.0, 1.0);
		p = exp(-pow(z - a, 2) / 2.0);
	} while (u > p);

	return pos ? z * s + m : -z * s + m;
}

double rtnorm(double m, double s, double a, double b)
{
	bool anan = std::isnan(a);
	bool bnan = std::isnan(b);

	if (anan && bnan) {
		return R::rnorm(m, s);
	}
	if (anan) {
		return rnormpos(m - b, s, 0) + b;
	}
	if (bnan) {
		return rnormpos(m - a, s, 1) + a;
	}
	return rnormint(m, s, a, b);
}

class cnorm
{
private:

	int n;
	dmat b;
	dvec v;
	dvec m;
	dmat c;

public:

  cnorm() {}

	cnorm(dmat c) : c(c)
	{
		n = c.n_cols;
		m = arma::zeros(n);
		b.set_size(n, n - 1);
		v.set_size(n);
		setc(c);
	}

	cnorm(dvec m, dmat c) : m(m), c(c)
	{
		n = m.n_elem;
		b.set_size(n, n - 1);
		v.set_size(n);
		setc(c);
	}

	void setn(int n) {
	  n = c.n_cols;
	  m = arma::zeros(n);
	  b.set_size(n, n - 1);
	  v.set_size(n);
	}

	void setm(dvec x)
	{
		m = x;
	}

	void setc(dmat c)
	{
		for (int i = 0; i < n; ++i) {
			dmat c22 = c;
			c22.shed_row(i);
			c22.shed_col(i);
			dmat r22 = inv(c22);
			dmat c12 = c.row(i);
			c12.shed_col(i);
			b.row(i) = c12 * r22;
			v(i) = c(i,i) - as_scalar(b.row(i) * c12.t());
		}
	}

  dmat gets()
  {
    return c;
  }

  dvec getm()
  {
    return m;
  }

	double gets(int i)
	{
		return sqrt(v(i));
	}

	double getm(int i, dvec y)
	{
		dvec m2(m);
		dvec y2(y);
		m2.shed_row(i);
		y2.shed_row(i);
		return m(i) + as_scalar(b.row(i) * (y2 - m2));
	}
};

dmat zsamp(uvec y, cnorm& zdist, int b, int n)
{
  const double ninf = arma::datum::nan;
  const double pinf = arma::datum::nan;

  int k = y.n_elem;
  int r = max(y);

  dmat u(n + b, k);
  dvec v(r, arma::fill::randn);
  dvec z(k);
  v = sort(v, "descend");
  for (int j = 0; j < r; ++j) {
    z(find(y == j + 1)).fill(v(j));
  }
  u.row(0) = z.t();
  u.row(0) = u.row(0) - u(0, k-1);

  std::vector<uvec> lower;
  std::vector<uvec> upper;

  lower.reserve(k);
  upper.reserve(k);
  for (int j = 0; j < k; ++j) {
    lower.emplace_back(find(y == y(j) + 1));
    upper.emplace_back(find(y == y(j) - 1));
  }

  dvec ui(k);
  double mj, sj, aj, bj;
  for (int i = 1; i < n + b; ++i) {
		ui = u.row(i-1).t();
    for (int j = 0; j < k - 1; ++j) {
      mj = zdist.getm(j, ui.head(k-1));
      sj = zdist.gets(j);
      aj = lower[j].is_empty() ? ninf : max(ui(lower[j]));
      bj = upper[j].is_empty() ? pinf : min(ui(upper[j]));
      ui(j) = rtnorm(mj, sj, aj, bj);
    }
		u.row(i) = ui.t();
  }

  return u.tail_rows(n);
}

#endif
