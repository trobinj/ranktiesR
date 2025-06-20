#ifndef misc_h
#define misc_h

#include <RcppArmadillo.h>
#include "ranktiesR_types.h"

// [[Rcpp::depends(RcppArmadillo)]]

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

uvec rankvec(dvec y)
{
  int n = y.n_elem;
  uvec indx = sort_index(y);
  uvec rnks(n);
  for (int i = 0; i < n; ++i) {
    rnks(indx(i)) = i + 1;
  }
  return rnks;
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

	if ((b - a) / d < sqrt2pi) {
		do {
			z = R::runif(low, upp);
			u = R::runif(0.0, 1.0);
			p = exp((d - pow(z,2)) / 2.0);
		} while (u > p);
	} else {
		do {
			z = R::rnorm(0.0, 1.0);
		} while (low > z || z > upp);
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

#endif
