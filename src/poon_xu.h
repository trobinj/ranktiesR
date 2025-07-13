#ifndef poonxu_h
#define poonxu_h

#include <RcppArmadillo.h>
#include "ranktiesR_types.h"
#include "misc.h"

// [[Rcpp::depends(RcppArmadillo)]]

// [[Rcpp::export]]
uvec poon_xu(dvec u, double delta)
{
	int n = u.n_elem;

	uvec r = rankvec(-u);
	uvec y(n);

	int a = 1;
	int t = 1;

	y(indx(r, 1)) = 1;
	for (int i = 2; i <= n; ++i) {
		y(indx(r, i)) = y(indx(r, i - 1)) + 1;
		if (u(indx(r, i - 1)) - u(indx(r, i)) >= delta) {
			if (u(indx(r, a)) - u(indx(r, i - 1)) < delta) {
				y(find(r >= a && r < i)).fill(t);
				y(indx(r, i)) = t + 1;
			}
			a = i;
			t = y(indx(r, i - 1)) + 1;
		}
	}
	if (u(indx(r, a)) - u(indx(r, n)) < delta) {
		y(find(r >= a && r <= n)).fill(t);
	} else {
		y(indx(r, n)) = y(indx(r, n - 1)) + 1;
	}

	return y;
}

#endif
