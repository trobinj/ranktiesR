#ifndef TYPES
#define TYPES

#include <RcppArmadillo.h>

using arma::dvec;
using arma::dmat;
using arma::uvec;
using arma::umat;
using arma::ivec;

using rvec = arma::rowvec;
using uint = unsigned int;

typedef uvec (*clfunc)(dvec, double);

#endif