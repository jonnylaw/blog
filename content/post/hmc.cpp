#include <Rcpp.h>
using namespace Rcpp;
// [[Rcpp::plugins(cpp11)]]

// [[Rcpp::export]]
NumericVector gradient(NumericMatrix ys, NumericVector position) {
  int n = ys.size();
  NumericVector ssy1 = (ys(_,0) - position(0)) * (ys(_,0) - position(0));
  NumericVector ssy2 = (ys(_,1) - position(2)) * (ys(_,1) - position(2));
  NumericVector out = NumericVector::create(1/position(1) * sum(ys(_,0) - position(0) ) - position(0) /position(1),
                                            -n/position(1) + (3 * std::accumulate(ssy1.begin(), ssy1.end(), 0.0)) / (2 * pow(position(1), 3)),
                                            1/position(3) *sum(ys(_,1) - position(2) ) - position(2) /position(3) ,
                                            -n/position(3) + (3 * std::accumulate(ssy2.begin(), ssy2.end(), 0.0)) / (2 * pow(position(3), 3)));
  
  return out;
}

// [[Rcpp::export]]
double logDensity(NumericMatrix ys, NumericVector p) {
  double ll = 0;
  int n = ys.nrow();
  for (int i = 0; i < n; i++) {
    ll += R::dnorm(ys(i, 0), p(0), p(1), true) + R::dnorm(ys(i, 1), p(2), p(3), true);
  }
  return ll;
}

// [[Rcpp::export]]
double logAcceptance(NumericMatrix ys,
                     NumericVector prop, 
                     NumericVector qp) {
  int d = qp.size() / 2;
  double phiP = 0.0;
  double propPhiP = 0.0;
  for (int i = 0; i < d; i++) {
    phiP += R::dnorm(qp(d + i), 0.0, 1.0, true);
    propPhiP += R::dnorm(prop(d + i), 0.0, 1.0, true);
  }
  
  NumericVector propPosition(d);
  NumericVector position(d);
  for (int i = 0; i < d; i++) {
    propPosition(i) = prop(i);
    position(i) = qp(i);
  }
  
  double a = logDensity(ys, propPosition) - phiP - logDensity(ys, position) + propPhiP;
  return a;
}

// [[Rcpp::export]]
NumericVector leapfrog(
    NumericMatrix ys,
    NumericVector qp,
    double stepSize) {
  
  // unasign values
  int d = qp.size() / 2;
  NumericVector momentum(d);
  NumericVector position(d);
  for (int i = 0; i < d; i++) {
    momentum(i) = qp(d + i);
    position(i) = qp(i);
  }
  
  NumericVector momentum1 = momentum + gradient(ys, position) * 0.5 * stepSize;
  NumericVector position1 = position + stepSize * momentum1;
  NumericVector momentum2 = momentum + gradient(ys, position1) * 0.5 * stepSize;
  
  NumericVector newqp(2 * d);
  for (int i = 0; i < d; i++) {
    newqp(i) = position1(i);
    newqp(d + i) = momentum2(i);
  }
  return newqp;
}

// [[Rcpp::export]]
NumericVector leapfrogs(
    NumericMatrix ys,
    NumericVector qp,
    double stepSize,
    int l) {
  
  // initialise vectors
  int d = qp.size();
  NumericVector prop(d);
  for (int i = 0; i < d; i++) {
    prop(i) = qp(i);
  }
  
  // perform l leapfrogs
  for (int i = 0; i < l; i++) {
    prop = leapfrog(ys, prop, stepSize);
  }
  
  return prop;
}

// [[Rcpp::export]]
NumericVector initialMomentum(int d) {
  NumericVector momentum(d);
  for (int j = 0; j < d; j++) {
    momentum(j) = R::rnorm(0.0, 1.0);
  }
  return momentum;
}

// [[Rcpp::export]]
NumericVector hmcStep(NumericMatrix ys,
                      NumericVector position,
                      double stepSize,
                      int l) {
  int d = position.size();
  NumericVector momentum = initialMomentum(d);
  NumericVector qp(2 * d);
  for (int i = 0; i < d; i++) {
    qp(i) = position(i);
    qp(d + i) = momentum(i);
  }
  NumericVector prop = leapfrogs(ys, qp, stepSize, l);
  double a = logAcceptance(ys, prop, qp);
  double u = R::runif(0, 1);
  NumericVector nextParameter(d);
  if (log(u) < a) {
    for (int i = 0; i < d; i++) {
      nextParameter(i) = qp(i);
    }
  } else {
    nextParameter = position;
  }
  
  return nextParameter;
}

// [[Rcpp::export]]
NumericMatrix hmc(
    NumericMatrix ys,
    NumericVector position,
    double stepSize,
    int l,
    int N) {
  int d = position.size();
  NumericMatrix mat(N, d);
  
  // initialise first row of output
  for (int j = 0; j < d; j++) {
    mat(0, j) = position(j);
  }
  
  // create markov chain
  for (int i = 1; i < N; i++) {
    NumericVector newP = hmcStep(ys, mat(i-1, _), stepSize, l);
    for (int j = 0; j < d; j++) {
      mat(i, j) = newP(j);
    }
  }
  return mat;
}

// You can include R code blocks in C++ files processed with sourceCpp
// (useful for testing and development). The R code will be automatically 
// run after the compilation.
//

/*** R
*/
