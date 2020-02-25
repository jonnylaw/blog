// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include <RcppArmadillo.h>
#include <Rcpp.h>

using namespace Rcpp;

// mvrnorm
arma::colvec mvrnorm(const arma::colvec mean, const arma::mat Sigma);
RcppExport SEXP _jonnylaw_mvrnorm(SEXP meanSEXP, SEXP SigmaSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::colvec >::type mean(meanSEXP);
    Rcpp::traits::input_parameter< const arma::mat >::type Sigma(SigmaSEXP);
    rcpp_result_gen = Rcpp::wrap(mvrnorm(mean, Sigma));
    return rcpp_result_gen;
END_RCPP
}
// mvrnormsvd
arma::colvec mvrnormsvd(const arma::colvec mean, const arma::mat Sigma);
RcppExport SEXP _jonnylaw_mvrnormsvd(SEXP meanSEXP, SEXP SigmaSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::colvec >::type mean(meanSEXP);
    Rcpp::traits::input_parameter< const arma::mat >::type Sigma(SigmaSEXP);
    rcpp_result_gen = Rcpp::wrap(mvrnormsvd(mean, Sigma));
    return rcpp_result_gen;
END_RCPP
}
// backwardSample
arma::mat backwardSample(const arma::mat g, const arma::mat w, const arma::mat mts, const arma::cube cts, const arma::mat ats, const arma::cube rts);
RcppExport SEXP _jonnylaw_backwardSample(SEXP gSEXP, SEXP wSEXP, SEXP mtsSEXP, SEXP ctsSEXP, SEXP atsSEXP, SEXP rtsSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::mat >::type g(gSEXP);
    Rcpp::traits::input_parameter< const arma::mat >::type w(wSEXP);
    Rcpp::traits::input_parameter< const arma::mat >::type mts(mtsSEXP);
    Rcpp::traits::input_parameter< const arma::cube >::type cts(ctsSEXP);
    Rcpp::traits::input_parameter< const arma::mat >::type ats(atsSEXP);
    Rcpp::traits::input_parameter< const arma::cube >::type rts(rtsSEXP);
    rcpp_result_gen = Rcpp::wrap(backwardSample(g, w, mts, cts, ats, rts));
    return rcpp_result_gen;
END_RCPP
}
// leapfrogCpp
NumericVector leapfrogCpp(Function gradient, NumericMatrix ys, NumericVector qp, double stepSize);
RcppExport SEXP _jonnylaw_leapfrogCpp(SEXP gradientSEXP, SEXP ysSEXP, SEXP qpSEXP, SEXP stepSizeSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Function >::type gradient(gradientSEXP);
    Rcpp::traits::input_parameter< NumericMatrix >::type ys(ysSEXP);
    Rcpp::traits::input_parameter< NumericVector >::type qp(qpSEXP);
    Rcpp::traits::input_parameter< double >::type stepSize(stepSizeSEXP);
    rcpp_result_gen = Rcpp::wrap(leapfrogCpp(gradient, ys, qp, stepSize));
    return rcpp_result_gen;
END_RCPP
}
// advState
List advState(const arma::mat g, const arma::colvec mt, const arma::mat ct, const arma::mat w);
RcppExport SEXP _jonnylaw_advState(SEXP gSEXP, SEXP mtSEXP, SEXP ctSEXP, SEXP wSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::mat >::type g(gSEXP);
    Rcpp::traits::input_parameter< const arma::colvec >::type mt(mtSEXP);
    Rcpp::traits::input_parameter< const arma::mat >::type ct(ctSEXP);
    Rcpp::traits::input_parameter< const arma::mat >::type w(wSEXP);
    rcpp_result_gen = Rcpp::wrap(advState(g, mt, ct, w));
    return rcpp_result_gen;
END_RCPP
}
// oneStep
List oneStep(const arma::mat ff, const arma::mat v, const arma::colvec at, const arma::mat rt);
RcppExport SEXP _jonnylaw_oneStep(SEXP ffSEXP, SEXP vSEXP, SEXP atSEXP, SEXP rtSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::mat >::type ff(ffSEXP);
    Rcpp::traits::input_parameter< const arma::mat >::type v(vSEXP);
    Rcpp::traits::input_parameter< const arma::colvec >::type at(atSEXP);
    Rcpp::traits::input_parameter< const arma::mat >::type rt(rtSEXP);
    rcpp_result_gen = Rcpp::wrap(oneStep(ff, v, at, rt));
    return rcpp_result_gen;
END_RCPP
}
// updateState
List updateState(const arma::mat ft, const arma::mat at, const arma::mat rt, const arma::colvec predicted, const arma::mat predcov, const arma::colvec y, const arma::mat v);
RcppExport SEXP _jonnylaw_updateState(SEXP ftSEXP, SEXP atSEXP, SEXP rtSEXP, SEXP predictedSEXP, SEXP predcovSEXP, SEXP ySEXP, SEXP vSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::mat >::type ft(ftSEXP);
    Rcpp::traits::input_parameter< const arma::mat >::type at(atSEXP);
    Rcpp::traits::input_parameter< const arma::mat >::type rt(rtSEXP);
    Rcpp::traits::input_parameter< const arma::colvec >::type predicted(predictedSEXP);
    Rcpp::traits::input_parameter< const arma::mat >::type predcov(predcovSEXP);
    Rcpp::traits::input_parameter< const arma::colvec >::type y(ySEXP);
    Rcpp::traits::input_parameter< const arma::mat >::type v(vSEXP);
    rcpp_result_gen = Rcpp::wrap(updateState(ft, at, rt, predicted, predcov, y, v));
    return rcpp_result_gen;
END_RCPP
}
// missing_f
arma::mat missing_f(const arma::mat f, const arma::colvec y);
RcppExport SEXP _jonnylaw_missing_f(SEXP fSEXP, SEXP ySEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::mat >::type f(fSEXP);
    Rcpp::traits::input_parameter< const arma::colvec >::type y(ySEXP);
    rcpp_result_gen = Rcpp::wrap(missing_f(f, y));
    return rcpp_result_gen;
END_RCPP
}
// missing_v
arma::mat missing_v(const arma::mat v, const arma::colvec y);
RcppExport SEXP _jonnylaw_missing_v(SEXP vSEXP, SEXP ySEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::mat >::type v(vSEXP);
    Rcpp::traits::input_parameter< const arma::colvec >::type y(ySEXP);
    rcpp_result_gen = Rcpp::wrap(missing_v(v, y));
    return rcpp_result_gen;
END_RCPP
}
// flatten
arma::colvec flatten(arma::colvec X);
RcppExport SEXP _jonnylaw_flatten(SEXP XSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::colvec >::type X(XSEXP);
    rcpp_result_gen = Rcpp::wrap(flatten(X));
    return rcpp_result_gen;
END_RCPP
}
// kalmanStep
List kalmanStep(const arma::colvec yo, const arma::mat f, const arma::mat g, const arma::mat v, const arma::mat w, const arma::colvec mt, const arma::mat ct);
RcppExport SEXP _jonnylaw_kalmanStep(SEXP yoSEXP, SEXP fSEXP, SEXP gSEXP, SEXP vSEXP, SEXP wSEXP, SEXP mtSEXP, SEXP ctSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::colvec >::type yo(yoSEXP);
    Rcpp::traits::input_parameter< const arma::mat >::type f(fSEXP);
    Rcpp::traits::input_parameter< const arma::mat >::type g(gSEXP);
    Rcpp::traits::input_parameter< const arma::mat >::type v(vSEXP);
    Rcpp::traits::input_parameter< const arma::mat >::type w(wSEXP);
    Rcpp::traits::input_parameter< const arma::colvec >::type mt(mtSEXP);
    Rcpp::traits::input_parameter< const arma::mat >::type ct(ctSEXP);
    rcpp_result_gen = Rcpp::wrap(kalmanStep(yo, f, g, v, w, mt, ct));
    return rcpp_result_gen;
END_RCPP
}
// kalmanFilter
List kalmanFilter(const arma::mat ys, const arma::mat f, const arma::mat g, const arma::mat v, const arma::mat w, const arma::colvec m0, const arma::mat c0);
RcppExport SEXP _jonnylaw_kalmanFilter(SEXP ysSEXP, SEXP fSEXP, SEXP gSEXP, SEXP vSEXP, SEXP wSEXP, SEXP m0SEXP, SEXP c0SEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::mat >::type ys(ysSEXP);
    Rcpp::traits::input_parameter< const arma::mat >::type f(fSEXP);
    Rcpp::traits::input_parameter< const arma::mat >::type g(gSEXP);
    Rcpp::traits::input_parameter< const arma::mat >::type v(vSEXP);
    Rcpp::traits::input_parameter< const arma::mat >::type w(wSEXP);
    Rcpp::traits::input_parameter< const arma::colvec >::type m0(m0SEXP);
    Rcpp::traits::input_parameter< const arma::mat >::type c0(c0SEXP);
    rcpp_result_gen = Rcpp::wrap(kalmanFilter(ys, f, g, v, w, m0, c0));
    return rcpp_result_gen;
END_RCPP
}

static const R_CallMethodDef CallEntries[] = {
    {"_jonnylaw_mvrnorm", (DL_FUNC) &_jonnylaw_mvrnorm, 2},
    {"_jonnylaw_mvrnormsvd", (DL_FUNC) &_jonnylaw_mvrnormsvd, 2},
    {"_jonnylaw_backwardSample", (DL_FUNC) &_jonnylaw_backwardSample, 6},
    {"_jonnylaw_leapfrogCpp", (DL_FUNC) &_jonnylaw_leapfrogCpp, 4},
    {"_jonnylaw_advState", (DL_FUNC) &_jonnylaw_advState, 4},
    {"_jonnylaw_oneStep", (DL_FUNC) &_jonnylaw_oneStep, 4},
    {"_jonnylaw_updateState", (DL_FUNC) &_jonnylaw_updateState, 7},
    {"_jonnylaw_missing_f", (DL_FUNC) &_jonnylaw_missing_f, 2},
    {"_jonnylaw_missing_v", (DL_FUNC) &_jonnylaw_missing_v, 2},
    {"_jonnylaw_flatten", (DL_FUNC) &_jonnylaw_flatten, 1},
    {"_jonnylaw_kalmanStep", (DL_FUNC) &_jonnylaw_kalmanStep, 7},
    {"_jonnylaw_kalmanFilter", (DL_FUNC) &_jonnylaw_kalmanFilter, 7},
    {NULL, NULL, 0}
};

RcppExport void R_init_jonnylaw(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
