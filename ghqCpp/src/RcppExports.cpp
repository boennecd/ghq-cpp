// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include <RcppArmadillo.h>
#include <Rcpp.h>

using namespace Rcpp;

#ifdef RCPP_USE_GLOBAL_ROSTREAM
Rcpp::Rostream<true>&  Rcpp::Rcout = Rcpp::Rcpp_cout_get();
Rcpp::Rostream<false>& Rcpp::Rcerr = Rcpp::Rcpp_cerr_get();
#endif

// expected_survival_term_to_R
double expected_survival_term_to_R(arma::vec const& eta, arma::vec const& ws, arma::mat const& M, arma::mat const& Sigma, arma::vec const& weights, arma::vec const& nodes, size_t const target_size, bool const use_adaptive);
RcppExport SEXP _ghqCpp_expected_survival_term_to_R(SEXP etaSEXP, SEXP wsSEXP, SEXP MSEXP, SEXP SigmaSEXP, SEXP weightsSEXP, SEXP nodesSEXP, SEXP target_sizeSEXP, SEXP use_adaptiveSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::traits::input_parameter< arma::vec const& >::type eta(etaSEXP);
    Rcpp::traits::input_parameter< arma::vec const& >::type ws(wsSEXP);
    Rcpp::traits::input_parameter< arma::mat const& >::type M(MSEXP);
    Rcpp::traits::input_parameter< arma::mat const& >::type Sigma(SigmaSEXP);
    Rcpp::traits::input_parameter< arma::vec const& >::type weights(weightsSEXP);
    Rcpp::traits::input_parameter< arma::vec const& >::type nodes(nodesSEXP);
    Rcpp::traits::input_parameter< size_t const >::type target_size(target_sizeSEXP);
    Rcpp::traits::input_parameter< bool const >::type use_adaptive(use_adaptiveSEXP);
    rcpp_result_gen = Rcpp::wrap(expected_survival_term_to_R(eta, ws, M, Sigma, weights, nodes, target_size, use_adaptive));
    return rcpp_result_gen;
END_RCPP
}
// mixed_mult_logit_term_to_R
double mixed_mult_logit_term_to_R(arma::mat const& eta, arma::mat const& Sigma, arma::uvec const& which_category, arma::vec const& weights, arma::vec const& nodes, size_t const target_size, bool const use_adaptive);
RcppExport SEXP _ghqCpp_mixed_mult_logit_term_to_R(SEXP etaSEXP, SEXP SigmaSEXP, SEXP which_categorySEXP, SEXP weightsSEXP, SEXP nodesSEXP, SEXP target_sizeSEXP, SEXP use_adaptiveSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::traits::input_parameter< arma::mat const& >::type eta(etaSEXP);
    Rcpp::traits::input_parameter< arma::mat const& >::type Sigma(SigmaSEXP);
    Rcpp::traits::input_parameter< arma::uvec const& >::type which_category(which_categorySEXP);
    Rcpp::traits::input_parameter< arma::vec const& >::type weights(weightsSEXP);
    Rcpp::traits::input_parameter< arma::vec const& >::type nodes(nodesSEXP);
    Rcpp::traits::input_parameter< size_t const >::type target_size(target_sizeSEXP);
    Rcpp::traits::input_parameter< bool const >::type use_adaptive(use_adaptiveSEXP);
    rcpp_result_gen = Rcpp::wrap(mixed_mult_logit_term_to_R(eta, Sigma, which_category, weights, nodes, target_size, use_adaptive));
    return rcpp_result_gen;
END_RCPP
}
// mixed_mult_logit_term_grad
Rcpp::NumericVector mixed_mult_logit_term_grad(arma::mat const& eta, arma::mat const& Sigma, arma::uvec const& which_category, arma::vec const& weights, arma::vec const& nodes, size_t const target_size, bool const use_adaptive);
RcppExport SEXP _ghqCpp_mixed_mult_logit_term_grad(SEXP etaSEXP, SEXP SigmaSEXP, SEXP which_categorySEXP, SEXP weightsSEXP, SEXP nodesSEXP, SEXP target_sizeSEXP, SEXP use_adaptiveSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::traits::input_parameter< arma::mat const& >::type eta(etaSEXP);
    Rcpp::traits::input_parameter< arma::mat const& >::type Sigma(SigmaSEXP);
    Rcpp::traits::input_parameter< arma::uvec const& >::type which_category(which_categorySEXP);
    Rcpp::traits::input_parameter< arma::vec const& >::type weights(weightsSEXP);
    Rcpp::traits::input_parameter< arma::vec const& >::type nodes(nodesSEXP);
    Rcpp::traits::input_parameter< size_t const >::type target_size(target_sizeSEXP);
    Rcpp::traits::input_parameter< bool const >::type use_adaptive(use_adaptiveSEXP);
    rcpp_result_gen = Rcpp::wrap(mixed_mult_logit_term_grad(eta, Sigma, which_category, weights, nodes, target_size, use_adaptive));
    return rcpp_result_gen;
END_RCPP
}

RcppExport SEXP run_testthat_tests(SEXP);

static const R_CallMethodDef CallEntries[] = {
    {"_ghqCpp_expected_survival_term_to_R", (DL_FUNC) &_ghqCpp_expected_survival_term_to_R, 8},
    {"_ghqCpp_mixed_mult_logit_term_to_R", (DL_FUNC) &_ghqCpp_mixed_mult_logit_term_to_R, 7},
    {"_ghqCpp_mixed_mult_logit_term_grad", (DL_FUNC) &_ghqCpp_mixed_mult_logit_term_grad, 7},
    {"run_testthat_tests", (DL_FUNC) &run_testthat_tests, 1},
    {NULL, NULL, 0}
};

RcppExport void R_init_ghqCpp(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
