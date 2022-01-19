#include "integrand-expected-survival.h"
#include "integrand-mixed-mult-logit-term.h"

using namespace ghqCpp;

static simple_mem_stack<double> R_mem;

ghq_data vecs_to_ghq_data(arma::vec const &weights, arma::vec const &nodes){
  if(nodes.size() != weights.size())
    throw std::invalid_argument("nodes.size() != weights.size()");
  return { &nodes[0], &weights[0], nodes.size() };
}

//' @export
// [[Rcpp::export("expected_survival_term", rng = false)]]
double expected_survival_term_to_R
  (arma::vec const &eta, arma::vec const &ws,
   arma::mat const &M, arma::mat const &Sigma, arma::vec const &weights,
   arma::vec const &nodes, size_t const target_size = 128,
   bool const use_adaptive = true){
  R_mem.reset();
  expected_survival_term<false> prob(eta, ws, M, Sigma);

  auto ghq_data_pass = vecs_to_ghq_data(weights, nodes);

  std::vector<double> res;
  if(use_adaptive){
    adaptive_problem prob_adap(prob, R_mem);
    res = ghq(ghq_data_pass, prob_adap, R_mem, target_size);

  } else
    res = ghq(ghq_data_pass, prob, R_mem, target_size);

  return res[0];
}

//' @export
// [[Rcpp::export("mixed_mult_logit_term", rng = false)]]
double mixed_mult_logit_term_to_R
  (arma::mat const &eta, arma::mat const &Sigma,
   arma::uvec const &which_category, arma::vec const &weights,
   arma::vec const &nodes, size_t const target_size = 128,
   bool const use_adaptive = true){
  R_mem.reset();

  mixed_mult_logit_term<false> prob(eta, Sigma, which_category);

  auto ghq_data_pass = vecs_to_ghq_data(weights, nodes);

  std::vector<double> res;
  if(use_adaptive){
    adaptive_problem prob_adap(prob, R_mem);
    res = ghq(ghq_data_pass, prob_adap, R_mem, target_size);

  } else
    res = ghq(ghq_data_pass, prob, R_mem, target_size);

  return res[0];
}


//' Computes the Gradient
//'
//' @description
//' Computes the gradient of the expectation of the multinomial logit factors
//' with respect to the offsets on the linear predictor scale.
//'
//' @export
// [[Rcpp::export(rng = false)]]
Rcpp::NumericVector mixed_mult_logit_term_grad
  (arma::mat const &eta, arma::mat const &Sigma,
   arma::uvec const &which_category, arma::vec const &weights,
   arma::vec const &nodes, size_t const target_size = 128,
   bool const use_adaptive = true){
  R_mem.reset();

  mixed_mult_logit_term<true> prob(eta, Sigma, which_category);

  auto ghq_data_pass = vecs_to_ghq_data(weights, nodes);

  std::vector<double> res;
  if(use_adaptive){
    adaptive_problem prob_adap(prob, R_mem);
    res = ghq(ghq_data_pass, prob_adap, R_mem, target_size);

  } else
    res = ghq(ghq_data_pass, prob, R_mem, target_size);

  Rcpp::NumericVector out(res.size() - 1);
  std::copy(res.begin() + 1, res.end(), &out[0]);
  out.attr("Value") = res[0];

  return out;
}
