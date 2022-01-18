#include "integrand-expected-survival.h"

using namespace ghqCpp;

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
   size_t const n_rep = 1, bool const use_adaptive = true){
  simple_mem_stack<double> mem;
  expected_survival_term prob(eta, ws, M, Sigma);

  auto ghq_data_pass = vecs_to_ghq_data(weights, nodes);

  std::vector<double> res;
  if(use_adaptive){
    adaptive_problem prob_adap(prob, mem);

    for(size_t i = 0; i < n_rep; ++i){
      mem.reset();
      res = ghq(ghq_data_pass, prob_adap, mem, target_size);
    }

  } else
    for(size_t i = 0; i < n_rep; ++i){
      mem.reset();
      res = ghq(ghq_data_pass, prob, mem, target_size);
    }

  return res[0];
}
