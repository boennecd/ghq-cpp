#include "integrand-mixed-mult-logit-term.h"
#include "ghq-lp-utils.h"

namespace ghqCpp {

template<bool comp_grad>
mixed_mult_logit_term<comp_grad>::mixed_mult_logit_term
  (arma::mat const &eta, arma::mat const &Sigma,
   arma::uvec const &which_category):
  eta{eta}, Sigma_chol{arma::chol(Sigma, "upper")},
  which_category{which_category} {
  if(which_category.n_elem != eta.n_cols)
    throw std::invalid_argument("which_category.n_elem != eta.n_cols");
  else if(Sigma.n_cols != Sigma.n_rows)
    throw std::invalid_argument("Sigma.n_cols != Sigma.n_rows");
  else if(Sigma.n_cols != eta.n_rows)
    throw std::invalid_argument("Sigma.n_cols != eta.n_rows");
  for(arma::uword i : which_category)
    if(i > eta.n_rows)
      throw std::invalid_argument
        ("which_category has entries with i > eta.n_rows");
}

template<bool comp_grad>
void mixed_mult_logit_term<comp_grad>::eval
  (double const *points, size_t const n_points, double * __restrict__ outs,
   simple_mem_stack<double> &mem) const {
  double * const __restrict__ us{mem.get(n_points * n_vars() + n_vars())},
         * const __restrict__ us_j{us + n_points * n_vars()};

  // do the matrix product points.chol(Sigma)
  std::copy(points, points + n_points * n_vars(), us);
  {
    int const m = n_points, n = n_vars();
    constexpr double const alpha{1};
    constexpr char const c_R{'R'}, c_U{'U'}, c_N{'N'};
    F77_CALL(dtrmm)
      (&c_R, &c_U, &c_N, &c_N, &m, &n, &alpha, Sigma_chol.memptr(), &n,
       us, &m, 1, 1, 1, 1);
  }

  if constexpr (comp_grad){
    double * const terms{mem.get(2 * eta.n_cols + eta.n_cols * eta.n_rows)},
           * const denoms{terms + eta.n_cols},
           * const lps{denoms + eta.n_cols};

    // do the computations point by point
    for(size_t j = 0; j < n_points; ++j){
      // copy the random effect
      for(size_t i = 0; i < n_vars(); ++i)
        us_j[i] = us[j + i * n_points];

      // compute the integrand
      outs[j] = 1;
      for(arma::uword k = 0; k < eta.n_cols; ++k){
        size_t const offset = k * n_vars();
        denoms[k] = 1;
        double const * eta_k{eta.colptr(k)};
        for(size_t i = 0; i < n_vars(); ++i, ++eta_k){
          lps[i + offset] = std::exp(*eta_k + us_j[i]);
          denoms[k] += lps[i + offset];
        }

        double const numerator
          {which_category[k] < 1 ? 1 : lps[which_category[k] - 1 + offset]};
        terms[k] = numerator / denoms[k];
        outs[j] *= terms[k];
      }

      // compute the gradient
      double * d_eta_j{outs + j + n_points};
      for(arma::uword k = 0; k < eta.n_cols; ++k)
        for(size_t i = 0; i < n_vars(); ++i, d_eta_j += n_points){
          *d_eta_j = i + 1 == which_category[k]
            ? (denoms[k] - lps[i + k * n_vars()])
            : -lps[i + k * n_vars()];
          *d_eta_j *= outs[j] / denoms[k];
        }
    }

  } else {
    double * const __restrict__ lp{mem.get(n_vars())};

    // do the computations point by point
    for(size_t j = 0; j < n_points; ++j){
      // copy the random effect
      for(size_t i = 0; i < n_vars(); ++i)
        us_j[i] = us[j + i * n_points];

      outs[j] = 1;
      for(arma::uword k = 0; k < eta.n_cols; ++k){
        double denom{1};
        double const * eta_k{eta.colptr(k)};
        for(size_t i = 0; i < n_vars(); ++i, ++eta_k){
          lp[i] = std::exp(*eta_k + us_j[i]);
          denom += lp[i];
        }

        double const numerator
          {which_category[k] < 1 ? 1 : lp[which_category[k] - 1]};
        outs[j] *= numerator / denom;
      }
    }
  }
}

template<bool comp_grad>
double mixed_mult_logit_term<comp_grad>::log_integrand
  (double const *point, simple_mem_stack<double> &mem) const {
  double * const __restrict__ u{mem.get(2 * n_vars())},
         * const __restrict__ lp{u + n_vars()};

  // do the matrix product point.chol(Sigma)
  std::copy(point, point + n_vars(), u);
  {
    int const m = 1, n = n_vars();
    constexpr double const alpha{1};
    constexpr char const c_R{'R'}, c_U{'U'}, c_N{'N'};
    F77_CALL(dtrmm)
      (&c_R, &c_U, &c_N, &c_N, &m, &n, &alpha, Sigma_chol.memptr(), &n,
       u, &m, 1, 1, 1, 1);
  }

  double out{};
  for(arma::uword k = 0; k < eta.n_cols; ++k){
    double denom{1};
    double const * eta_k{eta.colptr(k)};
    for(size_t i = 0; i < n_vars(); ++i, ++eta_k){
      lp[i] = *eta_k + u[i];
      denom += std::exp(lp[i]);
    }

    if(which_category[k] < 1)
      out -= log(denom);
    else
      out += lp[which_category[k] - 1] - log(denom);
  }

  return out;
}

template<bool comp_grad>
double mixed_mult_logit_term<comp_grad>::log_integrand_grad
  (double const *point, double * __restrict__ grad,
   simple_mem_stack<double> &mem) const {
  double * const __restrict__ u{mem.get(3 * n_vars())},
         * const __restrict__ lp{u + n_vars()},
         * const __restrict__ lp_exp{lp + n_vars()};

  std::copy(point, point + n_vars(), u);
  {
    int const m = 1, n = n_vars();
    constexpr double const alpha{1};
    constexpr char const c_R{'R'}, c_U{'U'}, c_N{'N'};
    F77_CALL(dtrmm)
      (&c_R, &c_U, &c_N, &c_N, &m, &n, &alpha, Sigma_chol.memptr(), &n,
       u, &m, 1, 1, 1, 1);
  }

  double out{};
  std::fill(grad, grad + n_vars(), 0);
  for(arma::uword k = 0; k < eta.n_cols; ++k){
    double denom{1};
    double const * eta_k{eta.colptr(k)};
    for(size_t i = 0; i < n_vars(); ++i, ++eta_k){
      lp[i] = *eta_k + u[i];
      lp_exp[i] = exp(lp[i]);
      denom += lp_exp[i];
    }

    // handle the denominator term of the derivative
    for(size_t i = 0; i < n_vars(); ++i)
      grad[i] -= lp_exp[i] / denom;

    if(which_category[k] < 1)
      out -= log(denom);
    else {
      out += lp[which_category[k] - 1] - log(denom);
      grad[which_category[k] - 1] += 1;
    }
  }

  // compute grad <- Sigma_chol.grad
  {
    int const m = 1, n = n_vars();
    constexpr double const alpha{1};
    constexpr char const c_L{'L'}, c_U{'U'}, c_N{'N'};
    F77_CALL(dtrmm)
      (&c_L, &c_U, &c_N, &c_N, &n, &m, &alpha, Sigma_chol.memptr(), &n,
       grad, &n, 1, 1, 1, 1);
  }

  return out;
}

template<bool comp_grad>
void mixed_mult_logit_term<comp_grad>::log_integrand_hess
  (double const *point, double *hess,
 simple_mem_stack<double> &mem) const {
  double * const __restrict__ u{mem.get(2 * n_vars())},
         * const __restrict__ lp_exp{u + n_vars()};

  std::copy(point, point + n_vars(), u);
  {
    int const m = 1, n = n_vars();
    constexpr double const alpha{1};
    constexpr char const c_R{'R'}, c_U{'U'}, c_N{'N'};
    F77_CALL(dtrmm)
      (&c_R, &c_U, &c_N, &c_N, &m, &n, &alpha, Sigma_chol.memptr(), &n,
       u, &m, 1, 1, 1, 1);
  }

  std::fill(hess, hess + n_vars() * n_vars(), 0);
  for(arma::uword k = 0; k < eta.n_cols; ++k){
    double denom{1};
    double const * eta_k{eta.colptr(k)};
    for(size_t i = 0; i < n_vars(); ++i, ++eta_k){
      lp_exp[i] = exp(*eta_k + u[i]);
      denom += lp_exp[i];
    }

    double const denom_sq{denom * denom};
    for(size_t j = 0; j < n_vars(); ++j){
      for(size_t i = 0; i < j; ++i){
        double entry{lp_exp[i] * lp_exp[j] / denom_sq};
        hess[i + j * n_vars()] += entry;
        hess[j + i * n_vars()] += entry;
      }
      hess[j + j * n_vars()] -=
        (denom - lp_exp[j]) * lp_exp[j] / denom_sq;
    }
  }

  // compute hess <- Sigma_chol.hess.Sigma_chol^T
  {
    int const m = n_vars();
    constexpr double const alpha{1};
    constexpr char const c_R{'R'}, c_U{'U'}, c_N{'N'}, c_T{'T'}, c_L{'L'};
    F77_CALL(dtrmm)
      (&c_L, &c_U, &c_N, &c_N, &m, &m, &alpha, Sigma_chol.memptr(), &m,
       hess, &m, 1, 1, 1, 1);
    F77_CALL(dtrmm)
      (&c_R, &c_U, &c_T, &c_N, &m, &m, &alpha, Sigma_chol.memptr(), &m,
       hess, &m, 1, 1, 1, 1);
  }
}

template class mixed_mult_logit_term<false>;
template class mixed_mult_logit_term<true>;

} // namespace ghqCpp
