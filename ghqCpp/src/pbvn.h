#ifndef PBVN_H
#define PBVN_H

#include <RcppArmadillo.h>
#include <array>
#include <Rmath.h> // Rf_dnorm4, Rf_pnorm5 etc.
#ifdef beta
// we get an error if we do not undefine beta
#undef beta
#endif
#include <algorithm>
#include <cmath>
#include <psqn-bfgs.h>
#include <iterator>

namespace ghqCpp {
namespace implementation {
inline double log_exp_add(double const x, double const y){
  double const max_v{std::max(x, y)},
               min_v{std::min(x, y)};
  return max_v + std::log1p(std::exp(min_v - max_v));
}

using std::begin;
using std::end;

inline double dnrm_log(double const x){
  constexpr double sqrt_double_max{4.23992114886859e+153}, // dput(sqrt(.Machine$double.xmax / 10))
                      log_sqrt_2pi{0.918938533204673};
  return x > sqrt_double_max ? -std::numeric_limits<double>::infinity()
                             : -log_sqrt_2pi - x * x / 2;
}

class tiltin_param_problem final : public PSQN::problem {
  static constexpr size_t dim{2};
  double const * const upper_limits;
  double const choleksy_off_diag;

  /// computes C^T.x setting the diagonal of C to zero
  void cholesky_product_T(double res[dim], double const x[dim]){
    res[0] = 0;
    res[1] = x[0] * choleksy_off_diag;
  }

  /// computes C.x setting the diagonal of C to zero
  void cholesky_product(double res[dim], double const x[dim]){
    res[0] = x[1] * choleksy_off_diag;
    res[1] = 0;
  }

  template<bool comp_grad>
  double eval(double const val[dim + 1],
              double        gr[dim + 1]) {
    double const tilt{val[0]};
    double const * const point{val + 1};

    double choleksy_T_point[dim];
    cholesky_product_T(choleksy_T_point, point);

    double derivs_pnrm_terms[dim],
             hess_pnrm_terms[comp_grad ? dim : 0];

    {
      double const ub_shift{upper_limits[0] - choleksy_T_point[0] - tilt},
                  denom_log{Rf_pnorm5(ub_shift, 0, 1, 1, 1)},
                dnrm_ub_log{dnrm_log(ub_shift)},
                    ratio_ub{std::exp(dnrm_ub_log - denom_log)};
      derivs_pnrm_terms[0] = -ratio_ub;
      if constexpr(comp_grad)
        hess_pnrm_terms[0] =
          -ub_shift * ratio_ub - derivs_pnrm_terms[0] * derivs_pnrm_terms[0];
    }
    {
      double const ub_shift{upper_limits[1] - choleksy_T_point[1]},
                  denom_log{Rf_pnorm5(ub_shift, 0, 1, 1, 1)},
                dnrm_ub_log{dnrm_log(ub_shift)},
                   ratio_ub{std::exp(dnrm_ub_log - denom_log)};
      derivs_pnrm_terms[1] = -ratio_ub;
      if constexpr(comp_grad)
        hess_pnrm_terms[1] =
          -ub_shift * ratio_ub - derivs_pnrm_terms[1] * derivs_pnrm_terms[1];
    }

    double gr_params[4];
    gr_params[0] = tilt - point[0] + derivs_pnrm_terms[0];
    gr_params[1] = - point[1] + derivs_pnrm_terms[1];

    double * choleksy_derivs_pnrm_terms = choleksy_T_point;
    cholesky_product(choleksy_derivs_pnrm_terms, derivs_pnrm_terms);

    gr_params[2] = -tilt + choleksy_derivs_pnrm_terms[0];
    gr_params[3] = choleksy_derivs_pnrm_terms[1];

    double const out
      {std::inner_product(begin(gr_params), end(gr_params), gr_params, 0.)};

    if constexpr(comp_grad){
      // TODO: we can avoid explicitly computing the Hessian
      constexpr size_t const dim_hess{2 * dim};
      double hess[dim_hess * dim_hess],
             diff_mat[dim * dim],
             diff_mat_outer[dim * dim];
      std::fill(begin(hess), end(hess), 0);
      std::fill(begin(diff_mat), end(diff_mat), 0);
      std::fill(begin(diff_mat_outer), end(diff_mat_outer), 0);

      for(size_t i = 0; i < dim; ++i)
        hess[i + i *dim_hess] = hess_pnrm_terms[i] + 1;

      diff_mat[dim] = hess_pnrm_terms[1] * choleksy_off_diag;

      for(size_t i = 0; i < dim; ++i){
        for(size_t j = 0; j < i; ++j){
          hess[j + dim + i * dim_hess] = diff_mat[j + i * dim];
          hess[i + (j + dim) * dim_hess] = diff_mat[j + i * dim];
        }
        hess[i + dim + i * dim_hess] = -1;
        hess[i + (i + dim) * dim_hess] = -1;
      }

      double * const hess_last_block{hess + dim * (1 + dim_hess)};
      for(size_t i = 0; i < dim; ++i)
        hess_last_block[i] += diff_mat[i + dim] * choleksy_off_diag;

      std::fill(gr, gr + dim + 1, 0);
      for(size_t i = 0; i < dim_hess; ++i)
        for(size_t j = 0; j < dim_hess; ++j){
          if(j == 1)
            continue;
          else
            gr[j - (j > 0)] += hess[j + i * dim_hess] * gr_params[i];
        }

      std::for_each(gr, gr + dim + 1, [](double &x) { x *= 2; });
    }

    return out;
  }

public:
  tiltin_param_problem
    (double const upper_limits[2], double const choleksy_off_diag):
  upper_limits{upper_limits}, choleksy_off_diag{choleksy_off_diag} { }

  PSQN::psqn_uint size() const {
    return 3;
  }

  double func(double const *val) {
    return eval<false>(val, nullptr);
  }

  double grad(double const * val,
              double       * gr){
    return eval<true>(val, gr);
  }

  void start_val(double res[3]){
    double &tilt{res[0]};
    double * const point{res + 1};

    for(size_t i = 0; i < dim; ++i)
      point[i] = upper_limits[i] - 1;

    // C^(-T).x that is a forward substitution with a unit diagonal matrix
    point[1] -= point[0] * choleksy_off_diag;
    tilt = 0;
  }

  bool is_interior_solution(double const res[3]) {
    double const * const point{res + 1};
    double const choleksy_T_point[]
      { point[0], point[0] * choleksy_off_diag + point[1] };

    bool out{true};
    for(size_t i = 0; i < dim && out; ++i)
      out &= choleksy_T_point[i] <= upper_limits[i];

    return out;
  }
};

} // implementation

struct find_tilting_param_res {
  double tilting;
  bool success, is_interior;
};

/**
 * finds the minimax tilting parameter suggested by
 *
 *   https://doi.org/10.1111/rssb.12162
 *
 * in the one dimensional case.
 */
inline find_tilting_param_res find_tilting_param
  (double const upper_limits[2], double const choleksy_off_diag,
   double const rel_eps){
  double param[3];

  implementation::tiltin_param_problem prob(upper_limits, choleksy_off_diag);
  prob.start_val(param);

  double wk_mem[PSQN::bfgs_n_wmem(3)];
  auto res = PSQN::bfgs(prob, param, wk_mem, rel_eps, 1000L);
  bool const succeeded =
    res.info == PSQN::info_code::converged ||
    res.info == PSQN::info_code::max_it_reached;
  bool const is_interior{prob.is_interior_solution(param)};

  return { succeeded ? param[0] : 0, succeeded, is_interior };
}

/**
 * computes the integral
 *
 *   int_(-inf)^0int_(-inf)^0phi(x; mu, Sigma) dx =
 *     int_(-inf)^(-mu_1)int_(-inf)^(-mu_2)phi(x; 0, Sigma) dx
 *
 * method = 1 yields the method by Drezner further extended by Genz in
 *
 *   https://doi.org/10.1023/B:STCO.0000035304.20635.31
 *
 * method = 0 yields a Gauss–Legendre quadrature based solution. This is less
 * precise and slower.
 *
 * method = 2 gives the same method as with method = 0 but with the minimax
 * tilting method suggested by
 *
 *   https://doi.org/10.1111/rssb.12162
 */
template<int method = 1>
double pbvn(double const *mu, double const *Sigma){
  static_assert(method >= 0 && method <= 2, "method is not implemented");

  if constexpr (method == 0 || method == 2){
    // setup before applying the quadrature rule
    // the, will be, scaled Cholesky decomposition of the covariance matrix
    std::array<double, 3> Sig_chol;

    double const sig1{std::sqrt(Sigma[0])},
                 sig2{std::sqrt(Sigma[3])};
    bool const permuted{-mu[1] / sig2  < -mu[0] / sig1 };
    if(permuted){
      Sig_chol[0] = sig2;
      Sig_chol[1] = Sigma[2] / sig2;
      Sig_chol[2] = std::sqrt(Sigma[0] - Sig_chol[1] * Sig_chol[1]);

    } else {
      Sig_chol[0] = sig1;
      Sig_chol[1] = Sigma[2] / sig1;
      Sig_chol[2] = std::sqrt(Sigma[3] - Sig_chol[1] * Sig_chol[1]);

    }
    if(!std::isfinite(Sig_chol[0]) || !std::isfinite(Sig_chol[2]))
      throw std::invalid_argument("Choleksy decomposition failed");

    // the scaled upper limits to add
    std::array<double, 2> const ubs
      { (permuted ? -mu[1] : -mu[0]) / Sig_chol[0],
        (permuted ? -mu[0] : -mu[1]) / Sig_chol[2] };
    Sig_chol[1] /= Sig_chol[2];

    double tilting_param{0};
    constexpr bool use_tilting{method == 2};
    if constexpr(use_tilting){
      auto res = find_tilting_param(ubs.data(), Sig_chol[1], 1e-8);
      if(res.success)
        tilting_param = res.tilting;
    }

    /* Gauss–Legendre quadrature nodes scale to the interval [0, 1]. I.e.
     n_nodes <- 50L
     stopifnot(n_nodes %% 2L == 0L)
     gq <- SimSurvNMarker::get_gl_rule(n_nodes)
     ord <- order(gq$node)[seq_len(n_nodes %/% 2L)]
     gq <- with(gq, list(node = node[ord], weight = weight[ord]))
     dput((gq$node + 1) / 2)
     log(gq$weight / 2) |> dput()
     */
    constexpr size_t n_nodes{50};
    constexpr double log_nodes[]{-0.000566958480554094, -0.0029884763343341, -0.00734990245553855, -0.0136606912560569, -0.0219333084120698, -0.0321840865300803, -0.0444334263052349, -0.0587059744760452, -0.0750308297011979, -0.0934417873644116, -0.113977629401458, -0.136682465056404, -0.16160612934026, -0.188804647272732, -0.218340773646487, -0.250284620098857, -0.284714383794182, -0.321717195134542, -0.36139010579344, -0.403841243226125, -0.449191163945253, -0.497574445655356, -0.549141568347433, -0.604061147395087, -0.66252259857298, -0.724739337142984, -0.790952642709519, -0.861436361274494, -0.936502669943514, -1.0165092041089, -1.10186795071833, -1.19305645819858, -1.29063212506562, -1.39525063892062, -1.50769009989463, -1.62888306795255, -1.75995987560697, -1.9023083167218, -2.05765774966192, -2.22820066895138, -2.41677374477853, -2.62713704535509, -2.86442316921556, -3.13589769885905, -3.45233204039424, -3.83069547524975, -4.30005539175986, -4.91674093755178, -5.81448547468502, -7.47550794930057},
                   log_weights[]{-6.53322283985511, -5.6899092746998, -5.24094051811074, -4.93500689320544, -4.70413118267991, -4.51989917726375, -4.36770276999406, -4.23903535290967, -4.12850975550265, -4.03250179468945, -3.94845992500549, -3.87452340980434, -3.80929710971246, -3.75171167066661, -3.70093303857589, -3.65630185246153, -3.61729167704724, -3.5834795327207, -3.55452470170788, -3.53015326113214, -3.51014668424804, -3.49433340776322, -3.48258262178658, -3.47479977715334, -3.47092346849629, -3.47092346849629, -3.47479977715334, -3.48258262178658, -3.49433340776322, -3.51014668424804, -3.53015326113214, -3.55452470170788, -3.5834795327207, -3.61729167704724, -3.65630185246153, -3.70093303857589, -3.75171167066661, -3.80929710971246, -3.87452340980434, -3.94845992500549, -4.03250179468945, -4.12850975550265, -4.23903535290967, -4.36770276999406, -4.51989917726375, -4.70413118267991, -4.93500689320544, -5.24094051811074, -5.6899092746998, -6.53322283985511};

    // do the computation
    double out{use_tilting ? -std::numeric_limits<double>::infinity() : 0};
    double const p_outer{Rf_pnorm5(ubs[0] - tilting_param, 0, 1, 1, 1)};
    for(size_t i = 0; i < n_nodes; ++i){
      double const z_outer{
                     Rf_qnorm5(log_nodes[i] + p_outer, 0, 1, 1, 1) +
                       tilting_param},
                   p_inner{
                     Rf_pnorm5(ubs[1] - Sig_chol[1] * z_outer, 0, 1, 1, 1)};

      if constexpr(method == 2){
        out = implementation::log_exp_add
          (out,
           p_inner + log_weights[i] +
             ((tilting_param - 2 * z_outer) * tilting_param / 2));

      } else
        out += std::exp(p_inner + log_weights[i]);
    }

    return use_tilting ? std::exp(p_outer + out)
                       : std::exp(p_outer) * out;
  }

  double const h{mu[0] / std::sqrt(Sigma[0])},
               k{mu[1] / std::sqrt(Sigma[3])};
  double rho{Sigma[1] / std::sqrt(Sigma[0] * Sigma[3])};

  auto pnrm = [](double const x){
    return Rf_pnorm5(x, 0, 1, 1, 0);
  };

  /* Gauss–Legendre quadrature nodes scale to the interval [0, 1]. I.e.
     gq <- SimSurvNMarker::get_gl_rule(12)
     ord <- order(gq$weight)
     dput((gq$node[ord] + 1) / 2)
     dput(gq$weight[ord] / 2)
   */

  constexpr double nodes6[]{0.966234757101576, 0.033765242898424, 0.830604693233132, 0.169395306766868, 0.619309593041598, 0.380690406958402},
                 weights6[]{0.0856622461895852, 0.0856622461895852, 0.180380786524069, 0.180380786524069, 0.233956967286346, 0.233956967286346},
                  nodes12[]{0.99078031712336, 0.00921968287664043, 0.952058628185237, 0.0479413718147626, 0.884951337097152, 0.115048662902848, 0.793658977143309, 0.206341022856691, 0.68391574949909, 0.31608425050091, 0.562616704255734, 0.437383295744266},
                weights12[]{0.0235876681932559, 0.0235876681932559, 0.0534696629976592, 0.0534696629976592, 0.0800391642716731, 0.0800391642716731, 0.101583713361533, 0.101583713361533, 0.116746268269177, 0.116746268269177, 0.124573522906701, 0.124573522906701},
                  nodes20[]{0.996564299592547, 0.00343570040745256, 0.981985963638957, 0.0180140363610431, 0.956117214125663, 0.0438827858743371, 0.919558485911109, 0.0804415140888906, 0.873165953230075, 0.126834046769925, 0.818026840363258, 0.181973159636742, 0.755433500975414, 0.244566499024587, 0.68685304435771, 0.31314695564229, 0.613892925570823, 0.386107074429178, 0.538263260566749, 0.461736739433251},
                weights20[]{0.00880700356957606, 0.00880700356957606, 0.0203007149001935, 0.0203007149001935, 0.0313360241670545, 0.0313360241670545, 0.0416383707883524, 0.0416383707883524, 0.0509650599086202, 0.0509650599086202, 0.0590972659807592, 0.0590972659807592, 0.0658443192245883, 0.0658443192245883, 0.071048054659191, 0.071048054659191, 0.0745864932363019, 0.0745864932363019, 0.0763766935653629, 0.0763766935653629};

  auto wo_border_correction = [&](double const *nodes, double const *weights,
                                  size_t const n_nodes){
    double const offset{h * h + k * k},
                  slope{2 * h * k},
                     ub{std::asin(rho)};

    double out{};
    for(size_t i = 0; i < n_nodes; ++i){
      double const n{ub * nodes[i]},
               sin_n{std::sin(n)};
      out += weights[i] * std::exp
        (-(offset - slope * sin_n) / (2 * (1 - sin_n * sin_n)));
    }
    out *= ub / (2 * M_PI);

    return out + pnrm(-h) * pnrm(-k);
  };

  if(std::abs(rho) <= .3)
    return wo_border_correction(nodes6, weights6, 6);
  else if(std::abs(rho) <= .75)
    return wo_border_correction(nodes12, weights12, 12);
  else if(std::abs(rho) <= .95)
    return wo_border_correction(nodes20, weights20, 20);

  // handle the large absolute correlation

  // computes the indefinite integral
  //   int exp(-b^2/2x^2)(1 + c * x^2 * (1 + d * x^2))
  // TODO: can likely be done a lot smarter
  auto taylor_term = [&](double const x, double const b, double const c,
                         double const d){
    double const x2{x * x},
                 b2{b * b},
                 b4{b2 * b2};
    double out{2 * x * std::exp(-b2 / (2 * x2))};
    out *= (b4 * c * d - b2 * c * (d * x2 + 5) +
      c * x2 * (3 * d * x2 + 5) + 15);

    constexpr double sqrt2pi{2.506628274631};
    out += sqrt2pi * b * (b4 * c * d - 5 * b2 * c + 15) *
      (2 * pnrm(b / x) - 1);

    return out / 30;
  };

  double const s{rho > 0 ? 1. : -1.},
              ub{std::sqrt(1 - rho * rho)},
             shk{s * h * k};

  double const numerator{-(h - s * k) * (h - s * k) / 2},
           exp_m_shk_d_2{std::exp(-shk / 2)};

  double out{};
  for(size_t i = 0; i < 20; ++i){
    double const x{nodes20[i] * ub},
                x2{x * x};
    double tay{1 + (12 - shk) * x2 / 16};
    tay *= (4 - shk) * x2 / 8;
    tay += 1;
    tay *= exp_m_shk_d_2;

    double const sqrt_1_m_x2{std::sqrt(1 - x2)},
                 fn{std::exp(-shk/(1 + sqrt_1_m_x2)) / sqrt_1_m_x2};

    out += weights20[i] * std::exp(numerator / x2) * (fn - tay);
  }
  out *= ub;

  double const b{std::abs(h - s * k)},
               c{(4 - shk) / 8},
               d{(12 - shk) / 16};

  out +=
    exp_m_shk_d_2 * (taylor_term(ub, b, c, d) - taylor_term(0, b, c, d));
  out *= (-s / (2 * M_PI));
  out += s > 0
    ? pnrm(-std::max(h, k))
    : std::max(0., pnrm(-h) - pnrm(k));
  return out;
}

/**
 * computes the derivative of the mean and covariance matrix in of pbvn. For the
 * mean, this is given by
 *
 *   Sigma^(-1)int_(-inf)^0int_(-inf)^0(x - mu)phi(x; mu, Sigma) dx =
 *     Sigma^(-1).int_(-inf)^(-mu_1)int_(-inf)^(-mu_2)x phi(x; 0, Sigma) dx
 *
 * For Sigma, we need to compute
 *
 *   2^(-1)Sigma^(-1)[int_(-inf)^0int_(-inf)^0
 *     ((x - mu).(x - mu)^T - Sigma)phi(x; mu, Sigma) dx]Sigma^(-1) =
 *   2^(-1)Sigma^(-1)[int_(-inf)^(-mu_1)int_(-inf)^(-mu_2)
 *     (x.x^T - Sigma)phi(x; mu, Sigma) dx]Sigma^(-1)
 *
 * the derivatives w.r.t. Sigma are stored as a 2 x 2 matrix ignoring the
 * symmetry. Thus, a 6D array needs to be passed for the gradient.
 */
template<int method = 1, bool comp_d_Sig = true>
double pbvn_grad(double const *mu, double const *Sigma, double *grad){
  static_assert(method == 1 || method == 0, "method is not implemented");
  std::array<double, 3> Sig_chol;

  double const sig1{std::sqrt(Sigma[0])},
               sig2{std::sqrt(Sigma[3])};
  bool const permuted{-mu[1] / sig2  < -mu[0] / sig1 };
  if(permuted){
    Sig_chol[0] = sig2;
    Sig_chol[1] = Sigma[2] / sig2;
    Sig_chol[2] = std::sqrt(Sigma[0] - Sig_chol[1] * Sig_chol[1]);

  } else {
    Sig_chol[0] = sig1;
    Sig_chol[1] = Sigma[2] / sig1;
    Sig_chol[2] = std::sqrt(Sigma[3] - Sig_chol[1] * Sig_chol[1]);

  }
  if(!std::isfinite(Sig_chol[0]) || !std::isfinite(Sig_chol[2]))
    throw std::invalid_argument("Choleksy decomposition failed");

  // the scaled upper limits to add
  std::array<double, 2> const ubs
  { (permuted ? -mu[1] : -mu[0]) / Sig_chol[0],
    (permuted ? -mu[0] : -mu[1]) / Sig_chol[2] };
  double const Sig_12_scaled{Sig_chol[1] / Sig_chol[2]};

  constexpr size_t n_nodes{50};
  constexpr double nodes[]{0.999433202210036, 0.00056679778996449, 0.997015984716045, 0.00298401528395464, 0.992677042024003, 0.00732295797599708, 0.986432192553346, 0.013567807446654, 0.978305477621404, 0.021694522378596, 0.968328309472439, 0.0316716905275611, 0.956539278327896, 0.043460721672104, 0.942983989761807, 0.0570160102381935, 0.927714884714973, 0.072285115285027, 0.910791035429668, 0.0892089645703321, 0.8922779164502, 0.1077220835498, 0.872247151113034, 0.127752848886966, 0.850776234353411, 0.149223765646589, 0.82794823284272, 0.17205176715728, 0.803851463592475, 0.196148536407525, 0.778579152257325, 0.221420847742675, 0.752229072453732, 0.247770927546268, 0.724903167487019, 0.275096832512981, 0.696707155948783, 0.303292844051217, 0.667750122709719, 0.332249877290281, 0.638144096889766, 0.361855903110234, 0.608003618438021, 0.391996381561979, 0.577445294999073, 0.422554705000927, 0.546587350780043, 0.453412649219957, 0.515549169163595, 0.484450830836406},
                 weights[]{0.00145431127657757, 0.00145431127657757, 0.0033798995978727, 0.0033798995978727, 0.00529527419182548, 0.00529527419182548, 0.00719041138074279, 0.00719041138074279, 0.0090577803567447, 0.0090577803567447, 0.0108901215850624, 0.0108901215850624, 0.0126803367850062, 0.0126803367850062, 0.0144214967902676, 0.0144214967902676, 0.016106864111789, 0.016106864111789, 0.0177299178075731, 0.0177299178075731, 0.0192843783062938, 0.0192843783062938, 0.0207642315450738, 0.0207642315450738, 0.0221637521694016, 0.0221637521694016, 0.0234775256519742, 0.0234775256519742, 0.0247004692247332, 0.0247004692247332, 0.0258278515347906, 0.0258278515347906, 0.0268553109444981, 0.0268553109444981, 0.0277788724031063, 0.0277788724031063, 0.0285949628238642, 0.0285949628238642, 0.0293004249066112, 0.0293004249066112, 0.0298925293521327, 0.0298925293521327, 0.0303689854208851, 0.0303689854208851, 0.0307279497951583, 0.0307279497951583, 0.0309680337103416, 0.0309680337103416, 0.0310883083276736, 0.0310883083276736};

  // do the computation
  double out{};
  std::fill(grad, comp_d_Sig ? grad + 6 : grad + 2, 0);
  double * const d_mu{grad},
         * const d_Sig{comp_d_Sig ? grad + 2 : nullptr};
  double const p_outer{Rf_pnorm5(ubs[0], 0, 1, 1, 0)};

  for(size_t i = 0; i < n_nodes; ++i){
    double const z_outer{Rf_qnorm5(nodes[i] * p_outer, 0, 1, 1, 0)},
             u_lim_inner{ubs[1] - Sig_12_scaled * z_outer},
                 p_inner{Rf_pnorm5(u_lim_inner, 0, 1, 1, 0)};
    if(method == 0)
      out += p_inner * weights[i];

    double const g1_fac{z_outer * p_inner},
      dnorm_u_lim_inner{Rf_dnorm4(u_lim_inner, 0, 1, 0)},
      trunc_mean_scaled{-dnorm_u_lim_inner};
    grad[0] += weights[i] * g1_fac;
    grad[1] += weights[i] * trunc_mean_scaled;

    if constexpr (comp_d_Sig){
      d_Sig[0] += weights[i] * g1_fac * z_outer;
      double const off_diag{z_outer * trunc_mean_scaled};
      d_Sig[1] += weights[i] * off_diag;
      double const trunc_sq_moment_scaled
        {p_inner - dnorm_u_lim_inner * u_lim_inner};
      d_Sig[3] += weights[i] * trunc_sq_moment_scaled;
    }
  }

  if(method == 1)
    out = pbvn<method>(mu, Sigma);
  else
    out *= p_outer;

  // handle the derivatives w.r.t. mu
  std::for_each(d_mu, d_mu + 2, [&](double &x){ x *= p_outer; });

  // performs backward substitution
  auto back_sub = [&](double *x){
    x[1] /= Sig_chol[2];
    x[0] = (x[0] - Sig_chol[1] * x[1]) / Sig_chol[0];
  };

  back_sub(d_mu);

  // possibly handle the derivatives w.r.t Sigma
  if constexpr (comp_d_Sig){
    d_Sig[2] = d_Sig[1]; // symmetry
    std::for_each(d_Sig, d_Sig + 4,
                  [&](double &x){ x *= p_outer / 2; });

    // subtract the identity matrix in the diagonal
    d_Sig[0] -= out / 2;
    d_Sig[3] -= out / 2;

    back_sub(d_Sig);
    back_sub(d_Sig + 2);
    std::swap(d_Sig[1], d_Sig[2]); // transpose
    back_sub(d_Sig);
    back_sub(d_Sig + 2);
  }

  if(permuted){
    std::swap(grad[0], grad[1]); // d_mu
    if constexpr (comp_d_Sig)
      std::swap(grad[2], grad[5]); // d_Sigma
  }

  return out;
}

/// computes the Hessian w.r.t. mu. Thus, a 4D array has to be passed
template<int method = 1>
void pbvn_hess(double const *mu, double const *Sigma, double *hess){
  double gr[6];
  pbvn_grad<method, true>(mu, Sigma, gr);

  arma::mat Sig(const_cast<double *>(Sigma), 2, 2, false);
  for(unsigned j = 0; j < 2; ++j)
    for(unsigned i = 0; i < 2; ++i)
      hess[i + j * 2] = 2 * gr[i + j * 2 + 2];
}

} // namespace ghqCpp

#endif
