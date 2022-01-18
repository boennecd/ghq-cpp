#include <testthat.h>
#include "integrand-expected-survival.h"
#include <iterator>

// the test data
namespace {
arma::vec const etas{-0.6381, -0.6343, -0.6285, -0.6227, -0.6193, -0.6212, -0.6313, -0.6515, -0.6827, -0.724, -0.7723, -0.8229, -0.8699, -0.9073, -0.93},
                  ws{0.0473878636519986, 0.10842748431263, 0.165122315533235, 0.215065333801771, 0.256205263034596, 0.286856653590245, 0.305764321437458, 0.312154084712178, 0.305764321437458, 0.286856653590245, 0.256205263034596, 0.215065333801771, 0.165122315533235, 0.10842748431263, 0.0473878636519986};

constexpr size_t n_vars{5}, n_lps{15};
arma::mat const M
  {
    ([]{
      arma::mat out{0.3115, 0.312, 0.3128, 0.3141, 0.3156, 0.3173, 0.3192, 0.3212, 0.3232, 0.3251, 0.3269, 0.3284, 0.3296, 0.3305, 0.331, -0.1315, -0.1378, -0.1488, -0.1641, -0.1831, -0.205, -0.2288, -0.2537, -0.2786, -0.3024, -0.3243, -0.3433, -0.3586, -0.3696, -0.3758, -0.4, -0.4, -0.4, -0.4, -0.4, -0.4, -0.4, -0.4, -0.4, -0.4, -0.4, -0.4, -0.4, -0.4, -0.4, -1.2253, -1.1941, -1.1392, -1.0629, -0.9683, -0.8593, -0.7404, -0.6164, -0.4924, -0.3734, -0.2644, -0.1699, -0.0936, -0.0387, -0.0074, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
      out.reshape(n_lps, n_vars);
      return out;
    })()
  };
arma::mat const V
  {
    ([]{
      arma::mat out{0.35, 0.08, -0.05, 0.01, 0, 0.08, 1.92, -0.24, -0.04, 0, -0.05, -0.24, 0.32, 0.09, 0, 0.01, -0.04, 0.09, 0.12, 0, 0, 0, 0, 0, 0.04};
      out.reshape(n_vars, n_vars);
      return out;
    })()
  };
} // namespace

/* the test data in R
 etas <- c(-0.6381, -0.6343, -0.6285, -0.6227, -0.6193, -0.6212, -0.6313, -0.6515, -0.6827, -0.724, -0.7723, -0.8229, -0.8699, -0.9073, -0.93)
 ws <- c(0.0473878636519986, 0.10842748431263, 0.165122315533235, 0.215065333801771, 0.256205263034596, 0.286856653590245, 0.305764321437458, 0.312154084712178, 0.305764321437458, 0.286856653590245, 0.256205263034596, 0.215065333801771, 0.165122315533235, 0.10842748431263, 0.0473878636519986)
 M <- structure(c(0.3115, 0.312, 0.3128, 0.3141, 0.3156, 0.3173, 0.3192, 0.3212, 0.3232, 0.3251, 0.3269, 0.3284, 0.3296, 0.3305, 0.331, -0.1315, -0.1378, -0.1488, -0.1641, -0.1831, -0.205, -0.2288, -0.2537, -0.2786, -0.3024, -0.3243, -0.3433, -0.3586, -0.3696, -0.3758, -0.4, -0.4, -0.4, -0.4, -0.4, -0.4, -0.4, -0.4, -0.4, -0.4, -0.4, -0.4, -0.4, -0.4, -0.4, -1.2253, -1.1941, -1.1392, -1.0629, -0.9683, -0.8593, -0.7404, -0.6164, -0.4924, -0.3734, -0.2644, -0.1699, -0.0936, -0.0387, -0.0074, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1), .Dim = c(15L, 5L))
 V <- structure(c(0.35, 0.08, -0.05, 0.01, 0, 0.08, 1.92, -0.24, -0.04, 0, -0.05, -0.24, 0.32, 0.09, 0, 0.01, -0.04, 0.09, 0.12, 0, 0, 0, 0, 0, 0.04), .Dim = c(5L, 5L))
 n <- NCOL(V)
 */

context("expected_survival_term works as expected") {
  test_that("log_integrand, log_integrand_grad, and log_integrand_x works") {
    /*
     log_integrand <- \(x){
     x <- crossprod(chol(V), x)
     x <- M %*% x |> drop()
     -sum(ws * exp(x + etas))
     }
     set.seed(1)
     dput(point <- runif(n, -1))
     dput(log_integrand(point))
     dput(numDeriv::grad(log_integrand, point))
     dput(numDeriv::hessian(log_integrand, point))
     */
    constexpr double point[]{-0.4689826737158, -0.25575220072642, 0.145706726703793, 0.816415579989552, -0.596636137925088},
                   true_func{-1.10080075347693},
                   true_gr[]{-0.197160146168177, 0.296478606049248, 0.342189520188315, 0.201433400716935, -0.220160150705414},
                 true_hess[]{-0.0353126608783229, 0.053070744397719, 0.0613042167821518,  0.0361080494990776, -0.0394320292347374, 0.053070744397719, -0.0925309789513749,  -0.0854928729080763, -0.0416930675300463, 0.0592957212053608,  0.0613042167821518, -0.0854928729080763, -0.109878789901267,  -0.0692217454435289, 0.0684379040230279, 0.0361080494990776,  -0.0416930675300463, -0.0692217454435289, -0.0492986185590044,  0.0402866801416676, -0.0394320292347374, 0.0592957212053608,  0.0684379040230279, 0.0402866801416676, -0.0440320301326317};

    simple_mem_stack<double> mem;
    expected_survival_term prob(etas, ws, M, V);

    expect_true
      (std::abs(prob.log_integrand(point, mem) - true_func) <
        std::abs(true_func) * 1e-6);

    constexpr double eps{1e-6};
    double gr_val[n_vars];
    expect_true
      (std::abs(prob.log_integrand_grad(point, gr_val, mem) - true_func) <
        std::abs(true_func) * eps);
    for(size_t i = 0; i < n_vars; ++i)
      expect_true(std::abs(gr_val[i] - true_gr[i]) < std::abs(true_gr[i]));

    double hess_val[n_vars * n_vars];
    prob.log_integrand_hess(point, hess_val, mem);
    for(size_t i = 0; i < n_vars * n_vars; ++i)
      expect_true
        (std::abs(hess_val[i] - true_hess[i]) < std::abs(true_hess[i]));
  }

  test_that("eval works and so does the gradient") {
    /*
     set.seed(1)
     brute_ests <- apply(mvtnorm::rmvnorm(1e7, sigma = V), 1L, \(u){
     x <- M %*% u |> drop()
     -sum(ws * exp(x + etas)) |> exp()
     })
     dput(3 * sd(brute_ests) / sqrt(length(brute_ests)))
     dput(mean(brute_ests))

     dput(gl <- fastGHQuad::gaussHermiteData(6))

     fn <- \(x){
     e <- x[seq_along(etas)]
     x <- x[-seq_along(etas)]
     M <- x[seq_along(M)] |> matrix(nrow = NROW(M))

     S <- matrix(0, NROW(V), NROW(V))
     S[upper.tri(V, TRUE)] <- tail(x, -length(M))
     S[lower.tri(S)] <- t(S)[lower.tri(S)]

     ghqCpp::expected_survival_term(
     eta = e, ws = ws, M = M, Sigma = S,
     weights = gl$w, nodes = gl$x)
     }
     num_grad <- numDeriv::grad(fn, c(etas, M, V[upper.tri(V, TRUE)]))

     d_V <- matrix(0, NROW(V), NCOL(V))
     d_V[upper.tri(V, TRUE)] <- tail(num_grad, 5 * 3)
     d_V[upper.tri(V)] <- d_V[upper.tri(V)] / 2
     d_V[lower.tri(V)] <- t(d_V)[lower.tri(V)]

     c(head(num_grad, -5 * 3), d_V) |> dput()
     */
    constexpr double true_fn{0.232477957568917},
                      eps_fn{0.000145864854007574},
                   true_gr[]{-0.00470068919427127, -0.0107629774323235, -0.0164045927647716, -0.0213645879440759, -0.0253943014427888, -0.0282583194355553, -0.0297633823546434, -0.02981159289582, -0.0284257075286721, -0.0257704924559485, -0.0221299039069513, -0.0178333392212364, -0.0131844834114841, -0.00840147875792821, -0.00360551341569692, 0.000256571567787004, 0.000586391052635518, 0.000891191605992053, 0.00115473558036259, 0.00136500225025227, 0.00150977500103163, 0.00157912314866448, 0.00157048379948513, 0.00148706171244333, 0.00133920135703345, 0.0011429447560898, 0.000916597018888391, 0.000675041592144285, 0.000428860198838876, 0.000183718732337745, -0.00191468382556292, -0.00424233917471771, -0.00608857715164614, -0.00724641429783803, -0.00760514172437183, -0.0071702835170275, -0.00607250038800452, -0.00453316357584309, -0.00284585450512053, -0.00130018275706299, -0.000106421116767668, 0.000620249872243076, 0.00087894823638763, 0.000752628302794794, 0.000369644410813391, 1.44565271753843e-05, -9.68973983661189e-06, -0.000129223210192829, -0.000375502305777943, -0.000752207638930474, -0.00122963562146454, -0.0017458025543791, -0.00222048048276578, -0.00256759899881634, -0.00271927066013012, -0.00264390607569465, -0.00234658811605965, -0.00186374774838956, -0.00124671373366828, -0.00054942162410276, 9.3221160980128e-05, 0.000173203360067087, 0.000156155010406422, 8.00211332928442e-06, -0.000278544199225233, -0.000679526354730239, -0.00114047941079771, -0.00158651939602583, -0.00193662760916597, -0.00212460548404177, -0.00211497852781407, -0.0019074351520065, -0.00153144924674956, -0.00103149406804648, -0.000456267158378686, 9.73048151371228e-05, 0.000222323935439903, 0.000337579970793631, 0.000437379408245862, 0.000516577448084777, 0.000570679782181458, 0.00059635907018704, 0.000592536470591176, 0.000560524101302998, 0.000504328090050606, 0.000430161533532258, 0.00034464163464313, 0.000253614779988949, 0.000161073589700659, 6.89912518917497e-05, 0.00731542921873622, -0.00558434154765034, -0.00913009895978359, -0.01509780820023, 0.0228252626060564, -0.00558434154765034, 0.0036624115833844, 0.00703338417624582, 0.0148181477486294, -0.0175834573148362, -0.00913009895978359, 0.00703338417624582, 0.0113881274144034, 0.0184930306288589, -0.0284702462443964, -0.01509780820023, 0.0148181477486294, 0.0184930306288589, 0.0131001149132894, -0.0462328174386099, 0.0228252626060564, -0.0175834573148362, -0.0284702462443964, -0.0462328174386099, 0.071175946461613};
    constexpr double ghq_nodes[]{-2.35060497367449, -1.3358490740137, -0.436077411927617, 0.436077411927617, 1.3358490740137, 2.35060497367449},
                   ghq_weights[]{0.00453000990550887, 0.157067320322856, 0.724629595224392, 0.724629595224392, 0.157067320322856, 0.00453000990550883};

    simple_mem_stack<double> mem;
    ghq_data dat{ghq_nodes, ghq_weights, 6};

    {
      expected_survival_term<false> prob(etas, ws, M, V);
      adaptive_problem prob_adap(prob, mem);

      auto res = ghq(dat, prob, mem);
      expect_true(res.size() == 1);
      expect_true(std::abs(res[0] - true_fn) < eps_fn);
    }
    {
      expected_survival_term<true> surv_term(etas, ws, M, V);
      outer_prod_problem outer_term(V.n_cols);

      std::vector<ghq_problem const *> const prob_dat
          { &surv_term, &outer_term };
      combined_problem prob(prob_dat);
      adaptive_problem prob_adap(prob, mem);

      auto res = ghq(dat, prob, mem);

      // handle the derivatives w.r.t. Sigma
      size_t const fixef_shift{surv_term.n_out()};
      std::vector<double> out(fixef_shift + V.n_cols * V.n_cols);
      std::copy(res.begin(), res.begin() + fixef_shift, &out[0]);
      outer_term.d_Sig
        (&out[fixef_shift], res.data() + fixef_shift, res[0], V);

      size_t const n_grad =
        std::distance(std::begin(true_gr), std::end(true_gr));
      expect_true(out.size() == 1 + n_grad);

      expect_true(std::abs(out[0] - true_fn) < eps_fn);
      for(size_t i = 0; i < n_grad; ++i)
        expect_true
          (std::abs(out[i + 1] - true_gr[i]) < 1e-3 * std::abs(true_gr[i]));
    }
  }
}