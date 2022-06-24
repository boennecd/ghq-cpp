eta <- matrix(c(-0.231792563572526, 0.539682839997113, -0.00460151582956314, 0.435237016528845, 0.983812189660966, -0.239929641131312, 0.554890442639589, 0.869410462211818, -0.575714957434684, 0.303347532171756, -0.748889808077365, -0.465558662544936), 3)
Sigma <- matrix(c(1.07173376588632, 0.760530258851724, -0.920427236518008, 0.760530258851724, 3.4214999078618, -1.56325086522103, -0.920427236518008, -1.56325086522103, 2.44510218991128), 3)
which_cat <- 1:4 - 1L
n <- NCOL(Sigma)

library(ghqCpp)

dat <- fastGHQuad::gaussHermiteData(10)

microbenchmark::microbenchmark(
  integral =
    mixed_mult_logit_term(
      eta, Sigma, which_cat, nodes = dat$x, weights = dat$w,
      use_adaptive = FALSE),
  `integral adaptive` =
    mixed_mult_logit_term(
      eta, Sigma, which_cat, nodes = dat$x, weights = dat$w,
      use_adaptive = TRUE),
  grad =
    mixed_mult_logit_term_grad(
      eta, Sigma, which_cat, nodes = dat$x, weights = dat$w,
      use_adaptive = FALSE),
  `grad adaptive` =
    mixed_mult_logit_term_grad(
      eta, Sigma, which_cat, nodes = dat$x, weights = dat$w,
      use_adaptive = TRUE),
  times = 10000)
