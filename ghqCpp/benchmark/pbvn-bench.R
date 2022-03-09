library(mvtnorm)
library(ghqCpp)

# compute relative errors for different configurations with the two
# implementations
comp_errors <- \(method)
sapply(seq_len(1e4), \(i){
  set.seed(i)
  truth <- -Inf
  while(!is.finite(truth) || truth < 1e-12){
    Sigma <- rWishart(1, 2, diag(2)) |> drop()
    mu <- rnorm(2)
    truth <- pmvnorm(upper = numeric(2), mean = mu, sigma = Sigma)
  }

  # compute the error and return
  est <- pbvn(mu, Sigma, method = method)
  c(absolute = est - truth, relative = (est - truth) / truth, truth = truth)
}, simplify = "array")

errs_method0 <- comp_errors(0)
errs_method1 <- comp_errors(1)

# look at stats for the absolute errors
abs(errs_method0[c("absolute", "relative"), ]) |>
  apply(1, quantile,
        probs = seq(0, 1, length.out = 21) |> c(.99, .999) |> sort())
abs(errs_method1[c("absolute", "relative"), ]) |>
  apply(1, quantile,
        probs = seq(0, 1, length.out = 21) |> c(.99, .999) |> sort())

# compare the computation time
set.seed(1)
Sigma <- rWishart(1, 2, diag(2)) |> drop()
mu <- rnorm(2)

bench::mark(
  `pbvn method 0` = pbvn(mu, Sigma, method = 0),
  `pbvn method 1` = pbvn(mu, Sigma, method = 1),
  check = FALSE)

# check the gradient
library(numDeriv)
errs <- sapply(seq_len(1e4), \(i){
  set.seed(i)
  truth <- -Inf
  while(!is.finite(truth) || truth < 1e-12){
    Sigma <- rWishart(1, 2, diag(2)) |> drop()
    mu <- rnorm(2)
    truth <- pmvnorm(upper = numeric(2), mean = mu, sigma = Sigma)
  }

  # compute the gradient numerically of the log of the interval
  fn <- pmvnorm(upper = numeric(2), mean = mu, sigma = Sigma)
  num_grad <- grad(
    \(par) {
      mu <- head(par, 2)
      S <- matrix(nrow = 2, ncol = 2)
      S[upper.tri(S, TRUE)] <- tail(par, 3)
      S[lower.tri(S)] <- t(S)[lower.tri(S)]

      pmvnorm(upper = numeric(2), mean = mu, sigma = S)
    }, c(mu, Sigma[upper.tri(Sigma, TRUE)]), method.args = list(r = 6))

  # compute the gradient, compute the error and return
  est <- pbvn_grad(mu = mu, Sigma = Sigma)

  # only keep the upper triangle
  d_Sig <- tail(est, 4) |> matrix(nrow = 2)
  d_Sig[upper.tri(d_Sig)] <- 2 * d_Sig[upper.tri(d_Sig)]
  val <- attr(est, "prob")
  est <- c(val, c(head(est, 2), d_Sig[upper.tri(d_Sig, TRUE)]) / val)

  truth <- c(fn, num_grad / fn) |>
    setNames(c("integral", "d mu1", "d mu2", "d Sig11", "d Sig12",
               "d Sig22"))
  err <- est - truth
  relative = ifelse(abs(truth) < 1e-12, err, err / abs(truth))

  cbind(absolute = err, relative = relative, truth = truth, est,
        rho = cov2cor(Sigma)[1, 2])
}, simplify = "array")

# quantiles of the absolute error
apply(abs(errs[, "absolute", ]),
      1, quantile, probs = seq(0, 1, length.out = 21) |> c(.99, .999) |> sort())

# quantiles of the relative error
apply(abs(errs[, "relative", ]),
      1, quantile, probs = seq(0, 1, length.out = 21) |> c(.99, .999) |> sort())

# check the computation time
bench::mark(pbvn_grad(mu = mu, Sigma = Sigma))

# # plot the errors versus the integral value
# vals <- errs["integral", "truth", ]
#
# par(mar = c(5, 5, 1, 1))
# plot(vals, errs["d mu1", "relative", ], type = "h")
# plot(vals, errs["d Sig12", "relative", ], type = "h")
# plot(vals, errs["d mu1", "absolute", ], type = "h")
# plot(vals, errs["d Sig12", "absolute", ], type = "h")
#
# rhos <- errs["integral", "rho", ]
# plot(rhos, errs["d mu1", "relative", ], type = "h")
# plot(rhos, errs["d Sig12", "relative", ], type = "h")
#
# # may fail with a large absolute correlation coefficient
# plot(rhos, errs["integral", "absolute", ], type = "h")
# plot(rhos, errs["d mu1", "absolute", ], type = "h")
# plot(rhos, errs["d mu2", "absolute", ], type = "h")
# plot(rhos, errs["d Sig11", "absolute", ], type = "h")
# plot(rhos, errs["d Sig12", "absolute", ], type = "h")
# plot(rhos, errs["d Sig22", "absolute", ], type = "h")
#
#
# local({
#   keep <- abs(rhos) < .99
#   rhos <- rhos[keep]
#   errs <- errs[, , keep]
#
#   plot(rhos, errs["integral", "absolute", ], type = "h")
#   plot(rhos, errs["d mu1", "absolute", ], type = "h")
#   plot(rhos, errs["d mu2", "absolute", ], type = "h")
#   plot(rhos, errs["d Sig11", "absolute", ], type = "h")
#   plot(rhos, errs["d Sig12", "absolute", ], type = "h")
#   plot(rhos, errs["d Sig22", "absolute", ], type = "h")
# })
