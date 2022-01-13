#!/bin/bash
R --no-save --no-restore -e "Rcpp::sourceCpp('ghq-cpp.cpp')" > out.log