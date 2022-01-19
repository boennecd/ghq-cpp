#!/bin/bash
R --no-save --no-restore -e "Rcpp::compileAttributes()"
cd ..
R --no-init-file CMD INSTALL --preclean --no-multiarch --with-keep.source ghqCpp
