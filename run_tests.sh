#!/bin/bash

# we have to cd to examples folder so that we change CWD
# examples need to be executed one by one (order is important)
pushd examples
xargs <.coverage_examples.txt -I filename coverage run -a  "filename"
# go back to root
popd
#we are adding to coverage results, so move these in root dir
mv examples/.coverage .coverage
coverage run -a -m pytest cddm --doctest-modules