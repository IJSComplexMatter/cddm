#!/bin/bash

pushd examples
xargs <.coverage_examples.txt -I filename coverage run -a  "filename"
popd
mv examples/.coverage .coverage
coverage run -a -m pytest cddm --doctest-modules