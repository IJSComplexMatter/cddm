#!/bin/bash

xargs <.coverage_examples.txt -I filename coverage run -a  "filename"
coverage run -a -m pytest cddm --doctest-modules