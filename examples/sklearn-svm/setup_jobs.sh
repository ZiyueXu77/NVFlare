#!/usr/bin/env bash

# setup virtual env.
# todo#
# todo
# todo

# install dependency
# todo

# set up jobs

for value in iris uniform linear
do
  cd "job_configs/sklearn_svm_${value}/app" || exit
  if [ ! -L "custom" ]; then
    cp -r ../../custom .
  fi
  cd - || exit
done


