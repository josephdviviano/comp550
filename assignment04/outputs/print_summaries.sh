#!/bin/bash

for f in *.txt; do
    echo "${f}:"
    echo ''
    cat ${f}
    echo " "
done
