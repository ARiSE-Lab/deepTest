#!/bin/bash
mkdir result
mv result/epoch_coverage_70000_images.csv result/epoch_coverage_70000_images.csv.bak
touch result/epoch_coverage_70000_images.csv
for i in `seq 0 141`;
do
        echo $i
        python epoch_testgen_coverage.py --index $i --dataset $1
done    
