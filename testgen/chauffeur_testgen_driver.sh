#!/bin/bash
mkdir result
mv result/chauffeur_rq2_70000_images.csv result/chauffeur_rq2_70000_images.csv.bak
touch result/chauffeur_rq2_70000_images.csv
for i in `seq 0 141`;
do
        echo $i
        python chauffeur_testgen_coverage.py --index $i --dataset $2
done    
