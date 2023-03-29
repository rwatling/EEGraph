#!/bin/bash

mkdir -p ./build/subgraph-results

inputs=( "./datasets/Google/web-Google-trimmed.txt" 
        "./datasets/LiveJournal/soc-LiveJournal1-trimmed.txt" 
        "./datasets/Pokec/soc-pokec-relationships.txt"
        "./datasets/Road/roadNet-CA-trimmed.txt"
        "./datasets/Skitter/as-skitter-trimmed.txt")

names=("google" "livejournal" "pokec" "road" "skitter")

for j in {1..6}
do

    echo $j
    count=0

    for i in "${inputs[@]}"
    do
        file="${names[count]}"
        echo $file
		#randNum=$((RANDOM % 32))
        randNum=0

        ./build/main-subgraph --input "${i}" > "./build/subgraph-results/${j}-${file}"
        sleep 5
        ./build/main-subgraph --input "${i}" --energy true --estats "./build/estats" --efile "./build/efile" > "./build/subgraph-results/${j}-${file}-energy"
        sleep 5

        count=$((count+1))

    done
done
