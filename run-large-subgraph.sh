#!/bin/bash

mkdir -p ./build/subgraph-large-results

inputs=( "/home/share/graph_data/raw/twitter_mpi/twitter.el"
        "/home/share/graph_data/raw/friendster_snap/fs.el"
        "/home/share/graph_data/raw/twitter_www/twitter.www.el")

names=("tw" "fs" "tw2")

for j in {1..4}
do

    echo $j
    count=0

    for i in "${inputs[@]}"
    do
        file="${names[count]}"
        echo $file
		#randNum=$((RANDOM % 32))
        randNum=0

        ./build/main-subgraph --input "${i}" > "./build/subgraph-large-results/${j}-${file}"
        sleep 5
        ./build/main-subgraph --input "${i}" --energy true --estats "./build/estats" --efile "./build/efile" > "./build/subgraph-large-results/${j}-${file}-energy"
        sleep 5

        count=$((count+1))

    done
done
