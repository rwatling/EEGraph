#!/bin/bash

mkdir -p ./data/

mkdir -p ./data/subway

mkdir -p ./data/subway/bfs
mkdir -p ./data/subway/bfs/async
mkdir -p ./data/subway/bfs/sync

mkdir -p ./data/subway/cc
mkdir -p ./data/subway/cc/async
mkdir -p ./data/subway/cc/sync

mkdir -p ./data/subway/pr
mkdir -p ./data/subway/pr/async
mkdir -p ./data/subway/pr/sync

mkdir -p ./data/subway/sssp
mkdir -p ./data/subway/sssp/async
mkdir -p ./data/subway/sssp/sync

mkdir -p ./data/subway/sswp
mkdir -p ./data/subway/sswp/async
mkdir -p ./data/subway/sswp/sync

mkdir -p ./data/subway-energy

mkdir -p ./data/subway-energy/bfs
mkdir -p ./data/subway-energy/bfs/async
mkdir -p ./data/subway-energy/bfs/sync

mkdir -p ./data/subway-energy/cc
mkdir -p ./data/subway-energy/cc/async
mkdir -p ./data/subway-energy/cc/sync

mkdir -p ./data/subway-energy/pr
mkdir -p ./data/subway-energy/pr/async
mkdir -p ./data/subway-energy/pr/sync

mkdir -p ./data/subway-energy/sssp
mkdir -p ./data/subway-energy/sssp/async
mkdir -p ./data/subway-energy/sssp/sync

mkdir -p ./data/subway-energy/sswp
mkdir -p ./data/subway-energy/sswp/async
mkdir -p ./data/subway-energy/sswp/sync

inputs=("../datasets/Google/web-Google-trimmed.txt" 
        "../datasets/LiveJournal/soc-LiveJournal1-trimmed.txt"
        "../datasets/Road/roadNet-CA-trimmed.txt"
        "../datasets/Skitter/as-skitter-trimmed.txt"
        "../datasets/Pokec/soc-pokec-relationships.txt")

names=("google" "lj" "road" "skitter" "pokec")

for i in {1..5}
do

    echo $i
    count=0

    for j in "${inputs[@]}"
    do
        file="${names[count]}"
        echo $file

        #./bfs-async --input "${j}" > "./data/subway/bfs/async/${i}-${file}"
        #./bfs-sync --input "${j}" > "./data/subway/bfs/sync/${i}-${file}"
        ./bfs-async --input "${j}" --energy true --efile "./data/subway-energy/bfs/async/${i}-${file}-readings" --estats "./data/subway-energy/bfs/async/${i}-${file}-stats" > "./data/subway-energy/bfs/async/${i}-${file}"
        ./bfs-sync --input "${j}" --energy true --efile "./data/subway-energy/bfs/sync/${i}-${file}-readings" --estats "./data/subway-energy/bfs/sync/${i}-${file}-stats" > "./data/subway-energy/bfs/sync/${i}-${file}"
        
        #./cc-async --input "${j}" > "./data/subway/cc/async/${i}-${file}"
        #./cc-sync --input "${j}" > "./data/subway/cc/sync/${i}-${file}"
        ./cc-async --input "${j}" --energy true --efile "./data/subway-energy/cc/async/${i}-${file}-readings" --estats "./data/subway-energy/cc/async/${i}-${file}-stats" > "./data/subway-energy/cc/async/${i}-${file}"
        ./cc-sync --input "${j}" --energy true --efile "./data/subway-energy/cc/sync/${i}-${file}-readings" --estats "./data/subway-energy/cc/sync/${i}-${file}-stats" > "./data/subway-energy/cc/sync/${i}-${file}"

        #./pr-async --input "${j}" > "./data/subway/pr/async/${i}-${file}"
        #./pr-sync --input "${j}" > "./data/subway/pr/sync/${i}-${file}"
        ./pr-async --input "${j}" --energy true --efile "./data/subway-energy/pr/async/${i}-${file}-readings" --estats "./data/subway-energy/pr/async/${i}-${file}-stats" > "./data/subway-energy/pr/async/${i}-${file}"
        ./pr-sync --input "${j}" --energy true --efile "./data/subway-energy/pr/sync/${i}-${file}-readings" --estats "./data/subway-energy/pr/sync/${i}-${file}-stats" > "./data/subway-energy/pr/sync/${i}-${file}"

        #./sssp-async --input "${j}" > "./data/subway/sssp/async/${i}-${file}"
        #./sssp-sync --input "${j}" > "./data/subway/sssp/sync/${i}-${file}"
        ./sssp-async --input "${j}" --energy true --efile "./data/subway-energy/sssp/async/${i}-${file}-readings" --estats "./data/subway-energy/sssp/async/${i}-${file}-stats" > "./data/subway-energy/sssp/async/${i}-${file}"
        ./sssp-sync --input "${j}" --energy true --efile "./data/subway-energy/sssp/sync/${i}-${file}-readings" --estats "./data/subway-energy/sssp/sync/${i}-${file}-stats" > "./data/subway-energy/sssp/sync/${i}-${file}"
        
        #./sswp-async --input "${j}" > "./data/subway/sswp/async/${i}-${file}"
        #./sswp-sync --input "${j}" > "./data/subway/sswp/sync/${i}-${file}"
        ./sswp-async --input "${j}" --energy true --efile "./data/subway-energy/sswp/async/${j}-${file}-readings" --estats "./data/subway-energy/sswp/async/${j}-${file}-stats" > "./data/subway-energy/sswp/async/${i}-${file}"
        ./sswp-sync --input "${j}" --energy true --efile "./data/subway-energy/sswp/sync/${i}-${file}-readings" --estats "./data/subway-energy/sswp/sync/${i}-${file}-stats" > "./data/subway-energy/sswp/sync/${i}-${file}"
        
        count=$((count+1))
    done
done
