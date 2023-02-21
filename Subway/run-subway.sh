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

inputs=("../datasets/Google/web-Google-trimmed.txt", 
        "../datasets/LiveJournal/soc-LiveJournal1-trimmed.txt", 
        "../datasets/Orkut/orkut-trimmed.el", 
        "../datasets/Pokec/soc-pokec-relationships.txt", 
        "../datasets/Road/roadNet-CA-trimmed.txt", 
        "../datasets/Skitter/as-skitter-trimmed.txt")

names=("google" "livejournal" "orkut" "pokec" "road" "skitter")

for i in {1..3}
do

    echo $i
    count=0

    for j in "${inputs[@]}"
    do
        file="${names[count]}"
        echo $file
		#randNum=$((RANDOM % 32))
        randNum=0

        ./bfs-async --input "${j}" --source "${randNum}" > "./data/subway/bfs/async/${i}-${file}"
        sleep 5
        ./bfs-sync--input "${j}" --source "${randNum}" > "./data/subway/bfs/sync/${i}-${file}"
        sleep 5
        ./bfs-async --input "${j}" --source "${randNum}" --energy true --efile "./data/subway-energy/bfs/async/${j}-${file}-readings" --estats "./data/subway-energy/bfs/async/${j}-${file}-stats"> "./data/subway-energy/bfs/async/${i}-${file}"
        sleep 5
        ./bfs-sync --input "${j}" --source "${randNum}" --energy true --efile "./data/subway-energy/bfs/sync/${j}-${file}-readings" --estats "./data/subway-energy/bfs/sync/${j}-${file}-stats"> "./data/subway-energy/bfs/sync/${i}-${file}"
        sleep 5

        ./cc-async --input "${j}" --source "${randNum}" > "./data/subway/cc/async/${i}-${file}"
        sleep 5
        ./cc-sync--input "${j}" --source "${randNum}" > "./data/subway/cc/sync/${i}-${file}"
        sleep 5
        ./cc-async --input "${j}" --source "${randNum}" --energy true --efile "./data/subway-energy/cc/async/${j}-${file}-readings" --estats "./data/subway-energy/cc/async/${j}-${file}-stats"> "./data/subway-energy/cc/async/${i}-${file}"
        sleep 5
        ./cc-sync --input "${j}" --source "${randNum}" --energy true --efile "./data/subway-energy/cc/sync/${j}-${file}-readings" --estats "./data/subway-energy/cc/sync/${j}-${file}-stats"> "./data/subway-energy/cc/sync/${i}-${file}"
        sleep 5

        ./pr-async --input "${j}" --source "${randNum}" > "./data/subway/pr/async/${i}-${file}"
        sleep 5
        ./pr-sync--input "${j}" --source "${randNum}" > "./data/subway/pr/sync/${i}-${file}"
        sleep 5
        ./pr-async --input "${j}" --source "${randNum}" --energy true --efile "./data/subway-energy/pr/async/${j}-${file}-readings" --estats "./data/subway-energy/pr/async/${j}-${file}-stats"> "./data/subway-energy/pr/async/${i}-${file}"
        sleep 5
        ./pr-sync --input "${j}" --source "${randNum}" --energy true --efile "./data/subway-energy/pr/sync/${j}-${file}-readings" --estats "./data/subway-energy/pr/sync/${j}-${file}-stats"> "./data/subway-energy/pr/sync/${i}-${file}"
        sleep 5

        ./sssp-async --input "${j}" --source "${randNum}" > "./data/subway/sssp/async/${i}-${file}"
        sleep 5
        ./sssp-sync--input "${j}" --source "${randNum}" > "./data/subway/sssp/sync/${i}-${file}"
        sleep 5
        ./sssp-async --input "${j}" --source "${randNum}" --energy true --efile "./data/subway-energy/sssp/async/${j}-${file}-readings" --estats "./data/subway-energy/sssp/async/${j}-${file}-stats"> "./data/subway-energy/sssp/async/${i}-${file}"
        sleep 5
        ./sssp-sync --input "${j}" --source "${randNum}" --energy true --efile "./data/subway-energy/sssp/sync/${j}-${file}-readings" --estats "./data/subway-energy/sssp/sync/${j}-${file}-stats"> "./data/subway-energy/sssp/sync/${i}-${file}"
        sleep 5

        ./sswp-async --input "${j}" --source "${randNum}" > "./data/subway/sswp/async/${i}-${file}"
        sleep 5
        ./sswp-sync--input "${j}" --source "${randNum}" > "./data/subway/sswp/sync/${i}-${file}"
        sleep 5
        ./sswp-async --input "${j}" --source "${randNum}" --energy true --efile "./data/subway-energy/sswp/async/${j}-${file}-readings" --estats "./data/subway-energy/sswp/async/${j}-${file}-stats"> "./data/subway-energy/sswp/async/${i}-${file}"
        sleep 5
        ./sswp-sync --input "${j}" --source "${randNum}" --energy true --efile "./data/subway-energy/sswp/sync/${j}-${file}-readings" --estats "./data/subway-energy/sswp/sync/${j}-${file}-stats"> "./data/subway-energy/sswp/sync/${i}-${file}"
        sleep 5

           
        count=$((count+1))
    done
done
