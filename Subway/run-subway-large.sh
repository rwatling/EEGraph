#!/bin/bash

mkdir -p ./data/

mkdir -p ./data/subway-large

mkdir -p ./data/subway-large/bfs
mkdir -p ./data/subway-large/bfs/async
mkdir -p ./data/subway-large/bfs/sync

mkdir -p ./data/subway-large/cc
mkdir -p ./data/subway-large/cc/async
mkdir -p ./data/subway-large/cc/sync

mkdir -p ./data/subway-large/pr
mkdir -p ./data/subway-large/pr/async
mkdir -p ./data/subway-large/pr/sync

mkdir -p ./data/subway-large/sssp
mkdir -p ./data/subway-large/sssp/async
mkdir -p ./data/subway-large/sssp/sync

mkdir -p ./data/subway-large/sswp
mkdir -p ./data/subway-large/sswp/async
mkdir -p ./data/subway-large/sswp/sync

mkdir -p ./data/subway-large-energy

mkdir -p ./data/subway-large-energy/bfs
mkdir -p ./data/subway-large-energy/bfs/async
mkdir -p ./data/subway-large-energy/bfs/sync

mkdir -p ./data/subway-large-energy/cc
mkdir -p ./data/subway-large-energy/cc/async
mkdir -p ./data/subway-large-energy/cc/sync

mkdir -p ./data/subway-large-energy/pr
mkdir -p ./data/subway-large-energy/pr/async
mkdir -p ./data/subway-large-energy/pr/sync

mkdir -p ./data/subway-large-energy/sssp
mkdir -p ./data/subway-large-energy/sssp/async
mkdir -p ./data/subway-large-energy/sssp/sync

mkdir -p ./data/subway-large-energy/sswp
mkdir -p ./data/subway-large-energy/sswp/async
mkdir -p ./data/subway-large-energy/sswp/sync

inputs=( "/home/share/graph_data/raw/twitter_mpi/twitter.el" 
        "/home/share/graph_data/raw/friendster_snap/fs.el")

names=("tw" "fs")

for i in {1..3}
do

    echo $i
    count=0

    for j in "${inputs[@]}"
    do
        file="${names[count]}"
        echo $file

        ./bfs-async --input "${j}" > "./data/subway-large/bfs/async/${i}-${file}"
        ./bfs-sync --input "${j}" > "./data/subway-large/bfs/sync/${i}-${file}"
        ./bfs-async --input "${j}" --energy true --efile "./data/subway-large-energy/bfs/async/${j}-${file}-readings" --estats "./data/subway-large-energy/bfs/async/${j}-${file}-stats" > "./data/subway-large-energy/bfs/async/${i}-${file}"
        ./bfs-sync --input "${j}" --energy true --efile "./data/subway-large-energy/bfs/sync/${j}-${file}-readings" --estats "./data/subway-large-energy/bfs/sync/${j}-${file}-stats" > "./data/subway-large-energy/bfs/sync/${i}-${file}"
        

        ./cc-async --input "${j}" > "./data/subway-large/cc/async/${i}-${file}"
        ./cc-sync --input "${j}" > "./data/subway-large/cc/sync/${i}-${file}"
        ./cc-async --input "${j}" --energy true --efile "./data/subway-large-energy/cc/async/${j}-${file}-readings" --estats "./data/subway-large-energy/cc/async/${j}-${file}-stats" > "./data/subway-large-energy/cc/async/${i}-${file}"
        ./cc-sync --input "${j}" --energy true --efile "./data/subway-large-energy/cc/sync/${j}-${file}-readings" --estats "./data/subway-large-energy/cc/sync/${j}-${file}-stats" > "./data/subway-large-energy/cc/sync/${i}-${file}"
        

        ./pr-async --input "${j}" > "./data/subway-large/pr/async/${i}-${file}"
        ./pr-sync --input "${j}" > "./data/subway-large/pr/sync/${i}-${file}"
        ./pr-async --input "${j}" --energy true --efile "./data/subway-large-energy/pr/async/${j}-${file}-readings" --estats "./data/subway-large-energy/pr/async/${j}-${file}-stats" > "./data/subway-large-energy/pr/async/${i}-${file}"
        ./pr-sync --input "${j}" --energy true --efile "./data/subway-large-energy/pr/sync/${j}-${file}-readings" --estats "./data/subway-large-energy/pr/sync/${j}-${file}-stats" > "./data/subway-large-energy/pr/sync/${i}-${file}"
        

        ./sssp-async --input "${j}" > "./data/subway-large/sssp/async/${i}-${file}"
        ./sssp-sync --input "${j}" > "./data/subway-large/sssp/sync/${i}-${file}"
        ./sssp-async --input "${j}" --energy true --efile "./data/subway-large-energy/sssp/async/${j}-${file}-readings" --estats "./data/subway-large-energy/sssp/async/${j}-${file}-stats" > "./data/subway-large-energy/sssp/async/${i}-${file}"
        ./sssp-sync --input "${j}" --energy true --efile "./data/subway-large-energy/sssp/sync/${j}-${file}-readings" --estats "./data/subway-large-energy/sssp/sync/${j}-${file}-stats" > "./data/subway-large-energy/sssp/sync/${i}-${file}"
        

        ./sswp-async --input "${j}" > "./data/subway-large/sswp/async/${i}-${file}"
        ./sswp-sync --input "${j}" > "./data/subway-large/sswp/sync/${i}-${file}"
        #./sswp-async --input "${j}" --energy true --efile "./data/subway-large-energy/sswp/async/${j}-${file}-readings" --estats "./data/subway-large-energy/sswp/async/${j}-${file}-stats" > "./data/subway-large-energy/sswp/async/${i}-${file}"
        ./sswp-sync --input "${j}" --energy true --efile "./data/subway-large-energy/sswp/sync/${j}-${file}-readings" --estats "./data/subway-large-energy/sswp/sync/${j}-${file}-stats" > "./data/subway-large-energy/sswp/sync/${i}-${file}"
        

        count=$((count+1))
    done
done
