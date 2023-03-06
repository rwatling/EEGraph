#!/bin/bash

mkdir -p ./build/pr-large/tw
mkdir -p ./build/pr-large/fs
mkdir -p ./build/pr-large/tw-energy
mkdir -p ./build/pr-large/fs-energy

inputs=("/home/share/graph_data/raw/twitter_mpi/twitter.el"
        "/home/share/graph_data/raw/friendster_snap/fs.el")

names=("tw" "fs")

for i in {1..1}
do

    echo $i
    count=0

    for j in "${inputs[@]}"
    do
        file="${names[count]}"
        echo $file

        ./build/pr-main --input "${j}" --um true --variant async_push_td > "./build/pr-large/${file}/async-push-td${i}"
        ./build/pr-main --input "${j}" --um true --variant async_push_dd > "./build/pr-large/${file}/async-push-dd${i}"
        ./build/pr-main --input "${j}" --um true --variant sync_push_td > "./build/pr-large/${file}/sync-push-td${i}"
        ./build/pr-main --input "${j}" --um true --variant sync_push_dd > "./build/pr-large/${file}/sync-push-dd${i}"

        ./build/pr-main --input "${j}" --um true --energy true --efile "./build/pr-large/${file}-energy/async-push-td-readings${i}" --estats "./build/pr-large/${file}-energy/async-push-td-stats${i}" --variant async_push_td > "./build/pr-large/${file}-energy/async-push-td${i}"
        ./build/pr-main --input "${j}" --um true --energy true --efile "./build/pr-large/${file}-energy/async-push-dd-readings${i}" --estats "./build/pr-large/${file}-energy/async-push-dd-stats${i}" --variant async_push_dd > "./build/pr-large/${file}-energy/async-push-dd${i}"
        ./build/pr-main --input "${j}" --um true --energy true --efile "./build/pr-large/${file}-energy/sync-push-td-readings${i}" --estats "./build/pr-large/${file}-energy/sync-push-td-stats${i}" --variant sync_push_td > "./build/pr-large/${file}-energy/sync-push-td${i}"
        ./build/pr-main --input "${j}" --um true --energy true --efile "./build/pr-large/${file}-energy/sync-push-dd-readings${i}" --estats "./build/pr-large/${file}-energy/sync-push-dd-stats${i}" --variant sync_push_dd > "./build/pr-large/${file}-energy/sync-push-dd${i}"

        count=$((count+1))
    done
done
