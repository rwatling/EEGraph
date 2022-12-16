#!/bin/bash

mkdir -p ./data/

mkdir -p ./data/regular
mkdir -p ./data/um

mkdir -p ./data/regular/async_push_td
mkdir -p ./data/regular/async_push_dd
mkdir -p ./data/regular/sync_push_td
mkdir -p ./data/regular/sync_push_dd

mkdir -p ./data/um/async_push_td
mkdir -p ./data/um/async_push_dd
mkdir -p ./data/um/sync_push_td
mkdir -p ./data/um/sync_push_dd

inputs=( "./datasets/Google/web-Google.txt" 
        "./datasets/Higgs/higgs-social_network.edgelist" 
        "./datasets/LiveJournal/soc-LiveJournal1.txt" 
        "./datasets/Pokec/soc-pokec-relationships.txt"
        "./datasets/Road/roadNet-CA.txt"
        "./datasets/Skitter/as-skitter.txt"
        "./datasets/Youtube/com-youtube.ungraph.txt")

names=("google" "higgs" "livejournal" "pokec" "road" "skitter" "youtube")

for j in {1..5}
do

    echo $j
    count=0

    for i in "${inputs[@]}"
    do
        file="${names[count]}"
        echo $file
		randNum=$((RANDOM % 1024))

        $1 --input "${i}" --variant async_push_td --source "${randNum}" > "./data/regular/async_push_td/${j}-${file}"
        sleep 5
        $1 --input "${i}" --variant async_push_dd --source "${randNum}" > "./data/regular/async_push_dd/${j}-${file}"
        sleep 5
        $1 --input "${i}"  --variant sync_push_td --source "${randNum}" > "./data/regular/sync_push_td/${j}-${file}"
        sleep 5
        $1 --input "${i}"  --variant sync_push_dd --source "${randNum}" > "./data/regular/sync_push_dd/${j}-${file}"
        sleep 5

        $1 --input "${i}" --variant async_push_td --um true --source "${randNum}"  > "./data/um/async_push_td/${j}-${file}"
        sleep 5
        $1 --input "${i}" --variant async_push_dd --um true --source "${randNum}" > "./data/um/async_push_dd/${j}-${file}"
        sleep 5
        $1 --input "${i}" --variant sync_push_td --um true --source "${randNum}" > "./data/um/sync_push_td/${j}-${file}"
        sleep 5
        $1 --input "${i}" --variant sync_push_dd --um true --source "${randNum}" > "./data/um/sync_push_dd/${j}-${file}"
        sleep 5

        count=$((count+1))

    done
done
