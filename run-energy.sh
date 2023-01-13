#!/bin/bash

mkdir -p ./data/

mkdir -p ./data/regular-energy
mkdir -p ./data/um-energy

mkdir -p ./data/regular-energy/async_push_td
mkdir -p ./data/regular-energy/async_push_dd
mkdir -p ./data/regular-energy/sync_push_td
mkdir -p ./data/regular-energy/sync_push_dd

mkdir -p ./data/um-energy/async_push_td
mkdir -p ./data/um-energy/async_push_dd
mkdir -p ./data/um-energy/sync_push_td
mkdir -p ./data/um-energy/sync_push_dd

inputs=( "./datasets/Google/web-Google.txt" 
        "./datasets/Higgs/higgs-social_network.edgelist" 
        "./datasets/LiveJournal/soc-LiveJournal1.txt" 
        "./datasets/Pokec/soc-pokec-relationships.txt"
        "./datasets/Road/roadNet-CA.txt"
        "./datasets/Skitter/as-skitter.txt")

names=("google" "higgs" "livejournal" "pokec" "road" "skitter")

for j in {1..3}
do

    echo $j
    count=0

    for i in "${inputs[@]}"
    do
        file="${names[count]}"
        echo $file
		randNum=$((RANDOM % 32))

        $1 --input "${i}" --variant async_push_td --source "${randNum}" --energy true --efile "./data/regular-energy/async_push_td/${j}-${file}-readings" --estats "./data/regular-energy/async_push_td/${j}-${file}-stats" > "./data/regular-energy/async_push_td/${j}-${file}"
        sleep 5
        $1 --input "${i}" --variant async_push_dd --source "${randNum}" --energy true --efile "./data/regular-energy/async_push_dd/${j}-${file}-readings" --estats "./data/regular-energy/async_push_dd/${j}-${file}-stats"> "./data/regular-energy/async_push_dd/${j}-${file}"
        sleep 5
        $1 --input "${i}" --variant sync_push_td --source "${randNum}" --energy true --efile "./data/regular-energy/sync_push_td/${j}-${file}-readings" --estats "./data/regular-energy/sync_push_td/${j}-${file}-stats"> "./data/regular-energy/sync_push_td/${j}-${file}"
        sleep 5
        $1 --input "${i}" --variant sync_push_dd --source "${randNum}" --energy true --efile "./data/regular-energy/sync_push_dd/${j}-${file}-readings" --estats "./data/regular-energy/sync_push_dd/${j}-${file}-stats"> "./data/regular-energy/sync_push_dd/${j}-${file}"
        sleep 5

        $1 --input "${i}" --variant async_push_td --um true --source "${randNum}" --energy true --efile "./data/um-energy/async_push_td/${j}-${file}-readings" --estats "./data/um-energy/async_push_td/${j}-${file}-stats"> "./data/um-energy/async_push_td/${j}-${file}"
        sleep 5
        $1 --input "${i}" --variant async_push_dd --um true --source "${randNum}" --energy true --efile "./data/um-energy/async_push_dd/${j}-${file}-readings" --estats "./data/um-energy/async_push_dd/${j}-${file}-stats"> "./data/um-energy/async_push_dd/${j}-${file}"
        sleep 5
        $1 --input "${i}" --variant sync_push_td --um true --source "${randNum}" --energy true --efile "./data/um-energy/sync_push_td/${j}-${file}-readings" --estats "./data/um-energy/sync_push_td/${j}-${file}-stats"> "./data/um-energy/sync_push_td/${j}-${file}"
        sleep 5
        $1 --input "${i}" --variant sync_push_dd --um true --source "${randNum}" --energy true --efile "./data/um-energy/sync_push_dd/${j}-${file}-readings" --estats "./data/um-energy/sync_push_dd/${j}-${file}-stats"> "./data/um-energy/sync_push_dd/${j}-${file}"
        sleep 5

        count=$((count+1))
    done
done
