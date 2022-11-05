#!/bin/bash

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

for j in {1..3}
do

    echo $j

    for i in ./data/*.edges
    do
        file=$(basename ${i})
        file="${file%%.edges}"

        echo $file

		randNum=$((RANDOM % 64))

        ./build/sssp --input "${i}" --variant async_push_td --source "${randNum}" --energy true --efile "./data/regular-energy/async_push_td/${j}-${file}-readings" --estats "./data/regular-energy/async_push_td/${j}-${file}-stats" > "./data/regular-energy/async_push_td/${j}-${file}"
        sleep 5
        ./build/sssp --input "${i}" --variant async_push_dd --source "${randNum}" --energy true --efile "./data/regular-energy/async_push_dd/${j}-${file}-readings" --estats "./data/regular-energy/async_push_dd/${j}-${file}-stats"> "./data/regular-energy/async_push_dd/${j}-${file}"
        sleep 5
        ./build/sssp --input "${i}" --variant sync_push_td --source "${randNum}" --energy true --efile "./data/regular-energy/sync_push_td/${j}-${file}-readings" --estats "./data/regular-energy/sync_push_td/${j}-${file}-stats"> "./data/regular-energy/sync_push_td/${j}-${file}"
        sleep 5
        ./build/sssp --input "${i}" --variant sync_push_dd --source "${randNum}" --energy true --efile "./data/regular-energy/sync_push_dd/${j}-${file}-readings" --estats "./data/regular-energy/sync_push_dd/${j}-${file}-stats"> "./data/regular-energy/sync_push_dd/${j}-${file}"
        sleep 5

        ./build/sssp --input "${i}" --variant async_push_td --um true --source "${randNum}" --energy true --efile "./data/um-energy/async_push_td/${j}-${file}-readings" --estats "./data/um-energy/async_push_td/${j}-${file}-stats"> "./data/um-energy/async_push_td/${j}-${file}"
        sleep 5
        ./build/sssp --input "${i}" --variant async_push_dd --um true --source "${randNum}" --energy true --efile "./data/um-energy/async_push_dd/${j}-${file}-readings" --estats "./data/um-energy/async_push_dd/${j}-${file}-stats"> "./data/um-energy/async_push_dd/${j}-${file}"
        sleep 5
        ./build/sssp --input "${i}" --variant sync_push_td --um true --source "${randNum}" --energy true --efile "./data/um-energy/sync_push_td/${j}-${file}-readings" --estats "./data/um-energy/sync_push_td/${j}-${file}-stats"> "./data/um-energy/sync_push_td/${j}-${file}"
        sleep 5
        ./build/sssp --input "${i}" --variant sync_push_dd --um true --source "${randNum}" --energy true --efile "./data/um-energy/sync_push_dd/${j}-${file}-readings" --estats "./data/um-energy/sync_push_dd/${j}-${file}-stats"> "./data/um-energy/sync_push_dd/${j}-${file}"
        sleep 5

    done
done
