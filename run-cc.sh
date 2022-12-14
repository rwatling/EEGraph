#!/bin/bash

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

for j in {1..5}
do

    echo $j

    for i in ./data/*.edges
    do
        file=$(basename ${i})
        file="${file%%.edges}"

        echo $file

			randNum=$((RANDOM % 64))

        ./build/cc --input "${i}" --variant async_push_td --source "${randNum}" > "./data/regular/async_push_td/${j}-${file}"
        sleep 5
        ./build/cc --input "${i}" --variant async_push_dd --source "${randNum}" > "./data/regular/async_push_dd/${j}-${file}"
        sleep 5
        ./build/cc --input "${i}"  --variant sync_push_td --source "${randNum}" > "./data/regular/sync_push_td/${j}-${file}"
        sleep 5
        ./build/cc --input "${i}"  --variant sync_push_dd --source "${randNum}" > "./data/regular/sync_push_dd/${j}-${file}"
        sleep 5

        ./build/cc --input "${i}" --variant async_push_td --um true --source "${randNum}"  > "./data/um/async_push_td/${j}-${file}"
        sleep 5
        ./build/cc --input "${i}" --variant async_push_dd --um true --source "${randNum}" > "./data/um/async_push_dd/${j}-${file}"
        sleep 5
        ./build/cc --input "${i}" --variant sync_push_td --um true --source "${randNum}" > "./data/um/sync_push_td/${j}-${file}"
        sleep 5
        ./build/cc --input "${i}" --variant sync_push_dd --um true --source "${randNum}" > "./data/um/sync_push_dd/${j}-${file}"
        sleep 5

    done
done
