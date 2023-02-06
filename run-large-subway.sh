#!/bin/bash

mkdir -p ./data/

mkdir -p ./data/subway-large
mkdir -p ./data/um-large

mkdir -p ./data/subway-large/async
mkdir -p ./data/subway-large/sync

mkdir -p ./data/um-large/async_push_td
mkdir -p ./data/um-large/async_push_dd
mkdir -p ./data/um-large/sync_push_td
mkdir -p ./data/um-large/sync_push_dd

inputs=( "/home/share/graph_data/raw/twitter_mpi/twitter.el" 
        "/home/share/graph_data/raw/friendster_snap/fs.el")

names=("tw" "fs")

for j in {1..1}
do

    echo $j
    count=0

    for i in "${inputs[@]}"
    do
        file="${names[count]}"
        echo $file
		#randNum=$((RANDOM % 32))
        randNum=0

        $1 --input "${i}" --variant async_push_td --um true --source "${randNum}"  > "./data/um-large/async_push_td/${j}-${file}"
        sleep 5
        $1 --input "${i}" --variant async_push_dd --um true --source "${randNum}" > "./data/um-large/async_push_dd/${j}-${file}"
        sleep 5
        $1 --input "${i}" --variant sync_push_td --um true --source "${randNum}" > "./data/um-large/sync_push_td/${j}-${file}"
        sleep 5
        $1 --input "${i}" --variant sync_push_dd --um true --source "${randNum}" > "./data/um-large/sync_push_dd/${j}-${file}"
        sleep 5

        $2 --input "${i}" --source "${randNum}" > "./data/subway-large/async/${j}-${file}"
        sleep 5
        $3 --input "${i}" --source "${randNum}" > "./data/subway-large/sync/${j}-${file}"
        sleep 5

        count=$((count+1))

    done
done
