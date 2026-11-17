#!/bin/bash
# Task:   Extract user metadata from the Brandwatch JSON exports with jq, one CSV per input file.
# Input:  data/JSONfiles/*.json.gz (not released)
# Output: data/UserCSVfiles/*.csv.gz
# Run:    bash 4-extract_user_data.sh

input_dir="../../data/JSONfiles"
output_dir="../../data/UserCSVfiles"

for input_file in "$input_dir"/*.json.gz; do

    base_name=$(basename "$input_file" .json.gz)
    output_file="$output_dir/${base_name}.csv.gz"
        
    (
        echo "author_id,username,name,verified,followers_count,following_count,tweet_count,created_at,description,location,expanded_urls";

        zcat "$input_file" | jq -r '
            .includes.users[] | [
                .id,
                .username,
                .name,
                .verified,
                (.public_metrics.followers_count // 0),
                (.public_metrics.following_count // 0),
                (.public_metrics.tweet_count // 0),
                .created_at,
                (.description | gsub("\n"; " ") | gsub("\r"; " ")) // "",
                (.location // ""),
                (
                  if .entities.url.urls then
                    (.entities.url.urls | map(.expanded_url) | join(" "))
                  else
                    ""
                  end
                )
            ] | @csv
        '
        ) | gzip > "$output_file"

    echo "Processed $base_name"

done
echo "All files processed"