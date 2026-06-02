#!/bin/bash
# ===== SCRIPT OVERVIEW =====
# Task:   Extract tweet-level fields from gzipped Brandwatch JSON exports to per-file CSVs using jq.
# Input:  *.json.gz
# Output: /*.csv.gz is one file per JSON input with columns: id, conversation_id, text, lang, created_at, author_id, retweet_count, reply_count, quote_count, like_count, impression_count, urls, expanded_urls, hashtags, mentions, tweet_type, retweeted_id, quoted_id, in_reply_to_user_id
# Usage:  bash 1-extract_tweet_data.sh

input_dir="/misinfo-panel-austria/data/JSONfiles"
output_dir="/misinfo-panel-austria/data/CSVtweets"

for input_file in "$input_dir"/*.json.gz; do

    base_name=$(basename "$input_file" .json.gz)
    output_file="$output_dir/${base_name}.csv.gz"
        
    (
        echo "id,conversation_id,text,lang,created_at,author_id,retweet_count,reply_count,quote_count,like_count,impression_count,urls,expanded_urls,hashtags,mentions,tweet_type,retweeted_id,quoted_id,in_reply_to_user_id";

        zcat "$input_file" | jq -r '
            .data[] | [
                .id,
                .conversation_id,
                (.text | gsub("\n"; " ") | gsub("\r"; " ")),
                .lang,
                .created_at,
                .author_id,
                .public_metrics.retweet_count,
                .public_metrics.reply_count,
                .public_metrics.quote_count,
                .public_metrics.like_count,
                .public_metrics.impression_count,
                ([.entities.urls?[]?.url] | join(" ")) // "",
                ([.entities.urls?[]?.expanded_url] | join(" ")) // "",
                ([.entities.hashtags?[]?.tag] | join(" ")) // "",
                ([.entities.mentions?[]?.username] | join(" ")) // "",
                ([.referenced_tweets?[]?.type] | join(" ")) // "",
                ([.referenced_tweets?[]? | select(.type=="retweeted").id] | join(" ")) // "",
                ([.referenced_tweets?[]? | select(.type=="quoted").id] | join(" ")) // "",
                (.in_reply_to_user_id) // ""
            ] | @csv'
        ) | gzip > "$output_file"

    echo "Processed $base_name"
done
echo "All files processed"