# Task:   Extract the tweet fields from the collected JSON with jq (one CSV line per tweet).
# Input:  <name>.json (from GetTweetsQuery.sh)
# Output: CSV on stdout (Panel-AT-BWdata.csv in the paper)
# Run:    bash processLines.sh <name>.json > Panel-AT-BWdata.csv

while read -r line
do
    echo $line | jq -r '.data[] | [.author_id, .conversation_id, .created_at, .id, .in_reply_to_user_id, .referenced_tweets, .lang, .public_metrics, .text] | @csv' >> Panel-AT-BWdata.csv
done < $1

echo "Processing completed."