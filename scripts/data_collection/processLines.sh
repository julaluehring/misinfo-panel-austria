while read -r line
do
    echo $line | jq -r '.data[] | [.author_id, .conversation_id, .created_at, .id, .in_reply_to_user_id, .referenced_tweets, .lang, .public_metrics, .text] | @csv' >> Panel-AT-BWdata.csv
done < $1

echo "Processing completed."