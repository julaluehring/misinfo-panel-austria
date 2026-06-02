#curl -X POST --data-urlencode 'password=YOUR_PASSWORD' 'https://api.brandwatch.com/oauth/token?username=YOUR_EMAIL&grant_type=api-password&client_id=brandwatch-api-client'

# YOUR_QUERY_ID

d=2023-04-01
while [ "$d" != "2018-12-31" ]; do
  echo $d
  nextd=$(date -I -d "$d - 1 day")
  curl --max-time 60 --retry 12 -H "Authorization: Bearer YOUR_BEARER_TOKEN" -X GET "https://api.brandwatch.com/projects/YOUR_PROJECT_ID/data/mentions?queryId=$1&startDate=$nextd&endDate=$d&pageSize=5000&page=0" >> $2.json
  echo >> $2.json
  sleep 20
  curl --max-time 60 --retry 12 -H "Authorization: Bearer YOUR_BEARER_TOKEN" -X GET "https://api.brandwatch.com/projects/YOUR_PROJECT_ID/data/mentions?queryId=$1&startDate=$nextd&endDate=$d&pageSize=5000&page=1" >> $2.json
  echo >> $2.json
  d=$nextd
  sleep 20
done


# curl -H "Authorization: Bearer YOUR_BEARER_TOKEN" -X GET "https://api.brandwatch.com/projects/YOUR_PROJECT_ID/data/mentions?queryId=YOUR_QUERY_ID&startDate=2022-01-01&endDate=2022-01-02&pageSize=5000&page=0" >> test.json
# curl -H "Authorization: Bearer YOUR_BEARER_TOKEN" -X GET "https://api.brandwatch.com/projects/YOUR_PROJECT_ID/data/mentions?queryId=YOUR_QUERY_ID&startDate=2022-01-01&endDate=2022-01-02&pageSize=5000&page=1" >> test2.json &

 
