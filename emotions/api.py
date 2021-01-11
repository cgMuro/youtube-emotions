import os
from tweepy import OAuthHandler, API, TweepError
from credentials import API_KEY, API_SECRET, ACCESS_TOKEN, ACCESS_TOKEN_SECRET
import pandas as pd


# ACCESS TO API
auth = OAuthHandler(API_KEY, API_SECRET)
auth.set_access_token(ACCESS_TOKEN, ACCESS_TOKEN_SECRET)
api = API(auth)
print('Successfully connected to the Twitter API.')


# SEARCH TWEETS
emotions = ['#happy', '#disgust', '#angry', '#joy', '#advice', '#love']

max_requests = 180
res = {}

for emotion in emotions:
    q = emotion + ' -filter:retweets'
    searched_tweets = []
    last_id = -1
    request_count = 0

    while request_count < max_requests:
        try:
            new_tweets = api.search(
                q=q,
                lang='en',
                count=100,
                max_id=str(last_id - 1),
                tweet_mode='extended'
            )

            if not new_tweets:
                break
            searched_tweets.extend(new_tweets)
            last_id = new_tweets[-1].id
            request_count += 1
        except TweepError as e:
            print(e)
            break
    res[emotion] = searched_tweets
    print(len(searched_tweets), ' ', emotion, ' tweets')


# FORMAT AND SAVE TO .CSV
data = []

if len(list(res.values())[0]) > 0:
    for emotion, values in res.items():
        for tweet in values:
            data.append([tweet.id, tweet.created_at, tweet.user.screen_name, tweet.full_text, emotion[1:]])

    df = pd.DataFrame(data=data, columns=['id', 'date', 'user', 'text', 'emotion'])

    df.to_csv('./data/raw/emotions_2.csv')
