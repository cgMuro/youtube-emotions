# SCRIPT TO GET COMMENTS FROM THE YOUTUBE API
import os
import sys
import requests
import json
import numpy as np
import pandas as pd
from env import API_KEY


# Get comments data from Youtube API
def get_youtube_data(video_ID, max_results=20, order='relevance'):
    # Make request to API
    res = requests.get(
        f'https://youtube.googleapis.com/youtube/v3/commentThreads',
        params={
            'part': 'snippet',
            'maxResults': max_results,
            'order': order,
            'textFormat': 'plainText',
            'videoId': video_ID,
            'key': API_KEY
        }
    )

    # Check request status
    try:
        res.raise_for_status()
        res = res.json()
        return res
    except:
        print(res.text)
        return 'Error'

# Parse data comments from Youtube API
def parse_data(data):
    # Check for errors
    if data == 'Error':
        print('ERROR')
        sys.exit(1)

    comments = []
    data = data['items']

    for item in data:
        comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
        comments.append(comment)

    print(comments)
    return comments



data = get_youtube_data(video_ID='ZgEFwgSgOOg', max_results=120, order='relevance')
comments = parse_data(data)

df = pd.DataFrame(data=np.array(comments), columns=['text'])

df.to_csv('./data/youtube.csv')