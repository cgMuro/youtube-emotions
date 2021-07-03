import os
import requests
import json
from env import set_env

# Set env
set_env()

API_KEY = os.getenv('API_KEY')

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
        return {
            'success': True,
            'data': res
        }
    except:
        print(res.text)
        return {
            'success': False,
            'status': res.status_code,
            'messagge': 'Something went wrong. Please try again.'
        }

# Parse data comments from Youtube API
def parse_data(data):
    comments = []
    data = data['items']

    for item in data:
        comment_info = item['snippet']['topLevelComment']['snippet']
        comment = {
            'author': comment_info['authorDisplayName'],
            'author_channel_url': comment_info['authorChannelUrl'],
            'author_profile_pic_url': comment_info['authorProfileImageUrl'],
            'comment': comment_info['textDisplay'],
        }
        comments.append(comment)

    print(comments)
    return comments
