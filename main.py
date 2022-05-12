import tweepy
import os
from dotenv import load_dotenv
from textblob import TextBlob
import pandas as pd

load_dotenv("./.env")
consumer_key = os.environ.get("twitter_api_key")
consumer_secret = os.environ.get("twitter_api_secret")

access_key = os.environ.get("access_key")
access_secret = os.environ.get("access_secret")

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_key, access_secret)

api = tweepy.API(auth)

public_tweets = api.search_tweets("python")

text_list = []

for tweet in public_tweets:
    text_list.append(tweet.text)

df = pd.DataFrame(text_list)

df.to_csv("./learning dataset/python_tweets.csv")
