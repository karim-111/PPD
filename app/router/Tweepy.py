from fastapi import Depends, APIRouter
from pydantic import BaseSettings
from functools import lru_cache
from . import config
import tweepy as tweepy
import pandas as pd

router = APIRouter()

# get the env API Keys


@lru_cache()
def get_settings():
    return config.Settings()


async def search(word, number, api):
    location_data = []

    if word != "":
        search_words = word
        new_search = search_words + "-filter:retweets"
        date_since = "2020-03-15"

        tweets = tweepy.Cursor(api.search_tweets,
                               q=new_search,
                               lang="en",
                               since=date_since).items(number)

        # Collect a list of tweets
        users_locs = [[tweet.created_at,
                       tweet.text,
                       tweet.user.screen_name,
                       " ".join([hashtag['text']
                                for hashtag in tweet.entities['hashtags']]),
                       tweet.user.location,
                       tweet.source,
                       tweet.retweet_count]
                      for
                      tweet in tweets]
        df = pd.DataFrame(data=users_locs,
                          columns=['Created_at', 'Text', 'user', 'hashtags', 'location', 'source', 'retweet_count'])

        return df


async def search_by_name(name, api):
    posts = api.user_timeline(
        screen_name=name, count=10, lang="en", tweet_mode="extended")
    users_locs = [[
        tweet.created_at,
        tweet.full_text,
        tweet.user.screen_name,
        " ".join([hashtag['text'] for hashtag in tweet.entities['hashtags']]),
        tweet.user.location,
        tweet.source,
        tweet.retweet_count
    ] for tweet in posts]

    df = pd.DataFrame(data=users_locs,
                      columns=['Created_at', 'Text', 'user', 'hashtags', 'location', 'source', 'retweet_count'])
    return df


@router.get("/tweets")
async def info(settings: config.Settings = Depends(get_settings)):

    auth = tweepy.OAuthHandler(settings.CONSUMER_KEY, settings.CONSUMER_SECRET)
    auth.set_access_token(settings.ACCESS_KEY, settings.ACCESS_SECRET)
    api = tweepy.API(auth, wait_on_rate_limit=True)

    df = await search("Macron", 100, api)
    print(df)
    return {
        "Succes": "Tweets are loaded"
    }


@router.get("/tweets_name/")
async def info(settings: config.Settings = Depends(get_settings)):

    auth = tweepy.OAuthHandler(settings.CONSUMER_KEY, settings.CONSUMER_SECRET)
    auth.set_access_token(settings.ACCESS_KEY, settings.ACCESS_SECRET)
    api = tweepy.API(auth, wait_on_rate_limit=True)

    df = await search_by_name("musk", api)
    print(df)
    return {
        "Succes": "Tweets are loaded"
    }
