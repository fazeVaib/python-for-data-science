import tweepy
from textblob import TextBlob

cus_key = 'MGURGewHn0r6Pch0Ta2zRiytV'
cus_secret = 'AMkWMSTdCR5TTzXUZdDS8RThi0eRGEFiO1dXzQpLZYYcHDOoHY'
access_token = '806000841798713344-nzuo0ydRCZNes0dTXkb6gZJY67m9jkR'
access_token_secret = '8B6dFXjkYe6GSEFO7WgsNxTO4ECxP2a49bEq01DPrlAeq'

auth = tweepy.OAuthHandler(cus_key, cus_secret)
auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth)

public_tweets = api.search('happy')

for tweets in public_tweets:
    print(tweets.text)
    analysis = TextBlob(tweets.text)
    print(analysis.sentiment)