import sys
from tweepy import API
from tweepy import OAuthHandler
from tweepy import Cursor
import jsonpickle

# Maximum number of tweets we want to collect
maxTweets = 60

# The twitter Search API allows up to 100 tweets per query
tweetsPerQry = 30


def get_twitter_auth():
    try:
        consumer_key = 'zJ4BXEKFomcGZqYpWi4gHbsdg'
        consumer_secret = 'hpxeVh77Cq2nM4i0BNm2FqEEwkXhlfNbZlTqNJRhhBcluCuS1g'
        access_token = '2350418059-lMgZIe14OJchaCjWVetLePnuZFahIan9mtgLoF8'
        access_secret = 'UQf1jy2FiWfJU7P0FJ4iAQrejLMtE9sIIjd2n7U93iFL3'
    except KeyError:
        sys.stderr.write('key not set')
        sys.exit(1)

    auth = OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_secret)
    return auth


def get_twitter_client():
    auth = get_twitter_auth()
    client = API(auth)
    return client


def twitter_query(location, query_string):
    # Getting Geo ID for USA
    places = api.geo_search(query=location, granularity="country")
    place_id = places[0].id

    # print('GEO id is: ', place_id)
    print(api.rate_limit_status()['resources']['search'])

    searchQuery = 'place:' + place_id + ' ' + query_string
    # searchQuery = query_string

    tweetCount = 0
    with open('../output/crawl_data.json', 'w') as f:

        # Tell the Cursor method that we want to use the Search API (api.search)
        # Also tell Cursor our query, and the maximum number of tweets to return
        for tweet in Cursor(api.search, q=searchQuery).items(maxTweets):

            # Verify the tweet has place info before writing (It should, if it got past our place filter)
            if tweet.place is not None:
                # Write the JSON format to the text file, and add one to the number of tweets we've collected
                f.write(jsonpickle.encode(tweet._json, unpicklable=False) + '\n')
                tweetCount += 1
            print tweet.created_at, tweet.text
        # Display how many tweets we have collected
        print("Downloaded {0} tweets".format(tweetCount))


if __name__ == '__main__':
    api = get_twitter_client()
    twitter_query('USA', 'python')
