from flask import Flask,render_template,request,flash
from sklearn.externals import joblib
import sklearn
import pandas as pd
from tweepy import API
from tweepy import Cursor
from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
from tweepy import Stream
import twitter_credentials

app =Flask(__name__)

@app.route("/")
def third():
  return render_template("index.html")
@app.route("/predict",methods=["GET","POST"])
def first():
	if request.method == "POST":

		class TwitterClient():

			def __init__(self, twitter_user=None):
				self.auth = TwitterAuthenticator().authenticate_twitter_app()
				self.twitter_client = API(self.auth)

				self.twitter_user = twitter_user

			def get_twitter_client_api(self):
				return self.twitter_client

			def get_user_timeline_tweets(self, num_tweets):
				tweets = []
				for tweet in Cursor(self.twitter_client.user_timeline, id=self.twitter_user).items(num_tweets):
					tweets.append(tweet)
				return tweets

			def get_friend_list(self, num_friends):
				friend_list = []
				for friend in Cursor(self.twitter_client.friends, id=self.twitter_user).items(num_friends):
					friend_list.append(friend)
				return friend_list

		# # # # TWITTER AUTHENTICATER # # # #
		class TwitterAuthenticator():

			def authenticate_twitter_app(self):
				auth = OAuthHandler(twitter_credentials.CONSUMER_KEY, twitter_credentials.CONSUMER_SECRET)
				auth.set_access_token(twitter_credentials.ACCESS_TOKEN, twitter_credentials.ACCESS_TOKEN_SECRET)
				return auth

		# # # # TWITTER STREAMER # # # #
		class TwitterStreamer():
			"""
			Class for streaming and processing live tweets.
			"""

			def __init__(self):
				self.twitter_autenticator = TwitterAuthenticator()

			def stream_tweets(self, fetched_tweets_filename, hash_tag_list):
				# This handles Twitter authetification and the connection to Twitter Streaming API
				listener = TwitterListener(fetched_tweets_filename)
				auth = self.twitter_autenticator.authenticate_twitter_app()
				stream = Stream(auth, listener)

				# This line filter Twitter Streams to capture data by the keywords:
				stream.filter(track=hash_tag_list)

		# # # # TWITTER STREAM LISTENER # # # #
		class TwitterListener(StreamListener):
			"""
			This is a basic listener that just prints received tweets to stdout.
			"""

			def __init__(self, fetched_tweets_filename):
				self.fetched_tweets_filename = fetched_tweets_filename

			def on_data(self, data):
				try:
					print(data)
					with open(self.fetched_tweets_filename, 'a') as tf:
						tf.write(data)
					return True
				except BaseException as e:
					print("Error on_data %s" % str(e))
				return True

			def on_error(self, status):
				if status == 420:
					# Returning False on_data method in case rate limit occurs.
					return False
				print(status)

		if __name__ == '__main__':
			twitter_client = TwitterClient()

			api = twitter_client.get_twitter_client_api()
			mydict = {}

			username = request.form['tweet']
			tweets = api.get_user(screen_name=username)
			# mention=api.mentions_timeline(screen_name="pranjal_003")
			# print(mention)
			user = api.user_timeline(screen_name=username)
			#mc = tweets.id
			from datetime import datetime
			import datetime

			dateTimeObj = tweets.created_at
			timestampStr = dateTimeObj.strftime("%d-%b-%Y (%H:%M:%S.%f)")
			# print('Current Timestamp : ', timestampStr)
			date = timestampStr.split(" ")
			dd = (date[0].split("-"))
			day = int(dd[0])
			year = int(dd[2])

			if (dd[1] == "Jan"):
				month = 1
			elif (dd[1] == "Feb"):
				month = 2
			elif (dd[1] == "Mar"):
				month = 3
			elif (dd[1] == "Apr"):
				month = 4
			elif (dd[1] == "May"):
				month = 5
			elif (dd[1] == "Jun"):
				month = 6
			elif (dd[1] == "Jul"):
				month = 7
			elif (dd[1] == "Aug"):
				month = 8
			elif (dd[1] == "Sep"):
				month = 9
			elif (dd[1] == "Oct"):
				month = 10
			elif (dd[1] == "Nov"):
				month = 11
			elif (dd[1] == "Dec"):
				month = 12
			# print(day, year,month)
			x = datetime.datetime(year, month, day)
			# print(x)
			xx = datetime.datetime.now()

			def numOfDays(date1, date2):
				return (date2 - date1).days

			mydict["Longevity"] = (numOfDays(x, xx))

			mydict["Length of screen name"] = len(tweets.screen_name)

			if tweets.description:
				mydict["Does the profile have description"] = 1
				mydict["Length of description"] = len(tweets.description)
			else:
				mydict["Does the profile have description"] = 0
				mydict["Length of description"] = len(tweets.description)
			if tweets.url:
				mydict["Does the profile have a url"] = 1
			else:
				mydict["Does the profile have a url"] = 0
			mydict["Followee count of the user"] = tweets.friends_count
			mydict["Followers count of the user"] = tweets.followers_count
			mydict["Followee-by-follower ratio"] = tweets.friends_count / tweets.followers_count
			mydict["Total number of tweets"] = tweets.statuses_count
			# mydict["Number of re-tweets per tweet"]=
			# mydict["Number of direct mentions per tweet"] = len(tweets.entities[].user_mentions)

			# mydict[" statuses likelihood per day for seven days"]= (tweets.statuses_count/(numOfDays(x, xx)))*7
			# mydict["retweets"]=tweets.retweet_count
			# # # # TWITTER CLIENT # # # #
		#query = pd.get_dummies(pd.DataFrame(mydict))
		#query = query.reindex(columns=model_columns, fill_value=0)
		#print(query)
		import json
		#y = json.dumps(mydict)
		# Python3 program to convert
		# list into a list of lists

		# def extractDigits(lst):
		# 	return list(map(lambda el: [el], lst))
		#
		# # Driver code
		y = list(mydict.values())
		# a=extractDigits(y)
		# print(a)
		z=json.dumps(mydict)
		print(z)

		a=[]
		a.append(y)
		print(a)

		prediction = lr.predict(a)
		print(prediction)
		if prediction == [1]:
			ans="Customer"
		else:
			ans="Normal user"
		return render_template("index.html", name=ans)

		# print(dir(tweets[0]))
		# print(tweets[0].retweet_count)
@app.route("/about")
def second():
	return render_template("about.html")
@app.route("/index.html")
def fourth():
	return render_template('index.html')


if __name__ == '__main__':
	lr = joblib.load("modnew.pkl")  # Load "model.pkl"
	print('Model loaded')
	model_columns = joblib.load("model_new.pkl")  # Load "model_columns.pkl"
	print('Model columns loaded')
	app.run(host='127.0.0.1', port=3130,debug=True)