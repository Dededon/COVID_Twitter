import pandas as pd
import os


def get_network():
	rts = pd.DataFrame()

	retweet_file_paths = 'data/tweets/retweets/retweets/'
	tweet_file_paths = 'data/tweets/tweets/'
 

	for file in os.listdir(retweet_file_paths):
		print (file)
		
		df = pd.read_csv(tweet_file_paths+file, error_bad_lines=False,dtype=str)
		
		rdf = pd.read_csv(retweet_file_paths+file, error_bad_lines=False,dtype=str)
		
		
		
		
		
		df = df[df['is_retweet']=='True']
		df = df[df['is_quote']=='False']
		
		print(file, 'filtered : ', len(df))
		

		df.columns = ['date', 'user', 'is_retweet', 'is_quote', 'text', 'quoted_text', 'lat',
		   'long', 'hts', 'mentions', 'tweet_id', 'likes',
		   'retweets', 'replies', 'quote_count', 'original_tweet_id',
		   'Unnamed: 16']
		
		df = df[['date', 'user','text', 'quoted_text','hts', 'mentions', 'tweet_id', 'original_tweet_id']]

		df = df[['date', 'user','text', 'quoted_text','hts', 'mentions', 'tweet_id', 'original_tweet_id']]

		df = df.drop_duplicates(keep='last')

		
		
		
		df.index = df['original_tweet_id'].tolist()
		
		print('-- \n',file, 'retweets : ', len(rdf))

		
		
		print(file, 'retweets filtered : ', len(rdf))


		rdf = rdf[['text', 'is_qoute_status', 'in_reply_to_status_id',
		   'created_at', 'time', 'id_str', 'hashtags', 'place', 'country','user_id']]

		rdf = rdf.drop_duplicates(keep='last')


		
		
		rdf.index = rdf['id_str'].tolist()


		
		rdf_ = df.join(rdf,lsuffix='_tweet_', how='inner')
		print ('all data" ', len(rdf_))
		
	#     break

	#     retweet_net = dd.from_pandas(rdf_, npartitions=8).groupby(['user','user_id']).size().compute().reset_index()
		
		rts = pd.concat([rts,rdf_],ignore_index=True)

		return rts
		
		
	#     break