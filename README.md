# COVID_Twitter
 HKUST COMP4641 Course Project
 
 # To-do list:
 ## 1. Interaction Network (Retweet & Commnet)
 ## 2. Tweet Similarity Network
 ## 3. Stack network together to see clustering
 ## 4. GNN
 
 Contributors: 
 Haoran Liang
 Yu Man Poon
 Chi Ho Wong

## Script Descriptions:
### file_adjustments.py
Script use to fix the bad lines in the tweet csv.
### retweet_net.py
Script use to fix the bad lines in the retweet csv data.
### keywords.txt
txt file use to store keywords. Feel free to play with it by adding new keywords.
### LangDetect.py & LanguageDetector.py
Two versions of script to filter out English data in tweet csv.
### KeyDetect.py & DetectAll.py
Script use to filter out keywords from the English tweet csv. Don't use KeyDetect.py, some bugs are unresolved.
### simple_retweet_network.ipynb
Some statistics with the dataset and some bugs. Don't use that.
### retweet_network_with_weight.ipynb
Fixed bugs from previous retweet network ipynb. Add weights to edges.
### Co_hashtag_network.ipynb
Generate co-retweeted network by number of common hashtags.
### co_mention_network.ipynb
Generate co-mentioned network by number of common mentions.
