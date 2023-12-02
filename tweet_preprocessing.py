#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 15:32:43 2023

@author: zozochunyu
"""

# Libraries
#pip install pyspellchecker
#pip install scattertext
#pip install nltk
#pip install -U kaleido

# Import Data Preprocessing and Wrangling libraries
import re
from tqdm.notebook import tqdm
import pandas as pd 
import numpy as np
from datetime import datetime
import dateutil.parser

# Import NLP Libraries
import nltk
from spellchecker import SpellChecker
from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA


# Downloading periphrals
nltk.download('vader_lexicon')
nltk.download('stopwords')

import warnings
warnings.filterwarnings('ignore')

import os
directory = '/Users/zozochunyu/Biostats_Fall23/QMSS5067NLP/Final'
csv_files = [file for file in os.listdir(directory) if file.endswith('.csv')]
df_list = []
for csv in csv_files:
    df = pd.read_csv(os.path.join(directory, csv))
    df_list.append(df)

# For sentiment analysis 
sia = SIA() 

# To identify misspelled words
spell = SpellChecker() 


# Storing csv dataset into a datframe
df = pd.concat(df_list)

# data preprocessing
data = df.copy()
data['original_tweet'] = df['tweet']
data['datetime'] = data['date']
data['datetime'] = data.datetime.apply(lambda x: dateutil.parser.parse(x))
rt_mask = data.tweet.apply(lambda x: "RT @" in x)

# standard tweet preprocessing 
data.tweet = data.tweet.str.lower()
#Remove twitter handlers
data.tweet = data.tweet.apply(lambda x:re.sub('@[^\s]+','',x))
#remove hashtags
data.tweet = data.tweet.apply(lambda x:re.sub(r'\B#\S+','',x))
# Remove URLS
data.tweet = data.tweet.apply(lambda x:re.sub(r"http\S+", "", x))
# Remove all the special characters
data.tweet = data.tweet.apply(lambda x:' '.join(re.findall(r'\w+', x)))
#remove all single characters
data.tweet = data.tweet.apply(lambda x:re.sub(r'\s+[a-zA-Z]\s+', '', x))
# Substituting multiple spaces with single space
data.tweet = data.tweet.apply(lambda x:re.sub(r'\s+', ' ', x, flags=re.I))


tidy_data = data[["date", "tweet"]]
tidy_data = tidy_data.dropna(axis=0,how='any')


filtered_data = tidy_data[(tidy_data['date'] >= '2021-02-01') & (tidy_data['date'] <= '2022-01-31')]

filtered_data  = filtered_data .sort_values(by=['date'])


filtered_data = filtered_data .sort_values(by=['date']).reset_index()
del filtered_data['index']


## Data Prep for VADER -------------------------------------------------------
grouped_data = filtered_data.groupby([filtered_data['date']]).agg(lambda column: "".join(column))
grouped_data = grouped_data.dropna(axis=0,how='any')
# grouped_data.to_csv('/Users/zozochunyu/Biostats_Fall23/QMSS5067NLP/Final/output/grouped_data.csv') 
# grouped_data = pd.read_csv('/Users/zozochunyu/Biostats_Fall23/QMSS5067NLP/Final/output/grouped_data.csv',index_col=0)


grouped_data['tweet'] = grouped_data['tweet'].astype(str)
grouped_data['scores'] = grouped_data['tweet'].apply(lambda tweet: sia.polarity_scores(tweet))
grouped_data.scores
grouped_data['compound'] = grouped_data['scores'].apply(lambda s: s.get('compound'))
grouped_data['positive'] = grouped_data['scores'].apply(lambda s: s.get('pos'))
grouped_data['negative'] = grouped_data['scores'].apply(lambda s: s.get('neg'))
grouped_data['neutral'] = grouped_data['scores'].apply(lambda s: s.get('neu'))
sentiment_scores_comb = grouped_data[['compound', 'positive', 'negative','neutral']]
sentiment_scores_comb  = sentiment_scores_comb.reset_index()

# sentiment_scores_comb.to_csv('/Users/zozochunyu/Biostats_Fall23/QMSS5067NLP/Final/output/sentiment_scores_comb.csv')
# sentiment_scores_comb = pd.read_csv('/Users/zozochunyu/Biostats_Fall23/QMSS5067NLP/Final/output/sentiment_scores_comb.csv', index_col=0)

