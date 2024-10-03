#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pymongo
import pandas as pd
import numpy as np

from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import TrainingArguments, Trainer
import random
import re
from tqdm import tqdm
import string
from nltk.corpus import stopwords
import nltk
import torch

def preprocess_text(text):
    # Ensure the text is of string type
    text = str(text)
    
    # Lowercase all text
    text = text.lower()
    
    # Remove newlines
    text = re.sub('\n', '', text)
    
    # Remove mentions
    text = re.sub('@[A-Za-z0-9_]+', '', text)
    text = re.sub('@', '', text)
    
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'www\S+', '', text)
    
    # Remove numbers
    text = re.sub('[0-9]+', '', text)
    
    # Remove punctuation
    table = str.maketrans('', '', string.punctuation)
    text = ' '.join([word.translate(table) for word in text.split()])
    
    # Remove stopwords
    stops = set(stopwords.words('turkish'))
    text = ' '.join([word for word in text.split() if word not in stops])
    
    return text


# Connect to mongodb
mongo_client = pymongo.MongoClient("mongodb://localhost:27017/")
db = mongo_client["politus_twitter"]
# Get the collections(tables)
user_col = db['users']
tweet_col = db['tweets']



# Read user IDs


# Path where the model and tokenizer were saved
model_name = 'AlkanCan/TurkishBERTweet-Immigration-Stance'

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
if torch.cuda.is_available():
    model = model.cuda()  # Move model to default GPU
    model = torch.nn.DataParallel(model) 

# Bu fonksiyonu tweet metnini alıp prediction'ı export edecek şekilde düzenlemeniz gerekiyor. Prediction liste de olabilir.
def run_model(tweets, model, tokenizer, batch_size=1000):
    model.eval()  # Set model to evaluation mode
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    predictions = []

    # Process tweets in batches
    for i in tqdm(range(0, len(tweets), batch_size)):
        batch_tweets = tweets[i:i+batch_size]
        inputs = tokenizer(batch_tweets, padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            batch_predictions = torch.argmax(outputs.logits, dim=1)
            predictions.extend(batch_predictions.cpu().numpy())
            
    return predictions

  # Ensure all tweets are strings


# Tweet predictionlarını database'e geçirirken bir field ismi vermeniz gerekiyor. Kısa olması iyi olur. Alttaki for loop'un sonunda kullanacağız.
### Tweet collection'daki field isimleri için: https://docs.google.com/document/d/1D-TrIfwO2xUHAxtdMKYDyTD3MQm9WgCfzFg_LYu12X0/edit?usp=sharing
### Örneğin "ideology_1", "ideology_2", "welfare",... bunlar şu anda Politus'taki modellerin predictionları için kullandığımız field isimleri.
### Mutlaka dokümantasyondan kontrol edin, var olan bir field ile aynı ismi vermeyin!!!


user_ids = list(pd.read_csv('/data01/alkanberra/final_user_ids.csv', dtype={'_id':str})['_id'])

# User IDler üzerinde for loop
for i, user_id in enumerate(user_ids):
    if i % 100 == 0:
        print(f"{i:,}/{len(user_ids):,} | {i/len(user_ids) * 100:.2f}%")

    user = user_col.find_one({"_id": user_id})
    curr_user_tweet_ids = [tweet.get("id", "") for tweet in user.get("tweets", [])]
    tweets_db = list(tweet_col.find({"_id": {"$in": curr_user_tweet_ids}, "immigration_topic_2": 1}, projection=["_id", "text"]))
    if tweets_db:
        tweets_df = pd.DataFrame(tweets_db)
        tweets_df.rename(columns={"_id": "tweet_id", "text": "tweet_text"}, inplace=True)
        tweets_df["preprocessed_text"] = tweets_df["tweet_text"].apply(preprocess_text)

        predictions = run_model(tweets_df["preprocessed_text"].tolist(), model, tokenizer, batch_size=1000)
        tweets_df["stance_predictions"] = predictions

    # Update the MongoDB collection with new stance predictions
        for tweet_id, prediction in zip(tweets_df["tweet_id"], tweets_df["stance_predictions"]):
            tweet_col.update_one({"_id": tweet_id}, {"$set": {"stance_immigration": prediction}})


print("hayirli olsun")
