from tokenize import String
import pandas as pd
from fastapi import Depends, APIRouter
from pydantic import BaseSettings
from fastapi.staticfiles import StaticFiles
import csv
import numpy as np
from asyncio import coroutine, run
import os
import nltk
import re
import string
import spacy
from nltk.stem.snowball import SnowballStemmer
from nltk.sentiment import SentimentAnalyzer
from textblob import TextBlob
import warnings
from pydantic import BaseModel
warnings.filterwarnings('ignore')
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer
from textblob import Blobber
from textblob_fr import PatternTagger, PatternAnalyzer
import datetime
import json
from typing import List, Optional

# nltk.download('twitter_samples')
# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('averaged_perceptron_tagger')

# nltk.download('stopwords')
# nltk.download('omw-1.4')
wn = nltk.WordNetLemmatizer()
stemmer = SnowballStemmer(language='french')
missed_stopwords ={'alors','au','aucuns','aussi','autre','avant','avec','avoir','bon','car','ce','cela','ces','ceux','chaque','ci','comme','comment','dans','dedans','dehors','depuis','des','devrait','doit','donc','dos','du','début','elle','elles','en','encore','essai','est','et','eu','fait','faites','fois','font','hors','ici','il','ils','je','juste','la','le','les','leur','là','ma','maintenant','mais','mes','mien','moins','mon','mot','même','ni','nommés','notre','nous','ou','où','par','parce','pas','peu','peut','plupart','pour','pourquoi','quand','que','quel','quelle','quelles','quels','qui','sa','sans','ses','seulement','si','sien','son','sont','sous','soyez','sujet','sur','ta','tandis','tellement','tels','tes','ton','tous','tout','trop','très','tu','voient','vont','votre','vous','vu','ça','étaient','état','étions','été','être'}
stopwords = set(stopwords.words('french')).union(set(missed_stopwords))
nlp = spacy.load("fr_core_news_sm")

router = APIRouter()

#Base models
class Item(BaseModel):
    nb_tweet: float
    total_nb_likes:float
    total_nb_RT:float
    
class Item_best_like(BaseModel):
    nlikes: float
    tweet:str

class Item_best_retweets(BaseModel):
    nretweets:float
    tweet:str

class Item_sentiment(BaseModel):
    name: str
    Sentiment:float

class Item_topic(BaseModel):
    topic: float
    words:list



# Cleans text
def remove_emoji(string):
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002500-\U00002BEF"  # chinese char
                               u"\U00002702-\U000027B0"
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               u"\U0001f926-\U0001f937"
                               u"\U00010000-\U0010ffff"
                               u"\u2640-\u2642"
                               u"\u2600-\u2B55"
                               u"\u200d"
                               u"\u23cf"
                               u"\u23e9"
                               u"\u231a"
                               u"\ufe0f"  # dingbats
                               u"\u3030"
                               "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', string)

def clean_text(text) -> coroutine:
    text = str(text).lower()  # Lowercase words
    text = re.sub(r"\[(.*?)\]", "", text)  # Remove [+XYZ chars] in content
    text = re.sub(r"(?:\@|http?\://|https?\://|www)\S+", "", text)  # Remove http links
    text = re.sub(r"\s+", " ", text)  # Remove multiple spaces in content
    text = re.sub(r"\w+…|…", "", text)  # Remove ellipsis (and last word)
    text = re.sub(r"(?<=\w)-(?=\w)", " ", text)  # Replace dash between words
    text = re.sub(r'[^A-Za-z0-9éàèç]+',' ',text)
    text= re.sub('[0-9]+', '', text)
    text =re.sub(r'\b\w{1,4}\b', '', text)
    text = re.sub(r"(é|è|ê|ë)", "e", text)
    text = re.sub(r"(à|â|ä)", "a", text)
    text = re.sub(r"(î|ï)", "i", text)
    text = re.sub(r"(ô|ö)", "o", text)
    text = re.sub(r"(ù|û|ü)", "u", text)
    text = re.sub( f"[{re.escape(string.punctuation)}]", "", text)  # Remove punctuation
    text=' '.join([word for word in text.split() if word not in stopwords])
    text=  remove_emoji(text)
    return text

def clean_col(col) :
  return col.map( clean_text)

async def stem_text(sentence):
    doc = nlp(sentence)
    return [stemmer.stem(X.text) for X in doc]

# Functions
def getSubjectivity(text):
    return TextBlob(text).sentiment.subjectivity

def getPolarity(text):
    return TextBlob(text).sentiment.polarity

def getAnalysis(score):
    if score < 0:
        return 'negative'
    elif score == 0:
        return 'neutral'
    else:
        return 'positive'
def sentiment_analysis(tweet):
     tweet["Subjectivity"] = tweet["cleaned_tweet"].apply(getSubjectivity)
     tweet ["Polarity"] = tweet["cleaned_tweet"].apply(getPolarity)
     tweet ["Analysis"] = tweet  ["Polarity"].apply(getAnalysis )
     return tweet

def get_nb_analysis(data):
    tb = Blobber(pos_tagger=PatternTagger(), analyzer=PatternAnalyzer())
    senti_list = [0, 0, 0]
    for i in data["cleaned_tweet"]:
        vs = tb(i).sentiment[0]
        if (vs > 0):
             senti_list[0]+=1
        elif (vs < 0):
             senti_list[1]+=1
        else:
            senti_list[2]+=1

    d = [
    {
      "name": 'Positives',
      "Sentiment": senti_list[0],
    },
    {
      "name": 'Neutres',
      "Sentiment": senti_list[2],

    },
    {
      "name": 'Négatives',
      "Sentiment": senti_list[1],

    }]
    return d
    

def get_top20_occurnace(Data):
    all_words = ', '.join(Data.cleaned_tweet).lower().split()
    freq = nltk.FreqDist(all_words)
    top20=freq.most_common(10)
    return top20

def topic_modeling(tweet, number):

    text = tweet['tweet'].apply(lambda x: clean_text(x))
    # use tfidf by removing tokens that don't appear in at least 50 documents
    vect = TfidfVectorizer(min_df=3, stop_words='english')
    X = vect.fit_transform(text)
    # NMF
    model = NMF(n_components=number, random_state=5)

    # Fit the model to TF-IDF
    model.fit(X)

    feat_names = vect.get_feature_names()

    word_dict = {}
    weight_dict ={}

    #print(model.components_)
    #print(len(model.components_))
    #print(len(model.components_[0]))
    res = []
    for i in range(number):

        #for each topic, obtain the largest values, and add the words they map to into the dictionary.
        words_ids = model.components_[i].argsort()[:-10 - 1:-1]
        tmp=[]
        for key in words_ids:
          tmp.append(
              {
                  "word": feat_names[key],
                  "weight": model.components_[i][key]
              }
          )
        res.append(tmp)
    return res
def topic_format(topic_dct):
  res = []
  for i in topic_dct.items():
    res.append({
        "topic": i[0],
        "words" : i[1]
    })
  return res

# KPI's
def KPI_nbs(df):
  nb_tweet = df.shape[0]
  total_nb_likes = df["nlikes"].sum()
  total_nb_RT = df["nretweets"].sum()

  res = {
      "nb_tweet": nb_tweet,
      "total_nb_likes": total_nb_likes,
      "total_nb_RT": total_nb_RT
  }

  return (res)

def KPI_most_liked(df):
  most_liked_tweet = df.iloc[df["nlikes"].idxmax()].to_dict()
  res = {
      "nlikes": most_liked_tweet["nlikes"],
      "tweet": most_liked_tweet["tweet"]
  }
  return res

def KPI_most_retweeted(df):
  most_RT_tweet = df.iloc[df["nretweets"].idxmax()].to_dict()
  res = {
      "nretweets": most_RT_tweet["nretweets"],
      "tweet": most_RT_tweet["tweet"]
  }
  return res

def KPI_date_count(df):
  date_counts = df["date"].map(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S').date()).value_counts().to_frame()
  date_counts.reset_index(inplace = True)
  date_counts.rename(columns = {'date':'count', 'index': "date"}, inplace = True)

  return date_counts



# Routes
@router.get("/top_occurence/{df_name}")
async def info(df_name):
    p = r'C:\Users\nassi\Desktop\PPD_Tweets\PPD\app\Dataset'
    full_path=os.path.join(p, '.'.join([df_name, 'csv']))
    df = pd.read_csv(full_path,sep=';')
    df['cleaned_tweet']= clean_col(df['tweet'])
    print("alller",get_top20_occurnace(df))
    return(get_top20_occurnace(df))
   
@router.get("/sentiments_analysis/{df_name}",response_model=List[Item_sentiment])
async def info(df_name):
    p = r'C:\Users\nassi\Desktop\PPD_Tweets\PPD\app\Dataset'
    full_path=os.path.join(p, '.'.join([df_name, 'csv']))
    df = pd.read_csv(full_path,sep=';')
    df['cleaned_tweet']= clean_col(df['tweet'])
    return(get_nb_analysis(df))
   

   
@router.get("/topic_modeling/{df_name}")
async def info(df_name):
    p = r'C:\Users\nassi\Desktop\PPD_Tweets\PPD\app\Dataset'
    full_path=os.path.join(p, '.'.join([df_name, 'csv']))
    df = pd.read_csv(full_path,sep=';')
    df['cleaned_tweet']= clean_col(df['tweet'])
    resultat = topic_modeling(df,3)
    # print(topic_format(resultat))
    return (resultat)

@router.get("/topic_modeling/")
async def info():
    p = r'C:\Users\nassi\Desktop\PPD_Tweets\PPD\app\Dataset'
    full_path=os.path.join(p, '.'.join(['em', 'csv']))
    df = pd.read_csv(full_path,sep=';')
    full_path2=os.path.join(p, '.'.join(['ml', 'csv']))
    df2 = pd.read_csv(full_path,sep=';')
    frames=[df,df2]
    result = pd.concat(frames)
    df = result
    df['cleaned_tweet']= clean_col(df['tweet'])
    resultat = topic_modeling(df,3)
    # print(topic_format(resultat))
    return (resultat)


@router.get("/kpi/{df_name}",response_model=Item)
async def info(df_name):
    p = r'C:\Users\nassi\Desktop\PPD_Tweets\PPD\app\Dataset'
    full_path=os.path.join(p, '.'.join([df_name, 'csv']))
    df = pd.read_csv(full_path,sep=';')
    return(KPI_nbs(df))

@router.get("/kpi/",response_model=Item)
async def info():
    p = r'C:\Users\nassi\Desktop\PPD_Tweets\PPD\app\Dataset'
    full_path=os.path.join(p, '.'.join(['em', 'csv']))
    df = pd.read_csv(full_path,sep=';')
    full_path2=os.path.join(p, '.'.join(['ml', 'csv']))
    df2 = pd.read_csv(full_path,sep=';')
    frames=[df,df2]
    result = pd.concat(frames)
    df = result
    
    return(KPI_nbs(df))
@router.get("/kpi_best_likes/{df_name}",response_model=Item_best_like)
async def info(df_name):
    p = r'C:\Users\nassi\Desktop\PPD_Tweets\PPD\app\Dataset'
    full_path=os.path.join(p, '.'.join([df_name, 'csv']))
    df = pd.read_csv(full_path,sep=';')
    return(KPI_most_liked(df))
    
@router.get("/kpi_best_retweets/{df_name}",response_model=Item_best_retweets)
async def info(df_name):
    p = r'C:\Users\nassi\Desktop\PPD_Tweets\PPD\app\Dataset'
    full_path=os.path.join(p, '.'.join([df_name, 'csv']))
    df = pd.read_csv(full_path,sep=';')
    return(KPI_most_retweeted(df))
    
   
   