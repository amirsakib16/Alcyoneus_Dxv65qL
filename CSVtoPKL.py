import pandas as pd
import numpy as np
movies = pd.read_csv("/content/drive/MyDrive/dataset/tmdb_5000_movies.csv")
credits = pd.read_csv("/content/drive/MyDrive/dataset/tmdb_5000_credits.csv")
df = movies.merge(credits, on="title")
df = df[["movie_id", "title", "overview", "genres", "keywords", "cast", "crew"]]
import json
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem.porter import PorterStemmer
def DICTtoLST(data, cast=False, crew=False):
    try:
        parsed_data = json.loads(data)
        if not cast and not crew:
            return [item["name"].replace(" ","").lower() for item in parsed_data]
        elif cast and not crew:
            return [item["name"].replace(" ","").lower() for item in parsed_data][0:3]
        elif not cast and crew:
            return [item["name"].replace(" ","").lower() for item in parsed_data if item["job"]=="Director"]
    except (json.JSONDecodeError, TypeError, KeyError):
        return []
def TXTtoLST(data):
    return [porterstemmer.stem(i.lower()) for i in data.split()]
def newDataFrame(df):
    df["overview"] = df["overview"].apply(TXTtoLST)
    df["TKN"]= df["overview"]+df["genres"]+df["keywords"]+df["cast"]+df["crew"]
    df["TKN"] = df["TKN"].apply(lambda x:" ".join(x))
    return modify_df(df)
def modify_df(df):
    df = df[["movie_id", "title", "TKN"]]
    return df
def userInput(movie):
    fetchINDEX = df[df["title"].str.lower() == movie.lower()].index[0]
    ORDER = {}
    data = RANK[fetchINDEX]
    for i in range(len(data)):
        ORDER[i] = data[i]
    order = [(i, j) for i, j in sorted(ORDER.items(), key=lambda x: x[1], reverse=True)][1:10]
    for i in order:
        print(df.iloc[i[0]].title)
porterstemmer = PorterStemmer()
df["genres"] = df["genres"].apply(DICTtoLST)
df["keywords"] = df["keywords"].apply(DICTtoLST)
df["cast"] = df["cast"].apply(lambda x: DICTtoLST(x, cast=True, crew=False))
df["crew"] = df["crew"].apply(lambda x: DICTtoLST(x, cast=False, crew=True))
df = newDataFrame(df)
countVector = CountVectorizer(max_features=5000, stop_words='english')
VECTOR = countVector.fit_transform(df["TKN"]).toarray()
RANK = cosine_similarity(VECTOR)
userInput("Batman begins")
#import pickle
#pickle.dump(df, open("information.pkl",'wb'))