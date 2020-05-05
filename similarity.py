# checking similarity of two article headlines
import pickle
import sklearn
import pandas as pd
import numpy as np

model = pickle.load(open("classifier.p", "rb"))
vectorizer = pickle.load(open("vectorizer.p", "rb"))


input = ['FED decreases interest rates']
input

badofwords_validation = vectorizer.transform(input)
X_validation = badofwords_validation.toarray()  # vector if zero and ones whether word occurd


predictions = model.predict(X_validation)  # making predictions


cosine_similarity = sklearn.metrics.pairwise.cosine_similarity(X_train, X_validation)



df_title_tmp = news.TITLE.reset_index()
df_similarity_tmp = pd.DataFrame(cosine_similarity, columns=["s"]).reset_index()
df_sim = df_title_tmp.merge(df_similarity_tmp, on='index')

df_tmp = df_sim.iloc[df_sim.s.nlargest(10).index].TITLE


# start here

model = pickle.load(open("classifier.p", "rb"))
vectorizer = pickle.load(open("vectorizer.p", "rb"))


# read data
vectorized_data = pd.read_csv("vectorized_news_data.csv")
news = pd.read_csv("uci-news-aggregator.csv")

vectorized_data = vectorized_data.drop([vectorized_data.columns[0]], axis=1)
vectorized_data.shape
news.shape

np.array(vectorized_data)

news = news.TITLE

cosine_similarity = sklearn.metrics.pairwise.cosine_similarity(X_train, X_validation)

df_title_tmp = news.TITLE.reset_index()
df_similarity_tmp = pd.DataFrame(cosine_similarity, columns=["s"]).reset_index()
df_sim = df_title_tmp.merge(df_similarity_tmp, on='index')





