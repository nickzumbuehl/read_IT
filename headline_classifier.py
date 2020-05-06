import re
import pandas as pd
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
import numpy as np
import sklearn
from sklearn.naive_bayes import MultinomialNB
import joblib
import pickle
import pydeepl
import requests

# loading data
news = pd.read_csv("uci-news-aggregator.csv")

# functions

def get_words(headlines):
    headlines_onlyletters = re.sub("[^a-zA-Z]", " ", headlines)
    words = headlines_onlyletters.lower().split()
    stops = set(stopwords.words("english"))
    meaningful_words = [w for w in words if not w in stops]
    return " ".join(meaningful_words)


cleanHeadlines_train = []

for i in range(0, news.shape[0]):
    cleanHeadline = get_words(news["TITLE"][i])
    cleanHeadlines_train.append(cleanHeadline)

vectorizer = sklearn.feature_extraction.text.CountVectorizer(analyzer="word", max_features=1700)

# pickle.dump(vectorizer, open("vectorizer.p", "wb"))  # saving vectorizer

bagOfWords_train = vectorizer.fit_transform(cleanHeadlines_train)
print(vectorizer.get_feature_names())

X_train = bagOfWords_train.toarray()
Y_train = np.array(news["CATEGORY"])

# merge X_train and the headline into a Data Frame :: then add column with similarity measure

# df_x = pd.DataFrame(X_train)
# df_x.to_csv('vectorized_news_data.csv')

# make classification :: try XG Boost as well (appears to work best)
nb = MultinomialNB()
nb.fit(X_train, Y_train)

s = pickle.dumps(nb)
pickle.dump(nb, open("classifier.p", "wb"))

# Validation of the set
list_of_headlines = [
    'New cancer medication has been found',
    'Roger Federer wins Wimbledon for the 20th time',
    'IBM sells less devices',
    'climate change causes bush fires',
    'FED decreases interest rate',
    'new wind energy farm build in germany',
    'google develops new AI',
    '23% unemployed people',
    'a heavy flu is hitting the united states',
    'microprocessors become cheaper',
    'Mikaela Shiffrin wins slalom world cup again',
    'China Virus death increases',
    'Goldman Sachs Revenue loss',
    'Diabetes patients are in the risk group'
]

badofwords_validation = vectorizer.transform(list_of_headlines)
X_validation = badofwords_validation.toarray()
predictions = nb.predict(X_validation)

predictions[0]

series_pred = pd.Series(predictions, name='Prediction').reset_index()
series_headlines = pd.Series(list_of_headlines, name='Headlines').reset_index()
df_results = series_headlines.merge(series_pred, on='index').drop(['index'], axis=1)
print(df_results)


sentence = 'I like turtles.'
from_language = 'EN'
to_language = 'ES'

translation = pydeepl.translate(sentence, to_language, from_lang=from_language)


print(translation)

# Using auto-detection
translation = pydeepl.translate(sentence, to_language)
print(translation)

# working with API's in Python

# website: https://xkcd.com
import requests
import json
from newsapi import NewsApiClient
import pandas as pd

# Initiation of API with key
newsapi = NewsApiClient(api_key='4ffaa0cb22f44814800f9b47f3fc176e')

all_articles = newsapi.get_top_headlines(country='ch', q='Corona')

all_articles['articles']
df_corona_articles = pd.DataFrame(all_articles['articles'])

df = pd.DataFrame(all_articles['articles'])
print(df.urlToImage[2])

all_nzz = newsapi.get_everything(sources='Www.nzz.ch', domains=)


# Access DeepL API
url_aut = 'https://api.deepl.com/v2/usage?auth_key=e90be2dd-92b3-920e-5d78-be332af77f0b'

url_trans = 'https://api.deepl.com/v2/translate'
payload = {'auth_key': 'e90be2dd-92b3-920e-5d78-be332af77f0b',
           'text': 'Ich heisse Kevin und bin schwul',
           'target_lang': 'EN'}
r = requests.get(url_trans, params=payload)
print(r.text)

output_text = json.loads(r.text)['translations'][0]
output_text['text']


print(help(r))


r = requests.get('https://xkcd.com')
print(r)
print(dir(r))
print(help(r))

print(r.text)  # html of the website

r_picture = requests.get('https://imgs.xkcd.com/comics/2020_google_trends.png')
print(r_picture.content)  # prints the bytes of the picture

with open('comic.png', 'wb') as f:
    f.write(r_picture.content)


url = 'https://api.exchangeratesapi.io/latest'
r = requests.get(url, params={'base': 'USD'})
print(r.text)  # this is a json file

import json


