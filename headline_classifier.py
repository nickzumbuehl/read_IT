import re
import pandas as pd
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
import numpy as np
import sklearn
from sklearn.naive_bayes import MultinomialNB
import joblib

# loading data
news = pd.read_csv("uci-news-aggregator.csv")

# functions


def get_words(headlines):
    headlines_onlyletters = re.sub("[^a-zA-Z]", " ",headlines)
    words = headlines_onlyletters.lower().split()
    stops = set(stopwords.words("english"))
    meaningful_words = [w for w in words if not w in stops]
    return " ".join(meaningful_words)


cleanHeadlines_train = []

for i in range(0, news.shape[0]):
    cleanHeadline = get_words(news["TITLE"][i])
    cleanHeadlines_train.append(cleanHeadline)

vectorizer = sklearn.feature_extraction.text.CountVectorizer(analyzer="word", max_features=1700)
bagOfWords_train = vectorizer.fit_transform(cleanHeadlines_train)
print(vectorizer.get_feature_names())

X_train = bagOfWords_train.toarray()
Y_train = np.array(news["CATEGORY"])


# make classification :: try XG Boost as well (appears to work best)
nb = MultinomialNB()
nb.fit(X_train, Y_train)

filename = 'headline_classifier.sav'
joblib.dump(nb, filename)

loaded_model = joblib.load(filename)

# Validation of the set
list_of_headlines = ['New cancer medication has been found',
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

series_pred = pd.Series(predictions, name='Prediction').reset_index()
series_headlines = pd.Series(list_of_headlines, name='Headlines').reset_index()
df_results = series_headlines.merge(series_pred, on='index').drop(['index'], axis=1)
print(df_results)


