import numpy as np
import pandas as pd
import string
import spacy

from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split

from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from sklearn.multiclass import OneVsRestClassifier
from sklearn import metrics

ratingsFile = open("ratings.txt", "rb")
nlp = spacy.load('en_core_web_sm')
punc = string.punctuation
stop_words = spacy.lang.en.stop_words.STOP_WORDS
parser = spacy.lang.en.English()

def spacy_tokenizer(review):
    #tokens = parser(review)
    return [token.text.lower() for token in nlp(review) if token.text not in '\n\n \n\n\n!"-#$%&()--.*+,-/:;<=>?@[\\]^_`{|}~\t\n ' and token.text.lower() not in stop_words]

class predictors(TransformerMixin):
    def transform(self, X, **transform_params):
        return [clean_text(text) for text in X]

    def fit(self, X, y=None, **fit_params):
        return self

    def get_params(self, deep=True):
        return {}

def clean_text(text):
    return text.strip().lower()

tfidf_vector = TfidfVectorizer(tokenizer = spacy_tokenizer)
count_vector = CountVectorizer(tokenizer = spacy_tokenizer)

ratingsList = ratingsFile.readlines()
i = 0
while (i < len(ratingsList)):
    ratingsList[i] = ratingsList[i].decode(errors='replace')
    ratingsList[i] = ratingsList[i].replace("\\u0027","'")
    ratingsList[i] = ratingsList[i].replace("\\u0026","&")
    ratingsList[i] = ratingsList[i].replace("\n","")
    ratingsList[i] = ratingsList[i].replace("\\'","")
    ratingsList[i] = ratingsList[i].replace("\\","")
    split = ratingsList[i].split("||")

    word_tokens = spacy_tokenizer(split[1])
    #print(filtered_sentence)
    if (len(word_tokens) >= 5):
        ratingsList[i] = split
        i += 1
    else:
        ratingsList.remove(ratingsList[i])

ratingsArray = np.asarray(ratingsList)
df = pd.DataFrame(ratingsArray, columns=['rating', 'review'])
print(df.head())

x = df['review']
y = df['rating']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

classifier = LogisticRegression()

pipe = Pipeline([("cleaner", predictors()),
                 ('vectorizer', tfidf_vector),
                 ('classifier', classifier)])


pipe.fit(x_train,y_train)


predicted = pipe.predict(x_test)
print("Logistic Regression Accuracy:",metrics.accuracy_score(y_test, predicted))
print("Logistic Regression Precision:",metrics.precision_score(y_test, predicted, average='micro'))
print("Logistic Regression Recall:",metrics.recall_score(y_test, predicted, average='micro'))