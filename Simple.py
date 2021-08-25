import pandas as pd
import pyinputplus as pyip
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import nltk
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
import numpy as np

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 2000)
pd.set_option('display.float_format', '{:20,.2f}'.format)
pd.set_option('display.max_colwidth', None)

df = pd.read_csv('../data/winemag-data_custom.csv')

print("Hello! My name is Winston and I am your personal sommelier! \n")
colour = pyip.inputChoice(['Red', 'White'], prompt="Are you looking for a red or a white wine? \n")
country = pyip.inputStr(prompt="Are you looking for a wine that originates from which country? \n")
point = pyip.inputNum(prompt="Whereabout should it be in the point range? \n")
price = pyip.inputNum(prompt="How much would you like to spend? \n")
notes = pyip.inputStr(prompt="Which notes should your wine have? \n")

df = df[df['country'] == country]
df = df[df['points'] == point]
df = df[df['price'].between(price - 10, price + 10)]


class LemmaTokenizer:
    ignore_tokens = [',', '.', ';', ':', '"', '``', "''", '`']

    def __init__(self):
        self.wnl = WordNetLemmatizer()

    def __call__(self, doc):
        return [self.wnl.lemmatize(t) for t in word_tokenize(doc) if t not in self.ignore_tokens]


variety = df['variety'].to_list()
titles = df['title'].to_list()
descriptions = df['description'].to_list()

stop_words = set(stopwords.words('english'))
tokenizer = LemmaTokenizer()

token_stop = tokenizer(' '.join(stop_words))
vectorizer = TfidfVectorizer(stop_words=token_stop, tokenizer=tokenizer)
notes = notes + " " + colour
vectors = vectorizer.fit_transform([notes] + descriptions)
cosine_similarities = linear_kernel(vectors[0:1], vectors).flatten()
document_scores = [item.item() for item in cosine_similarities[1:]]
score_titles = [(score, title) for score, title in zip(document_scores, titles)]
sorted_score_titles = sorted(score_titles, reverse=True, key=lambda x: x[0])

try:
    print("The perfect wine for you is " + sorted_score_titles[0][1])
    print((df[df['title'] == sorted_score_titles[0][1]]['description'].to_string())[9:])
except IndexError:
    print("I was unable to find any wines to your specifications.")
