import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import nltk
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
import numpy as np

#pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 2000)
#pd.set_option('display.float_format', '{:20,.2f}'.format)
pd.set_option('display.max_colwidth', None)

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')


df = pd.read_csv('../data/winemag-data_custom.csv')

print("Hello! My name is Winston and I am your personal sommelier! \n")
input_1 = input("What type of wine are you looking for? \n")
notes = input("How would you like your wine to taste like? \n")

country = ""
point = 0
price_1 = 0
price_2 = 0


if re.search("white", input_1):
    colour = "White"
else:
    colour = "Red"


country = re.search("from (?P<country>\w+)", input_1).group("country")

point = int(re.search("(?P<point>\d+) point", input_1).group("point"))

price_1 = int(re.search("(?P<price>\d+) to", input_1).group("price"))

price_2 = int(re.search("(?P<price>\d+) dollars", input_1).group("price"))

if country != "":
    df = df[df['country'] == country]

if point != 0:
    df = df[df['points'] == point]

if price_1 != 0 and price_2 != 0:
    df = df[df['price'].between(price_1, price_2)]
elif price_2 != 0:
    df = df[df['price'].between(price_2 - 10, price_2 + 10)]


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
