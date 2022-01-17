import pandas as pd
import re
import sklearn.feature_extraction.text as txt
import sklearn.metrics.pairwise as pair
import nltk
import nltk.corpus as corpus
import nltk.stem as stem


class LemmaTokenizer:
    """A class to create a function to eliminate punctuation from a document and tokenize it, thus eliminating the difference between
    capitalised and non-capitalised words.

    Returns:
        function: takes as input the document to tokenise and returns the tokenised document.
    """

    ignore_tokens = [",", ".", ";", ":", '"', "``", "''", "`"]

    def __init__(self):
        self.wnl = stem.WordNetLemmatizer()

    def __call__(self, doc):
        return [
            self.wnl.lemmatize(t)
            for t in nltk.word_tokenize(doc)
            if t not in self.ignore_tokens
        ]


def data_skimmer() -> pd.DataFrame:
    """Since I did not need some of the columns from the csv file downloaded from Kaggle, I drop the unneded columns and
    save the new dataframe to a csv file. In case the new skimmed dataframe has already been created, it just loads it.

    Returns:
        pd.DataFrame: dataframe containing the relevant wine data from the kaggle database.
    """

    try:
        df = pd.read_csv("./archive/winemag-data_custom.csv")
    except FileNotFoundError:
        df = pd.read_csv("./archive/winemag-data-130k-v2.csv", index_col=0)
        df = df.drop(
            [
                "taster_name",
                "taster_twitter_handle",
                "region_1",
                "region_2",
                "province",
                "winery",
            ],
            axis=1,
        )
        df = df.dropna()
        df.to_csv("./archive/winemag-data_custom.csv", index=False)
    return df


def Introduction() -> tuple:
    """The winebot introduces itself and asks the user for input. The input is then returned in form of a tuple.

    Returns:
        tuple: contains the two answers from the questions asked to the user in order to find the desired wine."""

    print("Hello! My name is Winston and I am your personal sommelier! \n")
    input_type = input(
        """What type of wine are you looking for? 
        (include colour, rating out of 100 stars, price range in dollars, from which country it originates) \n"""
    )
    input_type = input_type.lower()
    input_notes = input("How would you like your wine to taste like? \n")
    return (input_type, input_notes)


def Input_parser(user_input: tuple, df: pd.DataFrame) -> None:
    """The input provided by the user is parsed to obtain information like the country of origin,
    the rating, the price range and the colour of the wine.

    Returns:
        None: the function prints to console a message to the user containing the results of the wine search."""

    country = None
    point = 0
    price_1 = 0
    price_2 = 0
    input_type, input_notes = user_input

    if re.search("white", input_type):
        colour = "White"
    else:
        colour = "Red"

    # Here I make some checks in order to verify that the user did indeed include the information asked.
    # The program works even without some of the information but to look for a group where the regex
    # search had not found any results would raise an error.
    if country := re.search("from (?P<country>\w+)", input_type):
        country = country.group("country")

    if point := re.search("(?P<stars>\d+) stars", input_type):
        point = int(point.group("stars"))

    if price_1 := re.search("(?P<price>\d+) to", input_type):
        price_1 = int(price_1.group("price"))

    if price_2 := re.search("(?P<price>\d+) dollars", input_type):
        price_2 = int(price_2.group("price"))

    if country:
        df = df[df["country"].str.lower() == country]

    if point != None:
        df = df[df["points"] == point]

    if price_1 != None and price_2 != None:
        df = df[df["price"].between(price_1, price_2)]
    elif price_2 != None:
        df = df[df["price"].between(price_2 - 10, price_2 + 10)]

    titles = df["title"].to_list()
    descriptions = df["description"].to_list()

    stop_words = set(corpus.stopwords.words("english"))
    tokenizer = LemmaTokenizer()

    # Here we perform the necessary operations for our vector machine to sort the wines in order of similarity to the user input.
    token_stop = tokenizer(" ".join(stop_words))
    vectorizer = txt.TfidfVectorizer(stop_words=token_stop, tokenizer=tokenizer)
    input_notes += " " + colour
    vectors = vectorizer.fit_transform([input_notes] + descriptions)
    cosine_similarities = pair.linear_kernel(vectors[0:1], vectors).flatten()
    description_scores = [item.item() for item in cosine_similarities[1:]]
    score_titles = [(score, title) for score, title in zip(description_scores, titles)]
    sorted_score_titles = sorted(score_titles, reverse=True, key=lambda x: x[0])

    # Here we return the first wine from the list of sorted wine titles if one has been found
    # (i.e. if the dataframe filters have not made the dataframe empty), otherwise returns a
    # message that tells the user that no wine was found according to their specifications.
    try:
        print("The perfect wine for you is " + sorted_score_titles[0][1])
        choice = df[df["title"] == sorted_score_titles[0][1]]
        print(choice["description"].to_string()[7:])
    except IndexError:
        print("I was unable to find any wines to your specifications.")
