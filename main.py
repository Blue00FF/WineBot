import pandas as pd
import nltk
import Winebot

# These options are to ensure that the description of the wine is printed to console in its entirety
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 2000)
pd.set_option("display.max_colwidth", None)


def main():
    # Download the necessary information from nltk without printing the process to console thanks to the quiet keyword argument.
    nltk.download("stopwords", quiet=True)
    nltk.download("punkt", quiet=True)
    nltk.download("wordnet", quiet=True)
    nltk.download("omw-1.4", quiet=True)

    # Call the functions defined in the Winebot module.
    user_input = Winebot.Introduction()
    df = Winebot.data_skimmer()
    Winebot.Input_parser(user_input, df)


if __name__ == "__main__":
    main()
