import pandas as pd
import nltk
import string
import re
from nltk.corpus import stopwords
from collections import Counter

nltk.download('stopwords')

class ComplimentPreprocessor:
    def __init__(self, path, column_name="Compliment", nrows=None):
        # Load the data from the specified path
        with open(path, 'r', encoding='utf-8') as file:
            lines = file.readlines()

        # Limit the number of rows if specified
        if nrows:
            lines = lines[:nrows]

        # Store the data in a pandas DataFrame
        self.data = pd.DataFrame(lines, columns=[column_name])
        self.column_name = column_name

        # Define punctuation, stopwords, frequent and rare words to remove
        self.PUNCT_TO_REMOVE = string.punctuation
        self.STOPWORDS = set(stopwords.words('english'))
        self.FREQWORDS = set()
        self.RAREWORDS = set()

    def calculate_word_frequencies(self):
        # Count the frequency of each word in the dataset
        cnt = Counter()
        for text in self.data[self.column_name].values:
            for word in text.split():
                cnt[word] += 1
        self.FREQWORDS = set([w for (w, wc) in cnt.most_common(10)])
        n_rare_words = 10
        self.RAREWORDS = set([w for (w, wc) in cnt.most_common()[:-n_rare_words-1:-1]])

    # Define methods to remove unwanted text patterns

    def remove_frequent_words(self, text):
        """Remove words that appear very frequently in the dataset."""
        return " ".join([word for word in str(text).split() if word not in self.FREQWORDS])

    def remove_rare_words(self, text):
        """Remove words that appear very rarely in the dataset."""
        return " ".join([word for word in str(text).split() if word not in self.RAREWORDS])

    def remove_punctuation(self, text):
        """Remove punctuation from the text."""
        return text.translate(str.maketrans('', '', self.PUNCT_TO_REMOVE))

    def remove_stopwords(self, text):
        """Remove common stopwords from the text."""
        return " ".join([word for word in str(text).split() if word not in self.STOPWORDS])

    def remove_emoji(self, text):
        """Remove emojis from the text."""
        emoji_pattern = re.compile("["
                                   u"\U0001F600-\U0001F64F"  # emoticons
                                   u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                   u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                   u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                   u"\U00002702-\U0001F251"
                                   "]+", flags=re.UNICODE)
        return emoji_pattern.sub(r'', text)

    def remove_urls(self, text):
        """Remove URLs from the text."""
        url_pattern = re.compile(r'https?://\S+|www\.\S+')
        return url_pattern.sub(r'', text)

    def remove_initial_numbers(self, text):
        """Remove initial numbers that appear at the start of a text."""
        return re.sub(r"^\d+\.?", "", text).strip()


    def preprocess(self):
        # Filter out rows that do not start with a number.
        # Commented out to test uploaded data without numbered lists.
        #self.data = self.data[self.data[self.column_name].str.match(r'^\d+\.')]

        # Drop rows with missing data
        self.data = self.data.dropna(subset=[self.column_name])
        # Convert column to string type
        self.data[self.column_name] = self.data[self.column_name].astype(str)

        # Convert text to lowercase
        self.data[self.column_name] = self.data[self.column_name].str.lower()

        # Apply various preprocessing functions
        self.data[self.column_name] = self.data[self.column_name].apply(self.remove_initial_numbers)
        self.data[self.column_name] = self.data[self.column_name].apply(self.remove_emoji)
        self.data[self.column_name] = self.data[self.column_name].apply(self.remove_urls)
        self.data[self.column_name] = self.data[self.column_name].apply(self.remove_punctuation)

        # Calculate word frequencies after initial preprocessing
        self.calculate_word_frequencies()

        # Remove frequent and rare words
        #self.data[self.column_name] = self.data[self.column_name].apply(self.remove_frequent_words)
        self.data[self.column_name] = self.data[self.column_name].apply(self.remove_rare_words)

    def save_processed(self, output_path):
        """Save the processed data to a specified path."""
        self.data.to_csv(output_path, sep="\n", index=False)


# Usage:
preprocessor = ComplimentPreprocessor("compliment.csv", column_name="Compliment")
preprocessor.preprocess()
preprocessor.save_processed("processed_compliment.csv")
