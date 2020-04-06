"""Natural Language Processing"""
import string
import warnings

import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import BaseEstimator, TransformerMixin

try:
    import nltk
    from nltk.corpus import stopwords as sw
except ImportError:
    nltk = None
    sw = None


def _identity(x):
    """An identity function"""
    return x


# List of supported languages, including ISO 639-1 (two-letter codes), ISO 639-2/T (three-letter codes), and ISO name
languages = {
    "es": "spanish", "spa": "spanish", "spanish": "spanish",
    "en": "english", "eng": "english", "english": "english",
    "pt": "portuguese", "por": "portuguese", "portuguese": "portuguese",
    "fr": "french", "fra": "french", "french": "french",
    "de": "german", "deu": "german", "german": "german",
}


class Tokenizer:
    """A generic text tokenizer"""

    def __init__(self):
        pass

    def __call__(self, document):
        """
        Break a single document into tokens.

        Args:
            document (str): The original document to tokenize

        Yields:
            str: The next token.

        """
        raise NotImplementedError("Class must be inherited")


class TrivialTokenizer(Tokenizer):
    """A trivial tokenizer, splitting by any whitespace"""

    def __init__(self):
        super().__init__()

    def __call__(self, document):
        return document.split()


class NLTKTokenizer(Tokenizer):
    """A NLTK-based preprocessor, transforming sentences into lists of tokens"""

    def __init__(self, lemmatizer="snowball", language="english", stopwords=None, punctuation=None,
                 ignore_numbers=True, min_token_length=1):
        """
        Args:
            lemmatizer (str): Lemmatizer or stemmer to use. Available options are "None", "Porter" (default),
                              and "Snowball". The values are case-insensitive.
            language (str): Language used to tune the lemmatizer and the default stopwords.
            stopwords (list of str): A list of stopwords. If None, use a default one depending on the language
            punctuation (list of str): A list of punctuation characters.
            ignore_numbers (bool): Whether to drop numbers.
            min_token_length (int): Minimum length required to keep a token.

        """
        super().__init__()
        if nltk is None:
            raise ModuleNotFoundError("The nltk module is not available. Install it to use NLTKTokenizer.")
        self.stopwords = stopwords
        self.language = languages[language.lower()]
        self.punctuation = set(punctuation) if punctuation is not None else set(string.punctuation)
        self.lemmatizer = lemmatizer.lower()
        self.min_token_length = min_token_length
        self.drop_numbers = ignore_numbers

        if self.stopwords is None:
            try:
                self.stopwords = set(sw.words(self.language))
            except OSError:
                warnings.warn("Stopwords not found for language %s." % self._language)
                self.stopwords = {}

        if self.lemmatizer == "none":
            self._lemmatizer = _identity
        elif self.lemmatizer == "porter":
            self._lemmatizer = nltk.stem.porter.PorterStemmer().stem
            if self.language != "english":
                warnings.warn("Using a Porter Stemmer with a language different from English.")
        elif self.lemmatizer == "snowball":
            try:
                self._lemmatizer = nltk.stem.snowball.SnowballStemmer(self.language).stem
            except ValueError:
                warnings.warn("Invalid language %s for the Snowball Stemmer." % self.language)
                self._lemmatizer = _identity
        else:
            raise ValueError("Invalid lemmatizer selected.")

    def __call__(self, document):
        """
        Break a document into tokens.

        Args:
            document (str): The original document to tokenize

        Yields:
            str: The next token.

        """
        for token in nltk.word_tokenize(document):
            # Skip if too short
            if len(token) < self.min_token_length:
                continue

            # Skip if stopword
            if token in self.stopwords:
                continue

            # Skip if punctuation
            if all(char in self.punctuation for char in token):
                continue

            # Skip if number and dropping numbers
            if self.drop_numbers:
                try:
                    float(token)
                    continue
                except ValueError:
                    pass

            # Lemmatize the token and yield it
            yield self._lemmatizer(token)


class TextVectorizer(TransformerMixin, BaseEstimator):
    """A document vectorizer"""

    def __init__(self, tokenizer=None, tfidf_kwargs=None):
        """
        Args:
            tokenizer (Tokenizer): A tokenizer instance.
            tfidf_kwargs (dict): Kwargs for the TfidfVectorizer component.
        """
        self.tokenizer = tokenizer
        self.tfidf_kwargs = tfidf_kwargs

        self._tokenizer = None
        self._tfidf = None

    def fit_transform(self, X, y=None, **fit_params):
        # For consistency, input must be a DataFrame (or a square matrix)
        # However, TfidfVectorizer expects a collection of row documents
        # Perform this conversion:
        if isinstance(X, pd.DataFrame):
            return self.fit_transform(X.iloc[:, 0])

        # Prepare the tokenizer
        if self.tokenizer is None:
            self._tokenizer = TrivialTokenizer()
        else:
            self._tokenizer = self.tokenizer

        kwargs = self.tfidf_kwargs if self.tfidf_kwargs else {}
        self._tfidf = TfidfVectorizer(tokenizer=self._tokenizer, **kwargs)

        self._tfidf.fit(X)
        result = self._tfidf.transform(X)

        # Fix the vocabulary for future calls
        self._tfidf.vocabulary = self._tfidf.vocabulary_

        # Reduce pickling size (cf. TfidfVectorizer docs)
        self._tfidf.stop_words_ = None

        return result

    def fit(self, X):
        # Must actually transform with TFIDF to fix the vocabulary
        self.fit_transform(X)
        return self

    def transform(self, X):
        # Cf. fit_transform method
        if isinstance(X, pd.DataFrame):
            return self.transform(X.iloc[:, 0])
        return self._tfidf.transform(X)
