import re
from functools import lru_cache

import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.base import BaseEstimator, TransformerMixin

_STEMMER = PorterStemmer()
_STOP: set[str] | None = None

# Source "fingerprints" found in EDA. A model should not learn these.
_DATELINE = re.compile(r"^[^\n]{0,80}?\(reuters\)\s*[-–—]*\s*", re.IGNORECASE)
_SOURCE_PATTERNS = [
    re.compile(r"\breuters\b", re.IGNORECASE),
    re.compile(r"featured\s+image(\s+via)?", re.IGNORECASE),
    re.compile(r"pic\.?\s*twitter\.?\s*com/\S+", re.IGNORECASE),
    re.compile(r"pic\W{0,3}twitter\W{0,3}(com\b)?", re.IGNORECASE),
    re.compile(r"twitter\.com/\S+", re.IGNORECASE),
    re.compile(r"\b(photo|image|images|screenshot)\s+(by|via|credit|courtesy)\b", re.IGNORECASE),
    re.compile(r"\bgetty\s+images?\b", re.IGNORECASE),
    re.compile(r"\bdaily\s+(mail|caller)\b", re.IGNORECASE),
    re.compile(r"\bgateway\s+pundit\b", re.IGNORECASE),
    re.compile(r"21\s*st\s*century\s*wire", re.IGNORECASE),
    re.compile(r"\b21\s*wire\b", re.IGNORECASE),
    re.compile(r"\bbecome\s+a\s+member\b", re.IGNORECASE),
    re.compile(r"\bshare\s+this\b", re.IGNORECASE),
    re.compile(r"\b(read\s+more|watch\s+the\s+video|watch\s+below|watch:)", re.IGNORECASE),
    re.compile(r"\bwatch\s+(the\s+)?(full\s+)?video\b", re.IGNORECASE),
    re.compile(r"\bscreen\s*(capture|grab|shot)s?\b", re.IGNORECASE),
    re.compile(r"\bplease\s+(consider\s+)?(donat|like|share|subscribe|follow|support|click|visit)\w*", re.IGNORECASE),
    re.compile(r"\bwith\s+permission\b", re.IGNORECASE),
    # IFND fact-checker boilerplate
    re.compile(r"\bfact[\s-]*check(?:ed|s|ing)?\b", re.IGNORECASE),
    re.compile(r"\bhere.{0,3}s the truth(?: behind (?:this|it))?", re.IGNORECASE),
    re.compile(r"\bdon.{0,3}t fall for (?:this|it)\b", re.IGNORECASE),
    re.compile(r"\bno,? this is not\b", re.IGNORECASE),
]

# Outlet / platform / boilerplate tokens (applied to lowercase tokens before stemming)
_BLOCKLIST = {
    "reuters", "breitbart", "getty", "via", "twitter", "pic",
    "http", "https", "www", "flickr", "screenshot",
    "donate", "donating", "donation", "donations",
    "somodevilla", "mcnamee",
    "youtube", "gateway", "wire", "loading",
}

# Weekday / month words: a source-style signal (e.g. wire-service "said on Friday")
_DATE_WORDS = {
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
    "january", "february", "march", "april", "may", "june", "july", "august",
    "september", "october", "november", "december",
    "jan", "feb", "apr", "jun", "jul", "aug", "sept", "oct", "nov", "dec",
    "thu", "thur", "thurs", "tues", "weekday", "weekdays",
}

_URL = re.compile(r"https?://\S+|www\.\S+")
_NON_LETTERS = re.compile(r"[^a-z]+")


def _stopwords() -> set[str]:
    global _STOP
    if _STOP is None:
        try:
            _STOP = set(stopwords.words("english"))
        except LookupError:
            nltk.download("stopwords", quiet=True)
            _STOP = set(stopwords.words("english"))
    return _STOP


@lru_cache(maxsize=200_000)
def _stem(word: str) -> str:
    return _STEMMER.stem(word)


def clean_text(text: str, strip_source_tags: bool = True, min_token_len: int = 3,
               stem: bool = True, strip_dates: bool = True) -> str:
    text = "" if pd.isna(text) else str(text)

    if strip_source_tags:
        text = _DATELINE.sub("", text)
        for pat in _SOURCE_PATTERNS:
            text = pat.sub(" ", text)

    text = _URL.sub(" ", text)
    # Normalise apostrophes so "don’t" and "don t" end up identical
    text = text.replace("’", "'").replace("‘", "'")
    text = _NON_LETTERS.sub(" ", text.lower())

    drop = set(_stopwords())
    if strip_source_tags:
        drop |= _BLOCKLIST
    if strip_dates:
        drop |= _DATE_WORDS

    tokens = [t for t in text.split() if len(t) >= min_token_len and t not in drop]
    if stem:
        tokens = [_stem(t) for t in tokens]
    return " ".join(tokens)


class TextCleaner(BaseEstimator, TransformerMixin):
    """sklearn-compatible cleaner so preprocessing ships inside the saved model."""

    def __init__(self, strip_source_tags: bool = True, min_token_len: int = 3,
                 stem: bool = True, strip_dates: bool = True):
        self.strip_source_tags = strip_source_tags
        self.min_token_len = min_token_len
        self.stem = stem
        self.strip_dates = strip_dates

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return [clean_text(t, self.strip_source_tags, self.min_token_len,
                           self.stem, self.strip_dates) for t in X]