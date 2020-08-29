import re

from typing import List

import spacy
from gensim.corpora.wikicorpus import filter_wiki
from spacy.lang.pl import Polish, PolishDefaults


TOKEN_MIN_LEN = 1  # must be 1 to not ignore one-digit numbers
TOKEN_MAX_LEN = 16
PARAGRAPH_MIN_LEN = 300  # minimal number of characters that we want to have in paragraph; it is quite small because we
# don't want to accidentally remove some important information from text
# sometimes whole article is just a short stub - then we don't want it
MAX_UNIQUE_TOKENS = 4_000_000  # max number of unique tokens in a frequency dictionary

REDUNDANT_PL_END_SECTIONS = ["Przypisy", "Zobacz też", "Linki zewnętrzne"]
TABLE_TEMPLATE_LEFT = re.compile(r"\{\|")
TABLE_TEMPLATE_RIGHT = re.compile(r"\|\}")
FILE_MARKUP_START = re.compile(r"\[\[:?([fF]ile|[iI]mage|[pP]lik|[mM]edia]):")
FILE_BRACKETS = re.compile(r"\[\[|\]\]")
FORMATNUM_MARKUP = r"{{formatnum:(?P<number>[\w ,\.]+)}}"
CATEGORY_SENTENCE_START = "Kategoria:"


def filter_polish_wiki(raw):
    """
    Filter out redundant parts from polish wiki and pass result
    to gensim.corpora.wikicorpus.filter_wiki to remove markups.

    Parameters
    ----------
    raw : str
        Unicode or utf-8 encoded string.

    Returns
    -------
    str
        `raw` without markup.
    """
    # trim the end of the article (redundant sections like 'See also')
    pos = min(
        [
            match.start()
            for match in [re.search(rf"== ?{section} ?==", raw) for section in REDUNDANT_PL_END_SECTIONS]
            if match is not None
        ],
        default=len(raw),
    )
    raw = raw[:pos]

    # remove tables
    # filter_wiki already removes templates that start with {{ and ends with }}
    # however, tables in wikipedia are marked as {| and |}
    # so change it to {{ and }} and let filter_wiki finish the work
    raw = re.sub(TABLE_TEMPLATE_LEFT, "{{", raw)
    raw = re.sub(TABLE_TEMPLATE_RIGHT, "}}", raw)

    # filter_wiki just replaces file with its caption but we want here to fully remove all files and captions
    file_markup_match = FILE_MARKUP_START.search(raw)
    while file_markup_match is not None:
        start = file_markup_match.start()
        brackets = 0
        while True:
            bracket_match = FILE_BRACKETS.search(raw, start)
            found_bracket = bracket_match.group()[0]
            if found_bracket == "[":
                brackets += 1
            elif found_bracket == "]":
                brackets -= 1
            end = bracket_match.end()
            if brackets == 0:
                raw = raw[: file_markup_match.start()] + raw[end:]
                start = file_markup_match.start()
                file_markup_match = FILE_MARKUP_START.search(raw, start)
                break
            start = end

    # process formatnum markups - these markups usually would be removed by filter_wiki called below
    raw = process_formatnum_markups(raw)

    # filter_wiki removes markups, templates, links
    filtered = filter_wiki(raw)

    filtered = filtered.replace("'''", "")  # remove wrapping of a first word in an article (definition word)
    filtered = filtered.replace("''", "")  # remove italics wrapping
    filtered = filtered.replace("  ", " ")  # double spaces
    filtered = filtered.replace("\xa0", " ")
    filtered = filtered.replace("\u00a0", " ")  # some wikipedia markup leftover
    return filtered


def process_formatnum_markups(text):
    """
    Find all formatnum markups and extract numbers from them.

    formatnum is a markup that displays numbers with extra spaces to make it look good.
    (e.g. "{{formatnum:710231}}" -> "710 231")
    """
    text = re.sub(FORMATNUM_MARKUP, "\g<number>", text)
    return text


H_REGS = H2_REG, H3_REG, H4_REG, H5_REG, H6_REG = [re.compile(rf"{i*'='}.+?{i*'='}") for i in range(2, 7)]
H_REG = re.compile(r"={2,7}(?P<header>.+?)={2,7}")
# a single = is styled as the article title and should not be used within an article.
# headings 4,5 and 6 have the same size (https://en.wikipedia.org/wiki/Help:Wikitext)
H2, H3, H4, H5, H6 = [f"#H{i}#" for i in range(2, 7)]


def remove_headers(text):
    """Remove all headers from article."""
    for header_regex in H_REGS[::-1]:
        text = re.sub(header_regex, "", text)
    return text


def split_article_into_paragraphs(text) -> List[str]:
    """
    Take an article and return all its paragraphs.
    :return:
    list of str
    """
    # text = re.sub(H6_REG, H6, text)
    # text = re.sub(H5_REG, H5, text)
    # text = re.sub(H4_REG, H4, text)
    # text = re.sub(H3_REG, H3, text)
    # text = re.sub(H2_REG, H2, text)
    #
    # if len(text) < PARAGRAPH_MIN_LEN:
    #     return []
    #
    # paragraphs = []
    #
    # for h2 in text.split(H2):
    #     for h3 in h2.split(H3):
    #         for h4 in h3.split(H4):
    #             for h5 in h4.split(H5):
    #                 for h6 in h5.split(H6):
    #                     if len(h6) >= PARAGRAPH_MIN_LEN:
    #                         paragraphs.append(h6)
    #
    # return paragraphs
    return NotImplementedError()


nlp_sentencizer = Polish()  # just the language with no model
sentencizer = nlp_sentencizer.create_pipe("sentencizer")
nlp_sentencizer.add_pipe(sentencizer)


def split_article_into_sentences(text) -> List[str]:
    """
    Take an article and return all its sentences.
    :return:
    list of str
    """
    text = remove_headers(text)

    # NLTK sentence splitting
    # https://www.nltk.org/_modules/nltk/tokenize.html#sent_tokenize
    # https://stackoverflow.com/a/35279885
    # nltk punkt package already has pretrained polish tokenizer (trained on Polish National Corpus, ~1000000 articles)
    # however, spaCy gets better results in polish sentence splitting
    # sentences = nltk.tokenize.sent_tokenize(text, language="polish")

    # spaCy sentence splitting
    # https://spacy.io/usage/linguistic-features#sbd
    # rule-based
    sentences = [sent.text.strip() for t in text.split("\n") for sent in nlp_sentencizer(t).sents]
    return sentences


WORD_REGEX = re.compile(r"[\w%,\'\.:;\'\-\‐\−]+", re.UNICODE)
PUNCTUATION = {",", ".", ":", ";", "'", "-", "‐", "−"}


# def clear_text(text, token_min_len=1, token_max_len=100):
#     """
#     Clear text without markups from redundant characters like punctuation marks, parentheses and
#     some wikipedia markup leftovers (''').
#
#     It uses the code from gensim.corpora.wikicorpuss.tokenize.
#     """
#     return " ".join([
#         match.group() for match in WORD_REGEX.finditer(text)
#         if token_min_len <= len(match.group()) <= token_max_len and not match.group().startswith('_')
#     ])


# morf = morfeusz2.Morfeusz(case_handling=morfeusz2.IGNORE_CASE)


# def polish_lemmatization(word):
#     """Return lexically smallest lowered lemma of a word ."""
#     # this one uses plain morfeusz2
#     return sorted([res[2][1].lower() for res in morf.analyse(word)])[0].split(":", 1)[0]


# This commented code below is an attempt to make spacy.load("pl_spacy_model_morfeusz_big") work
# in `multiprocessing` environment
# It unfortunately fails (probably a problem with a model and keras/terraform packages used there)

# from multiprocessing.managers import BaseManager
#
#
# class MyManager(BaseManager):
#     ...
#
#
# def Manager():
#     m = MyManager()
#     m.start()
#     return m
#
#
# class NLPWrapper:
#     def __init__(self):
#         self.nlp = spacy.load("pl_spacy_model_morfeusz_big")
#
#     def process(self, text):
#         return self.nlp(text)
#
#
# MyManager.register('NLPWrapper', NLPWrapper)
#
# manager = Manager()
# nlp_wrapper = manager.NLPWrapper()


# https://github.com/ipipan/spacy-pl
polish_nlp = spacy.load("pl_spacy_model_morfeusz_big")


def tokenize(text, lemmatize=True) -> List[str]:
    """Tokenize text and lemmatize it optionally."""
    # use polish model for spaCy that uses morfeusz2 for lemmatization
    # https://github.com/ipipan/spacy-pl
    if not text:
        return []
    try:
        doc = polish_nlp(text)
    except KeyError:
        return []

    tokenized_text = [token.lemma_.lower() if lemmatize else token.text.lower() for token in doc]
    tokenized_text = [token for token in tokenized_text if WORD_REGEX.fullmatch(token) and token not in PUNCTUATION]
    return tokenized_text


def remove_polish_stopwords(tokens):
    # use spaCy stopwords
    return [token for token in tokens if token not in PolishDefaults.stop_words]
