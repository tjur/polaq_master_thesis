"""Preprocess polish wikipedia file to extract proper paragraphs."""

import bz2
import json
import multiprocessing
import pickle
from abc import ABC
from collections import defaultdict
from enum import Enum
from pickle import PicklingError
from typing import Union, Iterable, Tuple, List, Optional
from xml.etree.ElementTree import iterparse
from gensim import utils
from gensim.corpora.wikicorpus import IGNORED_NAMESPACES, get_namespace
from gensim.models.doc2vec import TaggedDocument

from squad_pl import DATA_DIR, logger
from squad_pl.doc2vec.utils import (
    tokenize,
    split_article_into_sentences,
    split_article_into_paragraphs,
    remove_polish_stopwords,
    filter_polish_wiki,
    CATEGORY_SENTENCE_START,
)
from squad_pl.preprocessing.preprocess import find_similar_pages

PLWIKI_PATH = DATA_DIR / "wikipedia/raw/pl/plwiki-20200401-pages-articles-multistream.xml.bz2"


class Tag:
    """Wikipedia paragraph tag."""

    def __init__(self, page_id, sentence_nums, title):
        self.page_id = int(page_id)
        self.sentence_nums = [int(sentence_num) for sentence_num in sentence_nums]
        self.title = title

    def __str__(self):
        return f"{self.page_id}_{';'.join(map(str, self.sentence_nums))}_{self.title}"

    def __repr__(self):
        return f"{self.page_id}_{';'.join(map(str, self.sentence_nums))}_{self.title}"

    @staticmethod
    def from_str(tag: str) -> "Tag":
        page_id, sentence_nums, title = tag.split("_", 2)
        return Tag(page_id=page_id, sentence_nums=sentence_nums.split(";"), title=title)


class SplitType(Enum):
    SENTENCE = "Sentence"
    PARAGRAPH = "Paragraph"


class ParagraphWikiCorpus:
    """
    Class extending TextCorpus with iteration over wikipedia paragraphs, removing stopwords and custom lemmatization.
    """

    def __init__(
        self,
        fname,
        processes=None,
        split_type=SplitType.SENTENCE,
        filter_wiki_func=filter_polish_wiki,
        lemmatize=False,
        filter_namespaces=("0",),
        tokenizer_func=tokenize,
        filter_articles=None,
        remove_stopwords_func=remove_polish_stopwords,
    ):
        """Initialize the corpus.

        It uses the code from gensim.corpora.wikicorpus.WikiCorpus.__init__
        """
        self.fname = fname
        self.filter_namespaces = filter_namespaces
        self.filter_articles = filter_articles
        if processes is None:
            processes = max(1, multiprocessing.cpu_count() - 1)
        self.processes = processes
        self.split_type = split_type
        self.filter_wiki_func = filter_wiki_func
        self.lemmatize = lemmatize
        self.remove_stopwords_func = remove_stopwords_func
        self.tokenizer_func = tokenizer_func

    def get_chunks(self):
        """Iterate over the dump, yielding a list of tokens for each paragraph that passed
        the length and namespace filtering.

        It uses the code from gensim.corpora.wikicorpus.WikiCorpus.get_texts method.
        """
        logger.info("Lemmatization: %s", "no" if self.lemmatize is None else "yes")
        logger.info("Removing stop words: %s", "no" if self.remove_stopwords_func is None else "yes")

        articles = 0
        paragraphs, positions = 0, 0

        tokenization_params = (self.tokenizer_func,)
        texts = (
            (
                text,
                self.filter_wiki_func,
                self.lemmatize,
                self.remove_stopwords_func,
                self.split_type,
                title,
                page_id,
                tokenization_params,
            )
            for title, text, page_id in extract_articles(
                bz2.BZ2File(self.fname), self.filter_namespaces, self.filter_articles
            )
        )

        # unfortunately had to abandon multiprocessing here because `nlp = spacy.load("pl_spacy_model_morfeusz_big")`
        # makes it impossible to share this one object with all processes
        # previous code here:
        # pool = multiprocessing.Pool(self.processes, init_to_ignore_interrupt)

        try:
            # process the corpus in smaller chunks of docs, because multiprocessing.Pool
            # is dumb and would load the entire input into RAM at once...
            page_ids = set()
            for group in utils.chunkize(texts, chunksize=50 * self.processes, maxsize=1):
                for result in map(_process_article, group):  # pool.imap(_process_article, group):
                    for tokens, raw_text, title, page_id, paragraph_num in result:
                        if page_id not in page_ids:
                            page_ids.add(page_id)
                            articles += 1
                            logger.info("Found articles: %s", articles)
                        # article redirects are pruned here
                        if any(title.startswith(ignore + ":") for ignore in IGNORED_NAMESPACES):
                            print(f"{page_id}, {title} is ignored")
                            continue
                        paragraphs += 1
                        positions += len(tokens)
                        yield (tokens, Tag(page_id=page_id, sentence_nums=[paragraph_num], title=title), raw_text)

        except KeyboardInterrupt:
            logger.warn(
                "user terminated iteration over Wikipedia corpus after %i documents and %i paragraphs "
                "with %i positions.",
                articles,
                paragraphs,
                positions,
            )
        except PicklingError as exc:
            raise PicklingError(
                "Can not send filtering function {} to multiprocessing, "
                "make sure the function can be pickled.".format(self.filter_articles)
            ) from exc
        else:
            logger.info(
                "finished iterating over Wikipedia corpus of %i documents (%i paragraphs) with %i positions.",
                articles,
                paragraphs,
                positions,
            )
            self.length = paragraphs  # cache corpus length
        finally:
            # pool.terminate()
            pass


def extract_articles(f, filter_namespaces=False, filter_articles=None):
    """
    Extract articles from a MediaWiki database dump.

    It uses the code from gensim.corpora.wikicorpus.WikiCorpus.extract_pages.

    Parameters
    ----------
    f : file
        File-like object.
    filter_namespaces : list of str or bool
         Namespaces that will be extracted.
    filter_articles : predicate func or None

    Yields
    ------
    tuple of (str or None, str, str)
        Title, text and page id.

    """
    elems = (elem for _, elem in iterparse(f, events=("end",)))

    # We can't rely on the namespace for database dumps, since it's changed
    # it every time a small modification to the format is made. So, determine
    # those from the first element we find, which will be part of the metadata,
    # and construct element paths.
    elem = next(elems)
    namespace = get_namespace(elem.tag)
    ns_mapping = {"ns": namespace}
    page_tag = "{%(ns)s}page" % ns_mapping
    text_path = "./{%(ns)s}revision/{%(ns)s}text" % ns_mapping
    title_path = "./{%(ns)s}title" % ns_mapping
    ns_path = "./{%(ns)s}ns" % ns_mapping
    page_id_path = "./{%(ns)s}id" % ns_mapping

    for elem in elems:
        if elem.tag == page_tag:
            title = elem.find(title_path).text
            text = elem.find(text_path).text

            if filter_namespaces:
                ns = elem.find(ns_path).text
                if ns not in filter_namespaces:
                    text = None

            page_id = elem.find(page_id_path).text

            if filter_articles is not None:
                if not filter_articles(page_id):
                    text = None

            if text:
                yield title, text, page_id

            # Prune the element tree, as per
            # http://www.ibm.com/developerworks/xml/library/x-hiperfparse/
            # except that we don't need to prune backlinks from the parent
            # because we don't use LXML.
            # We do this only for <page>s, since we need to inspect the
            # ./revision/text element. The pages comprise the bulk of the
            # file, so in practice we prune away enough.
            elem.clear()


def process_article(args, tokenizer_func=tokenize):
    """
    Process a Wikipedia article: split, lemmatize, extract tokens, remove stopwords.

    This is the most important function here.
    It has parts of code from gensim.corpora.wikicorpus.WikiCorpus.process_article.

    Returns
    -------
    list of (list of str, str, str, int, int)
        List of tokens from paragraph, raw paragraph, title, page id and paragraph number.

    """
    text, filter_wiki_func, lemmatize, remove_stopwords_func, split_type, title, page_id = args

    if not text:
        return []

    # clear article from markups, templates, files, etc.
    text = filter_wiki_func(text)

    if split_type == SplitType.SENTENCE:
        split = [
            sent
            for sent in split_article_into_sentences(text)
            if not (sent.startswith(CATEGORY_SENTENCE_START) or ".jpg" in sent)
        ]
    elif split_type == SplitType.PARAGRAPH:
        split = split_article_into_paragraphs(text)
    else:
        raise NotImplementedError()

    # tokenize all sentences/paragraphs and lemmatize all tokens if lemmatize=True
    tokenized_paragraphs = [tokenizer_func(paragraph, lemmatize) for paragraph in split]

    assert len(split) == len(tokenized_paragraphs)

    results_to_return = []
    for paragraph_num, result in enumerate(tokenized_paragraphs):
        if remove_stopwords_func is not None:
            result = remove_stopwords_func(result)
        results_to_return.append((result, split[paragraph_num], title, page_id, paragraph_num))

    return results_to_return


def _process_article(args):
    """
    Same as process_article function, but with args in list format.

    It uses the code from gensim.corpora.wikicorpus.WikiCorpus._process_article
    """
    (tokenizer_func,) = args[-1]
    args = args[:-1]

    return process_article(args, tokenizer_func=tokenizer_func)


FILTERED_SQUAD_WIKIPEDIA_PATH = DATA_DIR / "wikipedia/filtered/pl/filtered_squad_wiki.json"


def filter_squad_wikipedia():
    """
    Filter the whole wikipedia file and produce a json file with articles grouped by page_id.

    Left articles have removed markups and are those that might be useful for later polish SQuAD creation
    (articles are the result of `preprocessing.preprocess.find_all_linked_pages(depth=1)`).
    """

    page_ids = find_similar_pages(depth=1)

    def filter_articles(page_id):
        return int(page_id) in page_ids

    result = {}
    for title, text, page_id in extract_articles(
        bz2.BZ2File(PLWIKI_PATH), filter_namespaces=["0"], filter_articles=filter_articles
    ):
        # clear article from markups, templates, files, etc.
        text = filter_polish_wiki(text)
        result[int(page_id)] = text

    with open(FILTERED_SQUAD_WIKIPEDIA_PATH, "w") as filtered_squad_wikipedia_file:
        json.dump(result, filtered_squad_wikipedia_file, indent=4)


def load_squad_wikipedia(page_ids: Optional[Iterable[str]] = None):
    """
    Load SQuAD only wikipedia articles from file.

    If page_ids is specified then the result is filtered to contain only those page IDs
    (if they are part of SQuAD only wikipedia)
    """
    with open(FILTERED_SQUAD_WIKIPEDIA_PATH, "r") as squad_wikipedia_file:
        squad_wikipedia = json.load(squad_wikipedia_file)

    if not page_ids:
        return squad_wikipedia

    return {page_id: squad_wikipedia[page_id] for page_id in page_ids if page_id in squad_wikipedia}


class AbstractTaggedWikiParagraph(ABC):
    def __init__(self):
        ...

    def __iter__(self):
        for paragraph, tag, raw_text in self.wiki.get_chunks():
            yield TaggedDocument(paragraph, [str(tag)]), raw_text

    def save_to_file(self, filename):
        """Process and pickle all result sentences."""
        with open(filename, "wb") as tagged_wiki_file:
            num = 0
            for paragraph, tag, raw_text in self.wiki.get_chunks():
                pickle.dump((TaggedDocument(paragraph, [str(tag)]), raw_text), tagged_wiki_file)
                num += 1
                if num % 10000 == 0:
                    logger.info("Processed %i paragraphs", num)
            logger.info("Finished processing. Processed %i paragraphs in total", num)

    @staticmethod
    def _load_from_file(
        filename, sentence_num=1, full_articles=False, with_raw_text=False
    ) -> Union[Iterable[TaggedDocument], Iterable[Tuple[TaggedDocument, str]]]:
        """
        Load tagged documents and optionally raw text for every element.

        `sentence_num` specifies number of sentences that is required in one tagged document.
        If `full_articles` is True, then `sentence_num` is ignored and full article
        is treated as a one sentence.
        """
        with open(filename, "rb") as tagged_wiki_file:
            if full_articles:
                articles = defaultdict(list)
                while True:
                    try:
                        tagged_document, raw_text = pickle.load(tagged_wiki_file)
                        tag = Tag.from_str(tagged_document.tags[0])
                        articles[str(Tag(page_id=tag.page_id, sentence_nums=[0], title=tag.title))].append(
                            (tag.sentence_nums[0], tagged_document.words, raw_text)
                        )
                    except EOFError:
                        break

                for tag, data in articles.items():
                    article_words = []
                    article_raw_text = ""
                    for _, words, raw_text in sorted(data):
                        article_words += words
                        article_raw_text += " " + raw_text
                    tagged_document = TaggedDocument(article_words, [tag])
                    if with_raw_text:
                        yield tagged_document, article_raw_text
                    else:
                        yield tagged_document

            elif sentence_num == 1:
                while True:
                    try:
                        tagged_document, raw_text = pickle.load(tagged_wiki_file)
                        if with_raw_text:
                            yield tagged_document, raw_text
                        else:
                            yield tagged_document
                    except EOFError:
                        return

            elif sentence_num == 2:
                new_tagged_document, new_raw_text = pickle.load(tagged_wiki_file)
                current_article = [(new_tagged_document, new_raw_text)]
                current_page_id = Tag.from_str(new_tagged_document.tags[0]).page_id

                while True:
                    try:
                        new_tagged_document, new_raw_text = pickle.load(tagged_wiki_file)
                    except EOFError:
                        # yield the last document and return
                        if len(current_article) == 1:
                            if with_raw_text:
                                return current_article[0][0], current_article[0][1]
                            else:
                                return current_article[0][0]

                        for i in range(len(current_article) - 1):
                            tagged_document1, raw_text1 = current_article[i]
                            tagged_document2, raw_text2 = current_article[i + 1]
                            tag1 = Tag.from_str(tagged_document1.tags[0])
                            tag2 = Tag.from_str(tagged_document2.tags[0])
                            assert tag1.page_id == tag2.page_id == current_page_id
                            assert tag1.title == tag2.title
                            tagged_document = TaggedDocument(
                                tagged_document1.words + tagged_document2.words,
                                [str(Tag(current_page_id, tag1.sentence_nums + tag2.sentence_nums, tag1.title))],
                            )
                            if with_raw_text:
                                yield tagged_document, f"{raw_text1} {raw_text2}"
                            else:
                                yield tagged_document
                        return

                    new_page_id = Tag.from_str(new_tagged_document.tags[0]).page_id

                    if new_page_id == current_page_id:
                        current_article.append((new_tagged_document, new_raw_text))
                        continue

                    # yield whole old document
                    if len(current_article) == 1:
                        if with_raw_text:
                            yield current_article[0][0], current_article[0][1]
                        else:
                            yield current_article[0][0]

                    for i in range(len(current_article) - 1):
                        tagged_document1, raw_text1 = current_article[i]
                        tagged_document2, raw_text2 = current_article[i + 1]
                        tag1 = Tag.from_str(tagged_document1.tags[0])
                        tag2 = Tag.from_str(tagged_document2.tags[0])
                        assert tag1.page_id == tag2.page_id == current_page_id
                        assert tag1.title == tag2.title
                        tagged_document = TaggedDocument(
                            tagged_document1.words + tagged_document2.words,
                            [str(Tag(current_page_id, tag1.sentence_nums + tag2.sentence_nums, tag1.title))],
                        )
                        if with_raw_text:
                            yield tagged_document, f"{raw_text1} {raw_text2}"
                        else:
                            yield tagged_document

                    current_article = [(new_tagged_document, new_raw_text)]
                    current_page_id = new_page_id

            else:
                raise NotImplementedError()

    @staticmethod
    def load(filename, join=False, sentence_num=1, with_raw_text=False):
        """Load sentences from file and join sentence_1 and sentence_2 sentences if needed."""
        sentence_nums = [1, 2] if join else [sentence_num]

        for sentence_num in sentence_nums:
            sentence_res = TaggedWikiSentenceWithLemmas._load_from_file(
                filename, sentence_num=sentence_num, with_raw_text=with_raw_text
            )
            for res in sentence_res:
                tag_suffix = f" (sentence_{sentence_num})" if join else ""
                if with_raw_text:
                    (words, [tag]), raw_text = res
                    yield TaggedDocument(words, [tag + tag_suffix]), raw_text
                else:
                    words, [tag] = res
                    yield TaggedDocument(words, [tag + tag_suffix])


class TaggedWiki(AbstractTaggedWikiParagraph):
    def __init__(self, **kwargs):
        self.wiki = ParagraphWikiCorpus(PLWIKI_PATH, lemmatize=False, **kwargs)


class TaggedWikiWithLemmas(AbstractTaggedWikiParagraph):
    def __init__(self, **kwargs):
        self.wiki = ParagraphWikiCorpus(PLWIKI_PATH, lemmatize=True, **kwargs)


class TaggedWikiSentenceWithLemmas(TaggedWikiWithLemmas):
    def __init__(self, depth=0):
        page_ids = find_similar_pages(depth)

        def filter_articles(page_id):
            return int(page_id) in page_ids

        super().__init__(filter_articles=filter_articles, split_type=SplitType.SENTENCE)


TAGGED_WIKI_SENTENCE_DEPTH_WITH_LEMMAS_FILE = str(DATA_DIR / "doc2vec/tagged_wiki_depth_{}_with_lemmas")


if __name__ == "__main__":
    # depth = 0
    # wiki = TaggedWikiSentenceWithLemmas(depth=depth)
    # wiki.save_to_file(TAGGED_WIKI_SENTENCE_DEPTH_WITH_LEMMAS_FILE.format(depth))
    # wiki = list(
    #     TaggedWikiSentenceWithLemmas.load(
    #         TAGGED_WIKI_SENTENCE_DEPTH_WITH_LEMMAS_FILE.format(depth), with_raw_text=False
    #     )
    # )
    pass
