"""Evaluate paragraphs similarity using trained models."""
from typing import List, Union, Tuple

from gensim.models import Doc2Vec

from squad_pl import DATA_DIR
from squad_pl.doc2vec.preprocess import Tag, remove_polish_stopwords
from squad_pl.doc2vec.train import EpochSaver, EPOCHS  # import required to load Doc2Vec models without errors
from squad_pl.doc2vec.utils import tokenize

MODEL_PATH = DATA_DIR / "doc2vec/models"


def load_model(model_path):
    model = Doc2Vec.load(str(model_path))
    # replace cannot be True because then infer_vector training is impossible
    model.docvecs.init_sims(replace=False)
    return model


def normalize_text(text: str, lemmatize=True, remove_stopwords=True) -> List[str]:
    """Transform raw string into list of words."""
    if remove_stopwords:
        return remove_polish_stopwords(tokenize(text, lemmatize))
    return tokenize(text, lemmatize)


def _get_similar_sentences(
    text, page_ids, model=None, lemmatize=True, remove_stopwords=True, topn=20, only_tags=True, normalized_already=False
) -> List[Union[Tag, Tuple[Tag, float, int]]]:
    """
    Get topn most similar sentences using gensim.

    If page_ids is specified then result is filtered to have only sentences from these articles.
    `normalized_already` specifies whether `text` parameter is already a list of tokens from the sentence
    or it is just a raw string that need to be normalized.
    """
    if normalized_already:
        inferred_vector = model.infer_vector(text, epochs=EPOCHS)
    else:
        inferred_vector = model.infer_vector(normalize_text(text, lemmatize, remove_stopwords), epochs=EPOCHS)
    result = model.docvecs.most_similar([inferred_vector], topn=topn)
    result = [(Tag.from_str(tag), sim, place) for place, (tag, sim) in enumerate(result)]
    if only_tags:
        return [tag for tag, sim, place in result if not page_ids or tag.page_id in page_ids]
    return [(tag, sim, place) for tag, sim, place in result if not page_ids or tag.page_id in page_ids]


def get_similar_sentences(text, page_ids=None, **kwargs):
    if page_ids is None:
        page_ids = []
    return _get_similar_sentences(text, page_ids, **kwargs)


def filter_articles(tags):
    titles = [tag.title for tag in tags]

    def _filter_articles(*args, **kwargs):
        return kwargs["title"] in titles

    return _filter_articles


# def get_paragraphs_from_tags(wiki_path, tags: List[Tag], filter_namespaces=("0",)):
#     sentence_nums = defaultdict(list)
#     sentences = defaultdict(list)
#     for tag in tags:
#         sentence_nums[tag.page_id].append(tag.paragraph_num)
#     for title, text, page_id in extract_articles(bz2.BZ2File(wiki_path), filter_namespaces, filter_articles(tags)):
#         if text:
#             sentences[int(page_id)] = split_article_into_sentences(text)
#
#     return [filter_polish_wiki(sentences[tag.page_id][tag.paragraph_num]) for tag in tags]


if __name__ == "__main__":
    # tags, sims = zip(*get_similar_sentences("Some text.", only_tags=False))
    #
    # for tag in tags:
    #     print(tag)
    pass
