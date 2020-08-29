"""Check if trained models works properly."""

from collections import defaultdict

from gensim.models import Doc2Vec

from squad_pl import DATA_DIR, logger
from squad_pl.doc2vec.preprocess import TaggedWikiSentenceWithLemmas, TAGGED_WIKI_SENTENCE_DEPTH_0_WITH_LEMMAS_FILE
from squad_pl.doc2vec.train import EpochSaver  # import required to load Doc2Vec models without errors


MODEL_PATH = DATA_DIR / "doc2vec/models"


"""
From https://radimrehurek.com/gensim/auto_examples/tutorials/run_doc2vec_lee.html
Checking the inferred-vector against a training-vector is a sort of ‘sanity check’ as to whether the model is behaving
in a usefully consistent manner, though not a real ‘accuracy’ value.
"""


def simple_model_check(model, corpus):
    text_num = 0
    counter = defaultdict(int)
    logger.info("Starting model checking")
    for text, [tag] in corpus:
        text_num += 1
        inferred_vector = model.infer_vector(text)
        sims = model.docvecs.most_similar([inferred_vector], topn=20)
        rank = -1
        try:
            rank = [tag for tag, sim in sims].index(tag)
        except ValueError:
            pass
        counter[rank] += 1
        if text_num % 10000 == 0:
            logger.info("%d paragraphs processed", text_num)
            logger.info({rank: (count / text_num) for rank, count in counter.items()})

    logger.info("Finished")
    logger.info({rank: (count / text_num) for rank, count in counter.items()})


if __name__ == "__main__":
    model = Doc2Vec.load(str(MODEL_PATH / "lemma_model_dbow300_epoch20"))
    model.docvecs.init_sims(replace=True)
    simple_model_check(
        model,
        TaggedWikiSentenceWithLemmas.load_from_file(TAGGED_WIKI_SENTENCE_DEPTH_0_WITH_LEMMAS_FILE, with_raw_text=False),
    )
