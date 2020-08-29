"""Train Doc2Vec models."""

import multiprocessing
import os

# from gensim.models import Doc2Vec
from gensim.models import Doc2Vec
from gensim.models.callbacks import CallbackAny2Vec

from squad_pl import DATA_DIR, logger
from squad_pl.doc2vec.preprocess import (
    TaggedWiki,
    TaggedWikiWithLemmas,
    TaggedWikiSentenceWithLemmas,
    TAGGED_WIKI_SENTENCE_DEPTH_WITH_LEMMAS_FILE,
)


MODEL_PATH = DATA_DIR / "doc2vec/models"

# model path for joined 1 and 2 sentences
LEMMA_MODEL_DBOW_300_DEPTH_FILE = str(MODEL_PATH / "lemma_model_dbow300_depth_{}")

# model path for separate sentence 1 and 2
LEMMA_MODEL_DBOW_300_DEPTH_SENTENCE_FILE = str(MODEL_PATH / "lemma_model_dbow300_depth_{}_sentence_{}")

# LEMMA_MODEL_DBOW_300_FULL_ARTICLE_FILE = MODEL_PATH / "lemma_model_dbow300_full_article"


PV_DBOW = 0  # distributed bag of words model outperforms distributed memory - https://arxiv.org/pdf/1507.07998.pdf
PV_DM = 1

# Useful:
# https://stackoverflow.com/questions/50390455/gensim-doc2vec-memoryerror-when-training-on-english-wikipedia
# https://stackoverflow.com/questions/56323377/which-method-dm-or-dbow-works-well-for-document-similarity-using-doc2vec
# http://piyushbhardwaj.github.io/documents/w2v_p2vupdates.pdf
# https://stackoverflow.com/questions/53368915/genisim-doc2vec-how-is-short-doc-processed
# https://www.thinkinfi.com/2019/10/simple-explanation-of-doc2vec.html - pv-dbow is good for short documents


EPOCHS = 20

doc2vec_kwargs = {
    "dm": PV_DBOW,
    "dbow_words": 1,  # train word vectors too (only if PV_DBOW is used)
    # according to https://arxiv.org/pdf/1507.07998.pdf PV_DBOW outperforms PV_DM
    # and it gets better results with simultaneous word vectors training
    "window": 8,  # for PV_DBOW used only for word vectors training (when dbow_words=1, gensim uses skip gram)
    # (15 was used in https://arxiv.org/pdf/1607.05368.pdf (for whole paragraphs))
    # mean length of all my processed texts is ~22,5 for depth 0 and ~20 for depth 1
    #
    # set to 25 because in other papers they used window of mean sentence size when they
    # were computing sentence embeddings
    "vector_size": 300,  # normally the size is between 100-300
    "alpha": 0.025,  # default gensim value
    "min_alpha": 0.0001,  # default gensim value
    "hs": 0,  # default gensim value - use negative sampling,
    "negative": 10,  # 10 for depth 0 and 5 (default gensim value) for depth 1
    "ns_exponent": 0.75,  # default gensim value - 3/4 like in a doc2vec original paper
    "sample": 0.001,  # subsampling/downsampling - default gensim value
    #
    # actually common words are already removed by preprocessing (removing stopwords)
    # so bigger sample value 0.001 (default for gensim) should be enough
    #
    # 0.00005 for depth 0, removes 1450 most common words (from 59779 unique) - leaves 71.5% of all words (1986918)
    # (0.001 removes 9, 0.0001 removes 667, 0.00001 removes 5626 - leaves only 48% of all words!)
    # and 0.00001 for depth 1, removes ...
    # 0.00001 is a value from original doc2vec paper and
    # also used here https://arxiv.org/pdf/1607.05368.pdf)
    "min_count": 5,  # default gensim value - what value is the best here?
    # for depth 0 it's better to not ignore any word but for depth 1 min_count=5 might be good
    # https://stackoverflow.com/questions/47890052/improving-gensim-doc2vec-results
    "epochs": EPOCHS,  # usually 10-20 epochs (might be bigger for larger corpus)
    "workers": multiprocessing.cpu_count(),
}

###
### dm_concat, dm_tag_count are used only for PV_DM
###


class EpochSaver(CallbackAny2Vec):
    """Callback to save model after each epoch."""

    def __init__(self, model_path, start_epoch=1, epochs_to_leave=None):
        self.model_path = model_path
        self.epoch = start_epoch
        self.epochs_to_leave = [] if epochs_to_leave is None else epochs_to_leave

    def on_epoch_begin(self, model):
        logger.info(f"Epoch #{self.epoch} start")
        logger.info(f"Min alpha yet reached: {model.min_alpha_yet_reached}")

    def on_epoch_end(self, model):
        output_path = f"{self.model_path}_epoch{self.epoch}"
        model.save(output_path)
        logger.info(f"Epoch {self.epoch} model saved")
        self.remove_old_model(epoch_to_remove=self.epoch - 1)
        logger.info(f"Min alpha yet reached: {model.min_alpha_yet_reached}")
        logger.info(f"Epoch #{self.epoch} end")
        self.epoch += 1

    def remove_old_model(self, epoch_to_remove: int):
        """Removes old model to free up memory space."""
        if epoch_to_remove not in self.epochs_to_leave:
            old_model_path = f"{self.model_path}_epoch{epoch_to_remove}"
            for suffix in ["", ".docvecs.vectors_docs.npy", ".trainables.syn1neg.npy", ".wv.vectors.npy"]:
                path = old_model_path + suffix
                if os.path.exists(path):
                    os.remove(path)
                    logger.info(f"Epoch {epoch_to_remove} model file {path} removed")


# min_count: https://stackoverflow.com/questions/47890052/improving-gensim-doc2vec-results?rq=1
# least frequent words are not helpful


# def get_total_lemmas_count():
#     """Return total number of lemmas in the corpus."""
#     return len(get_lemma_frequencies()[0].keys())
#
#
# def get_total_words_count():
#     """Return total number of words in the corpus."""
#     return len(get_word_frequencies()[0].keys())


# def train_lemma_model_with_freq(model_to_load=LEMMA_MODEL_DBOW_100_FILE):
#     try:
#         model_lemma_dbow_100 = Doc2Vec.load(str(model_to_load))
#         total_lemmas_count = get_total_lemmas_count()
#     except FileNotFoundError:
#         logger.info("Model not found. Building a new one from lemma frequencies.")
#         lemma_freq, corpus_count = get_lemma_frequencies()
#         total_lemmas_count = len(lemma_freq.keys())
#         model_lemma_dbow_100 = Doc2Vec(**doc2vec_kwargs)
#         model_lemma_dbow_100.build_vocab_from_freq(lemma_freq, corpus_count=corpus_count)
#         model_lemma_dbow_100.save(str(LEMMA_MODEL_DBOW_100_FILE))
#
#     epoch_saver = EpochSaver(LEMMA_MODEL_DBOW_100_FILE)
#     model_lemma_dbow_100.train(
#         documents=TaggedWikiParagraphWithLemmas(),
#         callbacks=[epoch_saver],
#         total_examples=model_lemma_dbow_100.corpus_count,
#         total_words=total_lemmas_count,
#         epochs=model_lemma_dbow_100.epochs
#     )
#
#
# def train_word_model_with_freq(model_to_load=WORD_MODEL_DBOW_100_FILE):
#     try:
#         model_word_dbow_100 = Doc2Vec.load(str(model_to_load))
#         total_words_count = get_total_words_count()
#     except FileNotFoundError:
#         logger.info("Model not found. Building a new one from word frequencies.")
#         word_freq, corpus_count = get_word_frequencies()
#         total_words_count = len(word_freq.keys())
#         model_word_dbow_100 = Doc2Vec(**doc2vec_kwargs)
#         model_word_dbow_100.build_vocab_from_freq(word_freq, corpus_count=corpus_count)
#         model_word_dbow_100.save(str(WORD_MODEL_DBOW_100_FILE))
#
#     epoch_saver = EpochSaver(WORD_MODEL_DBOW_100_FILE)
#     model_word_dbow_100.train(
#         documents=TaggedWikiParagraph(),
#         callbacks=[epoch_saver],
#         total_examples=model_word_dbow_100.corpus_count,
#         total_words=total_words_count,
#         epochs=model_word_dbow_100.epochs
#     )


def train_lemma_model(model_path):
    epoch_saver = EpochSaver(model_path)
    model = Doc2Vec(documents=TaggedWikiWithLemmas(), **doc2vec_kwargs, callbacks=[epoch_saver])
    return model


def train_word_model(model_path):
    epoch_saver = EpochSaver(model_path)
    model = Doc2Vec(documents=TaggedWiki(), **doc2vec_kwargs, callbacks=[epoch_saver])
    return model


# def continue_lemma_training(model_to_load, start_epoch, epochs):
#     model = Doc2Vec.load(str(model_to_load))
#     epoch_saver = EpochSaver(LEMMA_MODEL_DBOW_300_FILE, start_epoch)
#     # model.min_alpha_yet_reached is the learning rate to be used in the next training epoch
#     # (according to a gensim code)
#     model.train(
#         documents=TaggedWikiWithLemmas(),
#         start_alpha=model.min_alpha_yet_reached,
#         epochs=epochs,
#         callbacks=[epoch_saver],
#         total_examples=model.corpus_count,
#     )


def train_model(depth=0, join=True, sentence_num=1):
    """Train doc2vec model."""
    if join:
        model_file = LEMMA_MODEL_DBOW_300_DEPTH_FILE.format(depth)
    else:
        model_file = LEMMA_MODEL_DBOW_300_DEPTH_SENTENCE_FILE.format(depth, sentence_num)

    epoch_saver = EpochSaver(model_file)
    sentences = list(
        TaggedWikiSentenceWithLemmas.load(
            TAGGED_WIKI_SENTENCE_DEPTH_WITH_LEMMAS_FILE.format(depth), join, sentence_num, with_raw_text=False
        )
    )
    Doc2Vec(documents=sentences, **doc2vec_kwargs, callbacks=[epoch_saver])


if __name__ == "__main__":
    train_model(depth=1, sentence_num=2, join=False)
    pass
