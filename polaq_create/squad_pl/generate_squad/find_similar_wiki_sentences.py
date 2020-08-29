import json
from collections import defaultdict

from squad_pl import logger, DATA_DIR
from squad_pl.doc2vec.preprocess import (
    TAGGED_WIKI_SENTENCE_DEPTH_WITH_LEMMAS_FILE,
    TaggedWikiSentenceWithLemmas,
    Tag,
)
from squad_pl.doc2vec.train import (
    LEMMA_MODEL_DBOW_300_DEPTH_FILE,
    # LEMMA_MODEL_DBOW_300_FULL_ARTICLE_FILE,
    EPOCHS,
    EpochSaver,
    LEMMA_MODEL_DBOW_300_DEPTH_SENTENCE_FILE,  # import required to load Doc2Vec models without errors
)
from squad_pl.doc2vec.similarity import get_similar_sentences, load_model
from squad_pl.doc2vec.utils import polish_nlp, remove_polish_stopwords
from squad_pl.preprocessing.preprocess import SQUAD_EXTENDED_PATH, find_similar_pages
from squad_pl.translate.translate_answers import load_translated_answers, get_answers_with_ner
from squad_pl.translate.translate_questions import load_translated_questions
from squad_pl.translate.utils import is_sublist


def is_valid_question(question: str):
    return len(question) > 5 and question.count(" ") > 1


def transform_question_into_affirmative_sentence(question, answer):
    question = question[:-1] if question[-1] in {"?", "."} else question
    return f"{answer.lower()} {question.lower()}"


def get_translated_data():
    translated_questions = load_translated_questions()
    translated_answers = load_translated_answers()
    answers_ner = get_answers_with_ner()

    for question_id, answers in translated_answers.items():
        if question_id in translated_questions and is_valid_question(translated_questions[question_id]):
            # there are some answers that don't have associated question
            # these are answers from file with CoreNLP annotations
            # apparently they finally weren't added to SQuAD
            translated_question = translated_questions[question_id]
            for english_answer, tr_answers in answers.items():
                # tr_answers is a list of possible polish translations of an english answer
                if tr_answers:
                    answer_ner = answers_ner[question_id][english_answer]
                    yield translated_question, english_answer, tr_answers, answer_ner, question_id


# paths for joined sentence 1 and 2 results
SIMILAR_WIKI_SENTENCES_JOINED_PATH = str(DATA_DIR / "squad/pl/similar_wiki_sentences_depth_{}.json")
FILTERED_SIMILAR_WIKI_SENTENCES_JOINED_PATH = str(DATA_DIR / "squad/pl/filtered_similar_wiki_sentences_depth_{}.json")
NLP_PROCESSED_SIMILAR_WIKI_SENTENCES_JOINED_PATH = str(
    DATA_DIR / "squad/pl/nlp_filtered_similar_wiki_sentences_depth_{}.json"
)

# paths for separate sentence 1 and 2 results
SIMILAR_WIKI_SENTENCES_SEPARATE_PATH = str(DATA_DIR / "squad/pl/similar_wiki_sentences_depth_{}_sentence_{}.json")
FILTERED_SIMILAR_WIKI_SENTENCES_SEPARATE_PATH = str(
    DATA_DIR / "squad/pl/filtered_similar_wiki_sentences_depth_{}_sentence_{}.json"
)
NLP_PROCESSED_SIMILAR_WIKI_SENTENCES_SEPARATE_PATH = str(
    DATA_DIR / "squad/pl/nlp_filtered_similar_wiki_sentences_depth_{}_sentence_{}.json"
)


def find_similar_sentences_in_wikipedia(depth, join=False, sentence_num=1):
    """
    For every (question_id, translated_question, translated_answer) triple use
    gensim Doc2Vec model to find sentences that are the most similar to
    the joined answer and question sentence (question is transformed into affirmative sentence using answer).

    Used gensim model is determined on a `depth` parameters.
    The result is saved in a json file.
    """
    tagged_wiki_file = TAGGED_WIKI_SENTENCE_DEPTH_WITH_LEMMAS_FILE.format(depth)

    if join:
        model = LEMMA_MODEL_DBOW_300_DEPTH_FILE.format(depth)
    else:
        model = LEMMA_MODEL_DBOW_300_DEPTH_SENTENCE_FILE.format(depth, sentence_num)

    model = str(model) + f"_epoch{EPOCHS}"
    model = load_model(model)

    if join:
        similar_wiki_sentences_path = SIMILAR_WIKI_SENTENCES_JOINED_PATH.format(depth)
    else:
        similar_wiki_sentences_path = SIMILAR_WIKI_SENTENCES_SEPARATE_PATH.format(depth, sentence_num)

    topn = 50 if depth == 0 else 100

    # create dict (tag of a document (with `sentence_num` sentences) as a string -> raw text)
    tagged_wiki = {}
    for tagged_document, raw_text in TaggedWikiSentenceWithLemmas.load(
        tagged_wiki_file, join, sentence_num, with_raw_text=True
    ):
        tagged_wiki[tagged_document.tags[0]] = raw_text

    # create dict (SQuAD question ID -> ID of a corresponding polish article)
    question_id_page_id = {}
    with open(SQUAD_EXTENDED_PATH, "r") as squad_file:
        for article in json.load(squad_file)["data"]:
            pl_page_id = article["plwiki_page_id"]  # it is None for 23 (= 490 - 467) articles
            for paragraph in article["paragraphs"]:
                for question_answers in paragraph["qas"]:
                    question_id_page_id[question_answers["id"]] = pl_page_id

    linked_pages = find_similar_pages(depth, mapping=True)

    counter = 0
    result = defaultdict(list)
    for translated_question, english_answer, translated_answers, answer_ner, question_id in get_translated_data():
        if question_id_page_id[question_id] is None:
            # skip if question belongs to one of 23 articles that don't have associated polish article in wikipedia
            continue

        # get page IDs of all articles that we want to take into account
        # depending on `depth` parameter this is just one ID of corresponding polish article (depth=0)
        # or this ID and IDs of pages similar to that article (depth=1)
        page_ids = linked_pages[question_id_page_id[question_id]]

        new_answer_similar_sentences = {}
        new_answer_similar_sentences["question"] = translated_question
        new_answer_similar_sentences["answer_ner"] = answer_ner
        new_answer_similar_sentences["similar_sentences"] = []

        for translated_answer in translated_answers:
            affirmative_sentence = transform_question_into_affirmative_sentence(translated_question, translated_answer)
            similar_sentences = get_similar_sentences(
                affirmative_sentence, model=model, topn=topn, page_ids=page_ids, only_tags=False,
            )

            new_answer_similar_sentences["similar_sentences"].append(
                {
                    "answer": translated_answer,
                    "en_answer": english_answer,
                    "sentences": [
                        {"text": tagged_wiki[str(tag)], "tag": str(tag), "similarity": sim, "place": place}
                        for tag, sim, place in similar_sentences
                    ],
                }
            )

        result[question_id].append(new_answer_similar_sentences)

        counter += 1
        if counter % 1000 == 0:
            logger.info("Processed %s questions", counter)

    logger.info("Finished")
    with open(similar_wiki_sentences_path, "w") as similar_wiki_sentences_file:
        json.dump(result, similar_wiki_sentences_file, indent=4)


SIMILAR_WIKI_ARTICLES_PATH = DATA_DIR / "squad/pl/similar_wiki_articles.json"


# def find_similar_articles_in_wikipedia(topn=100):
#     """
#     For every wikipedia article from depth 0 find similar articles from a depth 1 wikipedia subset.
#
#     This function works similarly to `find_similar_sentences_in_wikipedia` but uses different
#     model trained on a full wikipedia articles from a depth 1 wikipedia subset.
#     The result is saved in a json file.
#     """
#     tagged_wiki_file = TAGGED_WIKI_SENTENCE_DEPTH_WITH_LEMMAS_FILE.format(1)
#     model = LEMMA_MODEL_DBOW_300_FULL_ARTICLE_FILE
#     model = str(model) + f"_epoch{EPOCHS}"
#     model = load_model(model)
#
#     tagged_depth_0_articles = {}  # page_id --> list of all tokens (lemmatized) from the article
#     for tagged_document in TaggedWikiSentenceWithLemmas.load_from_file(
#         tagged_wiki_file, full_articles=True, with_raw_text=False
#     ):
#         tagged_depth_0_articles[Tag.from_str(tagged_document.tags[0]).page_id] = tagged_document.words
#
#     linked_pages = find_similar_pages(depth=1, mapping=True)
#
#     result = {}
#     for page_id, linked_articles in linked_pages.items():
#         similar_articles = get_similar_sentences(
#             tagged_depth_0_articles[page_id],
#             model=model,
#             topn=topn,
#             page_ids=linked_articles,
#             only_tags=False,
#             normalized_already=True,
#         )
#         result[page_id] = [
#             (str(tag), sim, place)
#             for tag, sim, place in similar_articles
#         ]
#
#     with open(SIMILAR_WIKI_ARTICLES_PATH, "w") as similar_wiki_articles_file:
#         json.dump(result, similar_wiki_articles_file, indent=4)


def filter_wiki_similar_sentences(depth, join=False, sentence_num=1, lemma_eq=True):
    """
    Filter the result of `find_similar_sentences_in_wikipedia`.

    Leave only those similar sentences that contain a corresponding answer.
    If lemma lemma_eq is True then lemma equality is taken into consideration (not simple tokens equality)
    """
    logger.info("Starting filtering")
    logger.info(f"depth={depth}, join={join}, sentence_num={sentence_num}, lemma_eq={lemma_eq}")

    if join:
        similar_wiki_sentences_path = SIMILAR_WIKI_SENTENCES_JOINED_PATH.format(depth)
    else:
        similar_wiki_sentences_path = SIMILAR_WIKI_SENTENCES_SEPARATE_PATH.format(depth, sentence_num)

    if join:
        filtered_similar_wiki_sentences_path = FILTERED_SIMILAR_WIKI_SENTENCES_JOINED_PATH.format(depth)
    else:
        filtered_similar_wiki_sentences_path = FILTERED_SIMILAR_WIKI_SENTENCES_SEPARATE_PATH.format(depth, sentence_num)

    counter = 0
    result = defaultdict(list)
    with open(similar_wiki_sentences_path, "r") as similar_wiki_sentences_file:
        for question_id, question_answers_data in json.load(similar_wiki_sentences_file).items():
            for question_answer_data in question_answers_data:
                question = question_answer_data["question"]
                answer_ner = question_answer_data["answer_ner"]

                question_answer_result = {"question": question, "answer_ner": answer_ner, "similar_sentences": []}

                for similar_sentences in question_answer_data["similar_sentences"]:
                    answer = similar_sentences["answer"]

                    # there are some invalid answers like "(25,700 km", "250,000 feet ("
                    if answer[0] in {"(", ")", "*"}:
                        answer = answer[1:].lstrip()
                    if answer[-1] in {"(", ")"}:
                        answer = answer[:-1].rstrip()

                    if not similar_sentences["sentences"]:
                        continue

                    processed_answer = polish_nlp(answer)
                    processed_sentences = [polish_nlp(sentence["text"]) for sentence in similar_sentences["sentences"]]

                    if lemma_eq:
                        lemmatized_answer = [token.lemma_.lower() for token in processed_answer]
                        # find sentences that contain lemmatized answer
                        sentences_with_answer = [
                            sentence
                            for i, sentence in enumerate(similar_sentences["sentences"])
                            if is_sublist(lemmatized_answer, [token.lemma_.lower() for token in processed_sentences[i]])
                        ]

                    else:
                        # just simple token equality check

                        # first, the most basic filtering
                        sentences_with_answer = [
                            sentence
                            for sentence in similar_sentences["sentences"]
                            if answer.lower() in sentence["text"].lower()
                        ]

                        tokenized_answer = [token.text.lower() for token in processed_answer]
                        # second filtering
                        # find sentences that contain tokenized answer
                        sentences_with_answer = [
                            sentence
                            for i, sentence in enumerate(sentences_with_answer)
                            if is_sublist(tokenized_answer, [token.text.lower() for token in processed_sentences[i]])
                        ]

                    if sentences_with_answer:
                        question_answer_result["similar_sentences"].append(
                            {"answer": answer, "sentences": sentences_with_answer}
                        )

                if question_answer_result["similar_sentences"]:
                    result[question_id].append(question_answer_result)

            counter += 1
            if counter % 1000 == 0:
                logger.info("Processed %s questions", counter)

    with open(filtered_similar_wiki_sentences_path, "w") as filtered_similar_wiki_sentences_file:
        json.dump(result, filtered_similar_wiki_sentences_file, indent=4)


def nlp_process_wiki_similar_sentences(depth, join=False, sentence_num=1):
    """Do the nlp processing with spaCy on filtered similar sentences and save the result."""
    if join:
        filtered_similar_wiki_sentences_path = FILTERED_SIMILAR_WIKI_SENTENCES_JOINED_PATH.format(depth)
    else:
        filtered_similar_wiki_sentences_path = FILTERED_SIMILAR_WIKI_SENTENCES_SEPARATE_PATH.format(depth, sentence_num)

    if join:
        nlp_similar_wiki_sentences_path = NLP_PROCESSED_SIMILAR_WIKI_SENTENCES_JOINED_PATH.format(depth)
    else:
        nlp_similar_wiki_sentences_path = NLP_PROCESSED_SIMILAR_WIKI_SENTENCES_SEPARATE_PATH.format(depth, sentence_num)

    with open(filtered_similar_wiki_sentences_path, "r") as filtered_similar_wiki_sentences_file:
        result = json.load(filtered_similar_wiki_sentences_file)

    counter = 0
    for question_id, question_answers_data in result.items():
        for i, question_answer_data in enumerate(question_answers_data):
            question = question_answer_data["question"]
            processed_question = polish_nlp(question)
            result[question_id][i]["processed_question"] = {
                "tokens": [token.text for token in processed_question],
                "lemmas": [token.lemma_ for token in processed_question],
                "tags": [token.tag_ for token in processed_question],
                "poss": [token.pos_ for token in processed_question],
                "nes": {entity.text: entity.label_ for entity in processed_question.ents},  # named entities
            }

            for j, similar_sentences in enumerate(question_answer_data["similar_sentences"]):
                answer = similar_sentences["answer"]
                processed_answer = polish_nlp(answer)
                result[question_id][i]["similar_sentences"][j]["processed_answer"] = {
                    "tokens": [token.text for token in processed_answer],
                    "lemmas": [token.lemma_ for token in processed_answer],
                    "tags": [token.tag_ for token in processed_answer],
                    "poss": [token.pos_ for token in processed_answer],
                    "nes": {entity.text: entity.label_ for entity in processed_answer.ents},
                }

                sentences = similar_sentences["sentences"]
                processed_sentences = [polish_nlp(sentence["text"]) for sentence in sentences]
                result[question_id][i]["similar_sentences"][j]["processed_sentences"] = [
                    {
                        "tokens": [token.text for token in processed_sentence],
                        "lemmas": [token.lemma_ for token in processed_sentence],
                        "tags": [token.tag_ for token in processed_sentence],
                        "poss": [token.pos_ for token in processed_sentence],
                        "nes": {entity.text: entity.label_ for entity in processed_sentence.ents},
                    }
                    for processed_sentence in processed_sentences
                ]

        counter += 1
        if counter % 1000 == 0:
            logger.info("Processed %s questions", counter)

    with open(nlp_similar_wiki_sentences_path, "w") as nlp_similar_wiki_sentences_file:
        json.dump(result, nlp_similar_wiki_sentences_file, indent=2)


def print_filtered(depth, join=False, sentence_num=1, max_similar_sentences=5):
    """Print filtered similar sentences in a nice format."""
    if join:
        filtered_similar_wiki_sentences_path = FILTERED_SIMILAR_WIKI_SENTENCES_JOINED_PATH.format(depth)
    else:
        filtered_similar_wiki_sentences_path = FILTERED_SIMILAR_WIKI_SENTENCES_SEPARATE_PATH.format(depth, sentence_num)

    with open(filtered_similar_wiki_sentences_path, "r") as filtered_similar_wiki_sentences_file:
        for question_id, question_answers_data in json.load(filtered_similar_wiki_sentences_file).items():
            for question_answer_data in question_answers_data:
                question = question_answer_data["question"]
                answer_ner = question_answer_data["answer_ner"]

                for similar_sentences in question_answer_data["similar_sentences"]:
                    answer = similar_sentences["answer"]
                    sentences = similar_sentences["sentences"][:max_similar_sentences]

                    print(64 * "=")
                    print()
                    print(f"Question: {question}")
                    print(f"Answer: {answer}")
                    print(f"Similar sentences:")
                    for i, sentence in enumerate(sentences):
                        print(f"{i + 1}. {sentence['text']}   (tag: {sentence['tag']!r})")
                    print()
                    print(64 * "=")


if __name__ == "__main__":
    # import time

    # find_similar_sentences_in_wikipedia(depth=1, join=False, sentence_num=2)
    # time.sleep(5*60)
    # find_similar_sentences_in_wikipedia(depth=1, join=False, sentence_num=1)
    # time.sleep(5*60)
    # find_similar_sentences_in_wikipedia(depth=0, join=False, sentence_num=2)
    # time.sleep(5*60)
    # find_similar_sentences_in_wikipedia(depth=0, join=False, sentence_num=1)

    # filter_wiki_similar_sentences(depth=1, join=False, sentence_num=2, lemma_eq=True)
    # time.sleep(60)
    # filter_wiki_similar_sentences(depth=1, join=False, sentence_num=1, lemma_eq=True)
    # time.sleep(60)
    # filter_wiki_similar_sentences(depth=0, join=False, sentence_num=2, lemma_eq=True)
    # time.sleep(60)
    # filter_wiki_similar_sentences(depth=0, join=False, sentence_num=1, lemma_eq=True)

    # nlp_process_wiki_similar_sentences(depth=1, join=False, sentence_num=2)
    # nlp_process_wiki_similar_sentences(depth=1, join=False, sentence_num=1)
    # nlp_process_wiki_similar_sentences(depth=0, join=False, sentence_num=2)
    # nlp_process_wiki_similar_sentences(depth=0, join=False, sentence_num=1)

    # nlp_process_wiki_similar_sentences(depth=1, join=True)
    # nlp_process_wiki_similar_sentences(depth=0, join=True)
    pass
