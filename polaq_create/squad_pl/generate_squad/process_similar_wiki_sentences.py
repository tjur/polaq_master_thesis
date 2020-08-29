import json
import os
import sys
from typing import List

import readchar

from squad_pl import DATA_DIR, logger, SQUAD_PATH_1_1
from squad_pl.doc2vec.preprocess import Tag
from squad_pl.generate_squad.find_similar_wiki_sentences import (
    FILTERED_SIMILAR_WIKI_SENTENCES_JOINED_PATH,
    FILTERED_SIMILAR_WIKI_SENTENCES_SEPARATE_PATH,
    NLP_PROCESSED_SIMILAR_WIKI_SENTENCES_JOINED_PATH,
    NLP_PROCESSED_SIMILAR_WIKI_SENTENCES_SEPARATE_PATH,
    SIMILAR_WIKI_ARTICLES_PATH,
)

# correct
from squad_pl.preprocessing.preprocess import SQUAD_EXTENDED_PATH, find_similar_pages
from squad_pl.translate.utils import is_sublist

FILTERED_CORRECT_SIMILAR_WIKI_SENTENCES_PATH = str(
    DATA_DIR / "squad/pl/manual/filtered_correct_similar_wiki_sentences_depth_{}_sentence_{}.json"
)
# incorrect
FILTERED_INCORRECT_SIMILAR_WIKI_SENTENCES_PATH = str(
    DATA_DIR / "squad/pl/manual/filtered_incorrect_similar_wiki_sentences_depth_{}_sentence_{}.json"
)


def manual_filter_wiki_similar_sentences_part_1():
    """
    Manually process wikipedia similar sentences from depth 0 matchings
    (all question/answers that have whether only sentence 1 or both sentence 1 and sentence 2 matches).

    Depth 1 sentences won't be manually processed here because most of the results are poor now.
    """
    correct_path_d0_s1 = FILTERED_CORRECT_SIMILAR_WIKI_SENTENCES_PATH.format(0, 1)
    incorrect_path_d0_s1 = FILTERED_INCORRECT_SIMILAR_WIKI_SENTENCES_PATH.format(0, 1)
    correct_path_d0_s2 = FILTERED_CORRECT_SIMILAR_WIKI_SENTENCES_PATH.format(0, 2)
    incorrect_path_d0_s2 = FILTERED_INCORRECT_SIMILAR_WIKI_SENTENCES_PATH.format(0, 2)

    with open(FILTERED_SIMILAR_WIKI_SENTENCES_SEPARATE_PATH.format(0, 1), "r") as filtered_similar_wiki_sentences_file:
        data_d0_s1 = json.load(filtered_similar_wiki_sentences_file)

    with open(FILTERED_SIMILAR_WIKI_SENTENCES_SEPARATE_PATH.format(0, 2), "r") as filtered_similar_wiki_sentences_file:
        data_d0_s2 = json.load(filtered_similar_wiki_sentences_file)

    already_processed = set()

    def init_file(filename, already_processed):
        try:
            with open(filename, "r") as f:
                for question_answer in json.load(f):
                    already_processed.add((question_answer["question_id"], question_answer["answer"]))
        except FileNotFoundError:
            with open(filename, "w") as f:
                json.dump([], f)

    init_file(correct_path_d0_s1, already_processed)
    init_file(incorrect_path_d0_s1, already_processed)
    init_file(correct_path_d0_s2, already_processed)
    init_file(incorrect_path_d0_s2, already_processed)

    def save_question_answer(filename, question_id, question, answer, answer_ner, sentence):
        with open(filename, "r") as f:
            question_answers = json.load(f)
        question_answers.append(
            {
                "question_id": question_id,
                "question": question,
                "answer": answer,
                "sentence": sentence["text"],
                "tag": sentence["tag"],
                "answer_ner": answer_ner,
                "place": sentence["place"],
            }
        )
        with open(filename, "w") as f:
            json.dump(question_answers, f, indent=4)

    class Color:
        RED = "\033[91m"
        BOLD = "\033[1m"
        END = "\033[0m"

    # max number of similar sentences that will be displayed for a question and answer
    sentences_to_display = 5

    question_ids = {}
    with open(SQUAD_PATH_1_1, "r") as squad_file:
        for article in json.load(squad_file)["data"]:
            for paragraph in article["paragraphs"]:
                for qa in paragraph["qas"]:
                    question_ids[qa["id"]] = qa["question"]

    try:
        for question_id, question_answers_data in data_d0_s1.items():
            for question_answer_data in question_answers_data:
                question = question_answer_data["question"]
                answer_ner = question_answer_data["answer_ner"]

                for similar_sentences in question_answer_data["similar_sentences"]:
                    answer = similar_sentences["answer"]
                    sentences = similar_sentences["sentences"]

                    if (question_id, answer) not in already_processed and sentences:
                        print(f"Question: {Color.BOLD}{question}{Color.END}   (sentence 1)")
                        print(f"({question_ids[question_id]})")
                        print()
                        print(f"Answer: {Color.BOLD}{answer}{Color.END}")
                        print()
                        for i, sentence in enumerate(sentences[:sentences_to_display]):
                            print(f"{i+1}. {Color.BOLD}{sentence['text']}{Color.END}   {sentence['tag']!r}")
                            print()

                        while True:
                            key = readchar.readkey()

                            if key in map(lambda i: str(i + 1), range(len(sentences[:sentences_to_display]))):
                                # correct
                                sentence = sentences[int(key) - 1]
                                save_question_answer(
                                    correct_path_d0_s1, question_id, question, answer, answer_ner, sentence
                                )
                                break

                            if key == readchar.key.ENTER:
                                # incorrect
                                # display sentence 2 possible sentences if exist

                                sim_sentences = [
                                    sim_sentences
                                    for qa in data_d0_s2.get(question_id, [])
                                    for sim_sentences in qa["similar_sentences"]
                                    if qa["question"] == question and sim_sentences["answer"] == answer
                                ]

                                if not sim_sentences or not sim_sentences[0]["sentences"]:
                                    # no corresponding question_answer or it has empty result
                                    sentence = sentences[0]
                                    save_question_answer(
                                        incorrect_path_d0_s1, question_id, question, answer, answer_ner, sentence
                                    )
                                    break

                                # clear screen
                                os.system("clear")

                                sentences = sim_sentences[0]["sentences"]

                                print(f"Question: {Color.BOLD}{question}{Color.END}   (sentence 2)")
                                print(f"({question_ids[question_id]})")
                                print()
                                print(f"Answer: {Color.BOLD}{answer}{Color.END}")
                                print()
                                for i, sentence in enumerate(sentences[:sentences_to_display]):
                                    print(f"{i + 1}. {Color.BOLD}{sentence['text']}{Color.END}   {sentence['tag']!r}")
                                    print()

                                while True:
                                    key = readchar.readkey()

                                    if key in map(lambda i: str(i + 1), range(len(sentences[:sentences_to_display]))):
                                        # correct
                                        sentence = sentences[int(key) - 1]
                                        save_question_answer(
                                            correct_path_d0_s2, question_id, question, answer, answer_ner, sentence
                                        )
                                        break

                                    if key == readchar.key.ENTER:
                                        # incorrect
                                        sentence = sentences[0]
                                        save_question_answer(
                                            incorrect_path_d0_s2, question_id, question, answer, answer_ner, sentence
                                        )
                                        break

                                    if key == readchar.key.CTRL_F:
                                        logger.info("Finished")
                                        sys.exit()

                                break

                            if key == readchar.key.CTRL_F:
                                logger.info("Finished")
                                sys.exit()

                        # clear screen
                        os.system("clear")

    except:
        pass

    logger.info("Finished")


def manual_filter_wiki_similar_sentences_part_2():
    """
    Manually process wikipedia similar sentences from depth 0 sentence 2 matchings
    that weren't checked in `manual_filter_wiki_similar_sentences_part_1`
    (those are question/answers that have only sentence 2 match).
    """
    correct_path_d0_s1 = FILTERED_CORRECT_SIMILAR_WIKI_SENTENCES_PATH.format(0, 1)
    incorrect_path_d0_s1 = FILTERED_INCORRECT_SIMILAR_WIKI_SENTENCES_PATH.format(0, 1)
    correct_path_d0_s2 = FILTERED_CORRECT_SIMILAR_WIKI_SENTENCES_PATH.format(0, 2)
    incorrect_path_d0_s2 = FILTERED_INCORRECT_SIMILAR_WIKI_SENTENCES_PATH.format(0, 2)

    with open(FILTERED_SIMILAR_WIKI_SENTENCES_SEPARATE_PATH.format(0, 2), "r") as filtered_similar_wiki_sentences_file:
        data_d0_s2 = json.load(filtered_similar_wiki_sentences_file)

    already_processed = set()

    def init_file(filename, already_processed):
        try:
            with open(filename, "r") as f:
                for question_answer in json.load(f):
                    already_processed.add((question_answer["question_id"], question_answer["answer"]))
        except FileNotFoundError:
            with open(filename, "w") as f:
                json.dump([], f)

    init_file(correct_path_d0_s1, already_processed)
    init_file(incorrect_path_d0_s1, already_processed)
    init_file(correct_path_d0_s2, already_processed)
    init_file(incorrect_path_d0_s2, already_processed)

    def save_question_answer(filename, question_id, question, answer, answer_ner, sentence):
        with open(filename, "r") as f:
            question_answers = json.load(f)
        question_answers.append(
            {
                "question_id": question_id,
                "question": question,
                "answer": answer,
                "sentence": sentence["text"],
                "tag": sentence["tag"],
                "answer_ner": answer_ner,
                "place": sentence["place"],
            }
        )
        with open(filename, "w") as f:
            json.dump(question_answers, f, indent=4)

    class Color:
        RED = "\033[91m"
        BOLD = "\033[1m"
        END = "\033[0m"

    # max number of similar sentences that will be displayed for a question and answer
    sentences_to_display = 5

    question_ids = {}
    with open(SQUAD_PATH_1_1, "r") as squad_file:
        for article in json.load(squad_file)["data"]:
            for paragraph in article["paragraphs"]:
                for qa in paragraph["qas"]:
                    question_ids[qa["id"]] = qa["question"]

    try:
        for question_id, question_answers_data in data_d0_s2.items():
            for question_answer_data in question_answers_data:
                question = question_answer_data["question"]
                answer_ner = question_answer_data["answer_ner"]

                for similar_sentences in question_answer_data["similar_sentences"]:
                    answer = similar_sentences["answer"]
                    sentences = similar_sentences["sentences"]

                    if (question_id, answer) not in already_processed and sentences:
                        print(f"Question: {Color.BOLD}{question}{Color.END}")
                        print(f"({question_ids[question_id]})")
                        print()
                        print(f"Answer: {Color.BOLD}{answer}{Color.END}")
                        print()
                        for i, sentence in enumerate(sentences[:sentences_to_display]):
                            print(f"{i+1}. {Color.BOLD}{sentence['text']}{Color.END}   {sentence['tag']!r}")
                            print()

                        while True:
                            key = readchar.readkey()

                            if key in map(lambda i: str(i + 1), range(len(sentences[:sentences_to_display]))):
                                # correct
                                sentence = sentences[int(key) - 1]
                                save_question_answer(
                                    correct_path_d0_s2, question_id, question, answer, answer_ner, sentence
                                )
                                break

                            if key == readchar.key.ENTER:
                                # incorrect
                                sentence = sentences[0]
                                save_question_answer(
                                    incorrect_path_d0_s2, question_id, question, answer, answer_ner, sentence
                                )
                                break

                            if key == readchar.key.CTRL_F:
                                sys.exit()

                        # clear screen
                        os.system("clear")

    except:
        pass

    logger.info("Finished")


# Some articles gives (almost always) bad matchings
RISKY_ARTICLES = {
    "Liczba pierwsza",
    "Program Apollo",
    "Buddyzm",
    "FC Barcelona",
    "Paryż",
    "Ból",
    "Warszawa",
    "Nowe Delhi",
}


def process_wiki_similar_sentences(depth, join=False) -> List[dict]:
    """
    Process wikipedia similar sentences.

    `join` argument tells whether the result should come from separate or joined training of sentence 1 and 2.
    """
    print(f"Depth: {depth}, Join: {join}")

    to_process = []
    if join:
        with open(
            NLP_PROCESSED_SIMILAR_WIKI_SENTENCES_JOINED_PATH.format(depth), "r"
        ) as nlp_similar_wiki_sentences_file:
            data = json.load(nlp_similar_wiki_sentences_file)
        to_process.append((data, depth, None))
    else:
        with open(
            NLP_PROCESSED_SIMILAR_WIKI_SENTENCES_SEPARATE_PATH.format(depth, 1), "r"
        ) as nlp_similar_wiki_sentences_file:
            data = json.load(nlp_similar_wiki_sentences_file)
        to_process.append((data, depth, 1))

        with open(
            NLP_PROCESSED_SIMILAR_WIKI_SENTENCES_SEPARATE_PATH.format(depth, 2), "r"
        ) as nlp_similar_wiki_sentences_file:
            data = json.load(nlp_similar_wiki_sentences_file)
        to_process.append((data, depth, 2))

    SIMILAR_PAGES_LIMIT = 30  # max number of the most similar articles (for the source article)
    similar_pages = find_similar_pages(depth=1, mapping=True)

    CHARS_WITHOUT_SPACES = {",", ".", ":", ";", "'", "-", "‐", "−", "–", "%", "’", "°", "(", ")", "/", "„", "”", "+"}

    question_id_to_page_id = {}
    with open(SQUAD_EXTENDED_PATH, "r") as squad_extended_file:
        for article in json.load(squad_extended_file)["data"]:
            for paragraph in article["paragraphs"]:
                for qas in paragraph["qas"]:
                    plwiki_page_id = article["plwiki_page_id"]
                    plwiki_page_id = int(plwiki_page_id) if plwiki_page_id is not None else plwiki_page_id
                    question_id_to_page_id[qas["id"]] = plwiki_page_id

    def process_data(data, depth, sentence_num):
        """Filter data and leave just one matching sentence for every (question_id, real answer) pair."""
        result = {}
        for question_id, question_answers_data in data.items():
            source_page_id = question_id_to_page_id[question_id]
            for i, question_answer_data in enumerate(question_answers_data):
                question = question_answer_data["question"]
                answer_ner = question_answer_data["answer_ner"]
                processed_question = question_answer_data["processed_question"]

                for j, similar_sentences in enumerate(question_answer_data["similar_sentences"]):
                    original_answer = similar_sentences["answer"]
                    sentences = similar_sentences["sentences"]
                    processed_answer = similar_sentences["processed_answer"]
                    processed_sentences = similar_sentences["processed_sentences"]

                    answer_tokens = processed_answer["tokens"]
                    # lemmatized tokens of answer
                    answer_lemmas = [l.lower() for l in processed_answer["lemmas"]]

                    sentences_tokens = [processed_sentence["tokens"] for processed_sentence in processed_sentences]

                    # lemmatized tokens of sentences
                    sentences_lemmas = [
                        [l.lower() for l in processed_sentence["lemmas"]] for processed_sentence in processed_sentences
                    ]

                    # find a real answer
                    for k, sentence in enumerate(sentences):
                        sentences[k]["tag_object"] = Tag.from_str(sentence["tag"])
                        pos = is_sublist(answer_lemmas, sentences_lemmas[k], pos=True)
                        assert pos >= 0
                        sentences[k]["real_answer"] = " ".join(sentences_tokens[k][pos : pos + len(answer_tokens)])
                        # JOINING TOKENS WITH SPACE HERE IS PROBABLY NOT A GOOD IDEA
                        # (TOKENS NOT ALWAYS ARE SEPARATED BY SPACE, - IS ALSO POSSIBLE)
                        # later we need to check that lemmatized real answer is equal to lemmatized original answer

                    # filter out common invalid matchings
                    # disallow sentences that are:
                    # a beginning of a list (ends with ":")
                    # an element of a list (starts with "*")
                    # something that start with #
                    # contains ".jpg" (some wikipedia markup leftover)
                    # most of them are incorrect matchings

                    sentences = [
                        sentence
                        for sentence in sentences
                        if sentence["text"].rstrip()[-1] != ":"
                        and sentence["text"].lstrip()[0] not in ("*", "#")
                        and "jpg" not in sentence["text"].lower()  # some preprocessing leftovers
                    ]

                    # leave only matchings from the most similar articles (only first similar_pages_limit)
                    # from articles from RISKY_ARTICLES require high similarity
                    sentences = [
                        sentence
                        for sentence in sentences
                        if sentence["tag_object"].page_id in similar_pages[source_page_id][:SIMILAR_PAGES_LIMIT]
                        and not ({sentence["tag_object"].title} & RISKY_ARTICLES and sentence["similarity"] < 0.8)
                    ]

                    # use matching sentence for question answer only if its similarity is bigger than
                    # for already found result (or matching sentence is a first
                    # found result for this question-answer)
                    if (
                        sentences
                        and len(sentences) <= 10  # too much similar sentences - too common answer
                        # and sentences[0]["place"] < 25
                        and sentences[0]["similarity"] >= 0.6  # less than 0.6 are usually incorrect
                    ):
                        sentence = sentences[0]
                        real_answer = sentence["real_answer"]

                        if real_answer.lower() != original_answer.lower():
                            # remove extra spaces added previously
                            for char in CHARS_WITHOUT_SPACES:
                                if f" {char}" in real_answer and f" {char}" not in original_answer:
                                    real_answer = real_answer.replace(f" {char}", f"{char}")
                                if f"{char} " in real_answer and f"{char} " not in original_answer:
                                    real_answer = real_answer.replace(f"{char} ", f"{char}")
                            # "808s"
                            real_answer = real_answer.replace(" s ", "s ")

                        # handling of some special/weird cases
                        real_answer = real_answer.replace("° C", "°C")
                        real_answer = real_answer.replace("/ t /", "/t/")
                        real_answer = real_answer.replace("AC / DC", "AC/DC")
                        real_answer = real_answer.replace("12 terabajtów / miesiąc", "12 terabajtów/miesiąc")
                        real_answer = real_answer.replace("George’a H. W. Busha", "George’a H.W. Busha")
                        real_answer = real_answer.replace("3: 4, 3: 5, 4: 7 i 2: 5", "3:4, 3:5, 4:7 i 2:5")
                        real_answer = real_answer.replace("rezonans 2: 3", "rezonans 2:3")
                        real_answer = real_answer.replace("rezonansem 2: 3", "rezonansem 2:3")
                        real_answer = real_answer.replace("nie przeszkadzać ”", "nie przeszkadzać")
                        original_answer = original_answer.replace("nie przeszkadzać ”", "nie przeszkadzać")
                        real_answer = real_answer.replace("Macintosha 128 K", "Macintosha 128K")
                        real_answer = real_answer.replace("soli miedzi (II", "soli miedzi(II)")
                        original_answer = original_answer.replace("soli miedzi (II", "soli miedzi(II)")
                        real_answer = real_answer.replace("sole miedzi (II", "sole miedzi(II)")
                        original_answer = original_answer.replace("sole miedzi (II", "sole miedzi(II)")
                        real_answer = real_answer.replace("tlenek ołowiu (II", "tlenek ołowiu(II)")
                        real_answer = real_answer.replace("km / h", "km/h")
                        real_answer = real_answer.replace("720 p", "720p")
                        real_answer = real_answer.replace("90 ° względem siebie", "90° względem siebie")
                        real_answer = real_answer.replace("subject–verb–object", "subject – verb – object")
                        real_answer = real_answer.replace("Alzacja - Lotaryngia", "Alzacja-Lotaryngia")
                        real_answer = real_answer.replace("La Isla Bonita ”", "La Isla Bonita")
                        original_answer = original_answer.replace("La Isla Bonita ”", "La Isla Bonita")
                        real_answer = real_answer.replace("HIV / AIDS", "HIV/AIDS")
                        real_answer = real_answer.replace("urządzenia wejścia / wyjścia", "urządzenia wejścia/wyjścia")
                        real_answer = real_answer.replace('" Into the Groove "', '"Into the Groove"')
                        original_answer = original_answer.replace('" Into the Groove "', '"Into the Groove"')

                        if real_answer not in sentence["text"]:
                            print(
                                f"Answer not in a sentence (real: {real_answer} | original: {original_answer} | "
                                f"sentence: {sentence['text']})"
                            )
                            continue

                        item = {
                            "question": question,
                            "question_id": question_id,
                            "original_answer": original_answer,
                            "answer": real_answer,
                            "sentence": sentence["text"],
                            "tag": sentence["tag"],
                            "similarity": sentence["similarity"],
                            # for statistics
                            "answer_ner": answer_ner,
                            "depth": depth,
                            "sentence_num": sentence_num,
                            "place": sentence["place"],
                        }
                        if (question_id, real_answer) in result:
                            # there is already some item set
                            # set the new one only, if it has better similarity
                            if item["similarity"] > result[(question_id, real_answer)]["similarity"]:
                                result[(question_id, real_answer)] = item
                        else:
                            result[(question_id, real_answer)] = item
        return result

    def merge_results(processed):
        """Merge results taking the one with bigger similarity value."""
        if len(processed) == 1:
            return list(processed[0].values())

        if len(processed) != 2:
            raise NotImplementedError()

        data1, data2 = processed
        result = {**data1, **data2}
        for key in data1.keys() & data2.keys():
            result[key] = max(data1[key], data2[key], key=lambda d: d["similarity"])

        return list(result.values())

    processed = []
    for data, depth, sentence_num in to_process:
        processed.append(process_data(data, depth, sentence_num))

    result = merge_results(processed)

    # n = depth + 2 * int(join)
    # with open(f"result{n}.txt", "w") as output:
    #     for (question_id, answer), question_answer in result.items():
    #         assert answer == question_answer["answer"]
    #         assert question_id == question_answer["question_id"]
    #
    #         print(f"Question: {question_answer['question']}", file=output)
    #         print(f"Question ID: {question_answer['question_id']}", file=output)
    #         print(f"Original answer: {question_answer['original_answer']}", file=output)
    #         print(f"Real answer: {question_answer['real_answer']}", file=output)
    #         print(f"Answer Ner: {question_answer['answer_ner']}", file=output)
    #         print(f"Sentence: {question_answer['sentence']}", file=output)
    #         print(f"Tag: {question_answer['tag']}", file=output)
    #         print(f"Similarity: {question_answer['similarity']}", file=output)
    #         print(f"Depth: {question_answer['depth']}", file=output)
    #         print(f"Sentence num: {question_answer['sentence_num']}", file=output)
    #         print(f"Place: {question_answer['place']}", file=output)
    #         print(64 * "=", file=output)

    return result


if __name__ == "__main__":
    # manual_filter_wiki_similar_sentences_part_1()
    # manual_filter_wiki_similar_sentences_part_2()
    process_wiki_similar_sentences(depth=0, join=False)
    process_wiki_similar_sentences(depth=1, join=False)
    process_wiki_similar_sentences(depth=0, join=True)
    process_wiki_similar_sentences(depth=1, join=True)
