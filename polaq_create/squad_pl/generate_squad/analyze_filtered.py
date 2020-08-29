import json
from collections import defaultdict

from squad_pl import DATA_DIR
from squad_pl.translate.translate_answers import get_answers_with_ner, load_translated_answers


def get_ner_for_polish_answer(question_id, polish_answer, answers_ner, translated_answers):
    if len(translated_answers[question_id]) == 1:
        return answers_ner[question_id][list(translated_answers[question_id].keys())[0]]

    for english_answer, polish_answers in translated_answers[question_id].items():
        if polish_answer.lower() in polish_answers:
            return answers_ner[question_id][english_answer]
        if any([(polish_answer.lower() in ans) or (ans in polish_answer.lower()) for ans in polish_answers]):
            return answers_ner[question_id][english_answer]
    raise NotImplementedError()


FILTERED_PATH_FORMAT = str(DATA_DIR / "squad/pl/filtered_similar_wiki_sentences_depth_{}_sentence_{}.json")


def analyze_filtered_answers(depth, sentence):
    answers_ner = get_answers_with_ner()
    translated_answers = load_translated_answers()

    by_category_count = defaultdict(int)
    filtered_path = FILTERED_PATH_FORMAT.format(depth, sentence)
    with open(filtered_path, "r") as filtered_file:
        filtered_data = json.load(filtered_file)

    for question_id, matchings in filtered_data.items():
        for match in matchings:
            for sim_sentences in match["similar_sentences"]:
                polish_answer = sim_sentences["answer"]
                ner = get_ner_for_polish_answer(question_id, polish_answer, answers_ner, translated_answers)
                if "|" in ner or ner == "O":
                    ner = "OTHER"
                by_category_count[ner] += 1
                by_category_count["ALL"] += 1

    for ner, number in sorted(by_category_count.items(), key=lambda p: p[1], reverse=True):
        print(f"{ner}  -->  {number}")
    print()

    """
    Results (depth 0 sentence 1):
    ALL  -->  6909
    OTHER  -->  2062
    DATE  -->  1240
    LOCATION  -->  993
    PERSON  -->  978
    ORGANIZATION  -->  596
    NUMBER  -->  518
    MISC  -->  331
    PERCENT  -->  91
    ORDINAL  -->  46
    DURATION  -->  41
    MONEY  -->  12
    TIME  -->  1
    
    Results (depth 0 sentence 2):
    ALL  -->  8497
    OTHER  -->  2499
    DATE  -->  1490
    PERSON  -->  1324
    LOCATION  -->  1224
    ORGANIZATION  -->  705
    NUMBER  -->  633
    MISC  -->  408
    PERCENT  -->  99
    ORDINAL  -->  54
    DURATION  -->  46
    MONEY  -->  13
    TIME  -->  2
    
    Results (depth 1 sentence 1):
    ALL  -->  17731
    OTHER  -->  5698
    LOCATION  -->  2775
    PERSON  -->  2744
    DATE  -->  2352
    ORGANIZATION  -->  1585
    MISC  -->  1080
    NUMBER  -->  1017
    PERCENT  -->  270
    ORDINAL  -->  97
    DURATION  -->  84
    MONEY  -->  26
    TIME  -->  3
    """


def compare_filtered(depth):
    filtered_path_sentence_1 = FILTERED_PATH_FORMAT.format(depth, 1)
    filtered_path_sentence_2 = FILTERED_PATH_FORMAT.format(depth, 2)

    with open(filtered_path_sentence_1, "r") as filtered_path_sentence_1_file:
        filtered_path_sentence_1_data = json.load(filtered_path_sentence_1_file)

    with open(filtered_path_sentence_2, "r") as filtered_path_sentence_2_file:
        filtered_path_sentence_2_data = json.load(filtered_path_sentence_2_file)

    sentence_1_result = defaultdict(dict)
    all_sentence_1 = 0

    for question_id, matchings in filtered_path_sentence_1_data.items():
        for match in matchings:
            for sim_sentences in match["similar_sentences"]:
                answer = sim_sentences["answer"]
                text = sim_sentences["sentences"][0]["text"]
                sentence_1_result[question_id][answer] = text
                all_sentence_1 += 1

    all_sentence_2 = 0
    new_matchings = 0
    common_matchings = 0
    different_matchings = 0  # different than for sentence 1

    for question_id, matchings in filtered_path_sentence_2_data.items():
        for match in matchings:
            for sim_sentences in match["similar_sentences"]:
                answer = sim_sentences["answer"]
                text = sim_sentences["sentences"][0]["text"]
                all_sentence_2 += 1
                if question_id not in sentence_1_result or answer not in sentence_1_result[question_id]:
                    new_matchings += 1
                    continue
                if sentence_1_result[question_id][answer] in text:
                    common_matchings += 1
                    continue
                different_matchings += 1

    print(f"All: {all_sentence_2}")
    print(f"New: {new_matchings}")
    print(f"Common: {common_matchings}")
    print(f"Different: {different_matchings}")
    print(
        f"Sentence 1 matches that weren't found in sentence 2: {all_sentence_1 - common_matchings - different_matchings}"
    )

    """
    Results (depth 0):
    All: 8497
    New: 2205
    Common: 5035
    Different: 1257
    Sentence 1 matches that weren't found in sentence 2: 617
    """


# analyze_filtered_answers(depth=0, sentence=1)
# analyze_filtered_answers(depth=0, sentence=2)
# analyze_filtered_answers(depth=1, sentence=1)

# compare_filtered(depth=0)
