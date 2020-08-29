import json
import os
import pickle
import re
from collections import defaultdict
from time import sleep
from typing import Set, Dict, DefaultDict

from names_dataset import NameDataset

from squad_pl import DATA_DIR, logger, SQUAD_PATH
from squad_pl.doc2vec.utils import polish_nlp
from squad_pl.proto.io import read_articles
from squad_pl.proto.dataset_pb2 import QuestionAnswer
from squad_pl.translate.utils import (
    UNWANTED_CHARS_IN_PERSON_TRANSLATION,
    UNWANTED_CHARS_IN_MISC_TRANSLATION,
    POLISH_ORDINALS,
    ORDINAL_WORDS_TRANSLATION,
    TIME_WORDS_TRANSLATION,
    DURATION_WORDS_TRANSLATION,
    DATE_WORDS_TRANSLATION,
    ROMAN_NUMERALS,
    is_letter_only,
    number_word_to_int,
    number_to_text_number,
    is_roman_numeral,
    split_from_currency,
    translate_gcp,
    GCP_REQUEST_LENGTH_LIMIT,
    GCP_SUBREQUEST_LIMIT,
    load_wiki_titles_translation,
    date_word_translation,
)

ANNOTATED_QAS_FILENAME = DATA_DIR / "squad/annotated/annotated_qas"

TRANSLATED_ANSWERS_GOOGLE_PATH = DATA_DIR / f"squad/translated/translated_answers_google"

TRANSLATED_ANSWERS_PATH = DATA_DIR / f"squad/translated/translated_answers"


def translate_answers_with_gcp(squad_path, wait=101):
    """Translate SQuAD answers using GCP translator."""
    with open(squad_path, "r") as squad_file:
        squad = json.load(squad_file)["data"]
        answers_to_translate = []
        answers_length = 0
        total_processed = 0

        if os.path.exists(TRANSLATED_ANSWERS_GOOGLE_PATH):
            with open(TRANSLATED_ANSWERS_GOOGLE_PATH, "rb") as translated_answers_file:
                already_translated_answers = set(pickle.load(translated_answers_file).keys())
        else:
            already_translated_answers = set()

        logger.info(f"Starting with {len(already_translated_answers)} already translated answers.")

        for article in squad:
            for paragraph in article["paragraphs"]:
                for question_answers in paragraph["qas"]:
                    for answer in question_answers["answers"]:
                        answer = answer["text"].strip()
                        total_processed += 1
                        # answer = answer["text"].strip()
                        if answer in already_translated_answers:
                            continue
                        answers_to_translate.append(answer)
                        answers_length += len(answer)
                        # save to not ask for a translation of the same answer twice
                        already_translated_answers.add(answer)

                    if answers_length >= GCP_REQUEST_LENGTH_LIMIT or len(answers_to_translate) >= GCP_SUBREQUEST_LIMIT:
                        if os.path.exists(TRANSLATED_ANSWERS_GOOGLE_PATH):
                            with open(TRANSLATED_ANSWERS_GOOGLE_PATH, "rb") as translated_answers_file:
                                translated_answers = pickle.load(translated_answers_file)
                        else:
                            translated_answers = {}

                        result = translate_gcp(answers_to_translate)

                        for i, answer in enumerate(answers_to_translate):
                            translated_answers[answer] = result[i]

                        with open(TRANSLATED_ANSWERS_GOOGLE_PATH, "wb") as translated_answers_file:
                            pickle.dump(translated_answers, translated_answers_file)

                        logger.info(f"Translated {len(answers_to_translate)} answers in a request")
                        logger.info(f"Translated {len(translated_answers.keys())} unique in total")
                        logger.info(f"Translated {total_processed} in total")
                        answers_to_translate = []
                        answers_length = 0
                        logger.info(f"Wait {wait} seconds before next request")
                        sleep(wait)

        #  iteration is over but there might be some answers left to translate
        if answers_to_translate:
            with open(TRANSLATED_ANSWERS_GOOGLE_PATH, "rb") as translated_answers_file:
                translated_questions = pickle.load(translated_answers_file)

            result = translate_gcp(answers_to_translate)

            for i, answer in enumerate(answers_to_translate):
                translated_questions[answer] = result[i]

            with open(TRANSLATED_ANSWERS_GOOGLE_PATH, "wb") as translated_answers_file:
                pickle.dump(translated_questions, translated_answers_file)

            logger.info(f"Translated {len(answers_to_translate)} answers in a request")
            logger.info(f"Translated {len(translated_answers.keys())} unique in total")
            logger.info(f"Translated {total_processed} in total")


def load_google_answer_translations() -> DefaultDict[str, str]:
    with open(TRANSLATED_ANSWERS_GOOGLE_PATH, "rb") as f:
        google_answer_translation = pickle.load(f)

    google_answer_translation.update({"thesis 86": "teza 86"})
    # make result a deafultdict that returns empty string when there is no translation
    # this is the case when we ask for a translation of answer that is not from SQuAD
    # (we only translated SQuAD answers but proto CoreNLP annotation files contains some additional data)
    return defaultdict(lambda: "", google_answer_translation)


def get_core_nlp_qas_annotations():
    annotated_qas = {}
    context_tokens = {}
    for name in ["dev", "train"]:
        for annotated_article in read_articles(DATA_DIR / f"squad/annotated/{name}-annotated.proto"):
            for paragraph in annotated_article.paragraphs:
                tokens = []
                for sentence in paragraph.context.sentence:
                    tokens += [{"text": token.originalText, "ner": token.ner} for token in sentence.token]
                for question_answers in paragraph.qas:
                    annotated_qas[question_answers.id] = question_answers.SerializeToString()
                    context_tokens[question_answers.id] = tokens
    return annotated_qas, context_tokens


def save_annotated_qas_to_file():
    with open(ANNOTATED_QAS_FILENAME, "wb") as annotated_qas_file:
        annotated_qas, context_tokens = get_core_nlp_qas_annotations()
        pickle.dump((annotated_qas, context_tokens), annotated_qas_file)


def load_annotated_qas():
    with open(ANNOTATED_QAS_FILENAME, "rb") as annotated_qas_file:
        annotated_qas, context_tokens = pickle.load(annotated_qas_file)
        for qa_id, qa in annotated_qas.items():
            deserialized_qa = QuestionAnswer()
            deserialized_qa.ParseFromString(qa)
            annotated_qas[qa_id] = deserialized_qa
        return annotated_qas, context_tokens


IGNORED_TOKENS = [".", ",", '"', "'", "\\", "/", "``", "-", "–", "the", "a", "an"]

"""
Simple answer - it's an answer that has some NER category assigned (all except "O" - Other category) which means
that all answer's tokens have exactly the same NER category
So the answer is simple if we know something about it (and we use it in translation)
All found NER categories:
{'PERSON', 'TIME', 'MISC', 'ORDINAL', 'ORGANIZATION', 'DATE', 'PERCENT', 'LOCATION', 'NUMBER', 'MONEY'}
"""


MAX_TOKENS_IN_SQUAD_ANSWER = 9

INCORRECT_NER_ANNOTATIONS = {
    "Ike": "MISC",
    "Thesis 86": "O",
    "Season 11": "O",
    "over 1000": "NUMBER",
    "mid-1850s": "DATE",
    "mid-1970s": "DATE",
    "mid-2000s": "DATE",
    "anonymous": "O",
    "three": "NUMBER",
    "3000 years": "DURATION",
    "20 years": "DURATION",
    "two": "NUMBER",
    "Constantinople": "LOCATION",
    "Bladderball": "O",
    "two decades": "DURATION",
    "four years": "DURATION",
    "two years": "DURATION",
    "mid-1950s": "DATE",
    "mid-1960s": "DATE",
}


def get_answers_with_ner():
    """Get all answers from SQuAD and theirs NER categories using CoreNLP annotations."""
    annotated_qas, contexts_tokens = load_annotated_qas()
    answers_ner = defaultdict(dict)
    known_ner = {}
    name_dataset = NameDataset()

    for qas_id, annotated_qas in annotated_qas.items():
        context_tokens = contexts_tokens[qas_id]
        for answer in annotated_qas.answers:
            # there are some empty answers in the dataset
            if answer.sentence:
                # There are only 105 multi sentence answers in SQuAD 1.1 (these are mostly weird or incorrect answers)
                # but none of them has all tokens with the same NER category (fortunately)

                # check if all answer's tokens have exactly the same NER category
                # if so, the whole answer has this NER (exclude "Other" NER category case here and handle it in else)
                cats = set(
                    token.ner
                    for sentence in answer.sentence
                    for token in sentence.token
                    if token.originalText.lower() not in IGNORED_TOKENS
                )
                answer_tokens = []
                for sentence in answer.sentence:
                    for token in sentence.token:
                        answer_tokens.append(token.originalText)

                cats_from_context = set()
                for i in range(len(context_tokens) - len(answer_tokens) + 1):
                    if context_tokens[i]["text"] == answer_tokens[0]:
                        part_of_context = [token["text"] for token in context_tokens[i : i + len(answer_tokens)]]
                        if part_of_context == answer_tokens:
                            cats_from_context = set(
                                token["ner"]
                                for token in context_tokens[i : i + len(answer_tokens)]
                                if token["text"].lower() not in IGNORED_TOKENS
                            )

                if cats != cats_from_context:
                    if len(cats_from_context) == 1 and cats_from_context != {"O"}:
                        cats = cats_from_context

                answer_text = answer.text.strip()
                if len(cats) == 1 and "O" not in cats:
                    answers_ner[qas_id][answer_text] = list(cats)[0]
                    known_ner[answer_text] = list(cats)[0]

                elif len(answer.sentence) == 1 and 0 < len(answer.sentence[0].token) <= MAX_TOKENS_IN_SQUAD_ANSWER:

                    # check if it is a number - change its NER if so
                    try:
                        float(answer_text)
                        answers_ner[qas_id][answer_text] = "NUMBER"
                        known_ner[answer_text] = "NUMBER"
                        continue
                    except ValueError:
                        pass

                    # check if answer is like "in 1994" or "in November 1994" - then it has a date NER
                    if re.fullmatch(
                        rf"^(in )?(({'|'.join(DATE_WORDS_TRANSLATION.keys())}) )?\d\d\d\d$",
                        answer_text,
                        flags=re.IGNORECASE,
                    ):
                        answers_ner[qas_id][answer_text] = "DATE"
                        known_ner[answer_text] = "DATE"
                        continue

                    tokens = answer.sentence[0].token

                    # check if it's a person
                    # use `names-dataset` package
                    if (
                        len(tokens) < 5
                        and all(token.word[0].isupper() for token in tokens)
                        and (
                            (
                                len(tokens) == 2
                                and name_dataset.search_first_name(tokens[0].word)
                                and (name_dataset.search_last_name(tokens[1].word) or is_roman_numeral(tokens[1].word))
                            )
                            or (
                                len(tokens) == 3
                                and name_dataset.search_first_name(tokens[0].word)
                                and (
                                    name_dataset.search_first_name(tokens[1].word)
                                    or name_dataset.search_last_name(tokens[1].word)
                                )
                                and (name_dataset.search_last_name(tokens[2].word) or is_roman_numeral(tokens[2].word))
                            )
                            or (
                                len(tokens) == 4
                                and name_dataset.search_first_name(tokens[0].word)
                                and (
                                    name_dataset.search_first_name(tokens[1].word)
                                    or name_dataset.search_last_name(tokens[1].word)
                                )
                                and (
                                    name_dataset.search_first_name(tokens[2].word)
                                    or name_dataset.search_last_name(tokens[2].word)
                                )
                                and (name_dataset.search_last_name(tokens[3].word) or is_roman_numeral(tokens[3].word))
                            )
                        )
                    ):
                        answers_ner[qas_id][answer_text] = "PERSON"
                        known_ner[answer_text] = "PERSON"
                        continue

                    # try annotation with spaCy polish model
                    # named_entities = {entity.text: entity.label_ for entity in polish_nlp(answer_text).ents}
                    # if answer_text in named_entities:
                    #     answers_ner[qas_id][answer_text] = named_entities[answer_text]
                    #     known_ner[answer_text] = named_entities[answer_text]
                    #     continue

                    # otherwise don't look at NER categories - but discard here all multi sentence answers
                    # (so all 105 answers mentioned above are rejected) and too long answers (because chances
                    # for correct translations and using them in polish squad are almost equal to zero)
                    # give that answer category of all concatenated tokens' NER categories
                    answers_ner[qas_id][answer_text] = "|".join([token.ner for token in answer.sentence[0].token])

    # iterate again and update some unrecognized answers (with "Other" NER) that
    # possibly have been marked with some NER in other case
    # correct also some bad annotations
    for qas_id, answers in answers_ner.items():
        for answer_text, ner in answers.items():
            assert not ("|" in ner and len(set(ner.split("|"))) == 1 and ner.split("|", 1)[0] != "O")
            if ("|" in ner or ner == "O") and answer_text in known_ner:
                answers_ner[qas_id][answer_text] = known_ner[answer_text]
            if answer_text in INCORRECT_NER_ANNOTATIONS:
                answers_ner[qas_id][answer_text] = INCORRECT_NER_ANNOTATIONS[answer_text]

    return answers_ner


def translate_person_answer(answer_text) -> Set[str]:
    if (
        answer_text in TITLES_EN_PL
        and not (UNWANTED_CHARS_IN_PERSON_TRANSLATION & set(TITLES_EN_PL[answer_text]))
        and len(TITLES_EN_PL[answer_text]) <= 2 * len(answer_text)
    ):
        # In SQuAD 1.1 there were found 5349 answer_text matchings in TITLES_EN_PL
        # (3370 unique answers_text)
        # sometimes we get strange polish title so use some filtering here
        # with `UNWANTED_CHARS_IN_PERSON_TRANSLATION` and
        # len(TITLES_EN_PL[answer_text]) <= 2*len(answer_text) check
        # add here also original answer - full name in english
        return {TITLES_EN_PL[answer_text], GOOGLE_ANSWER_TRANSLATIONS[answer_text], answer_text}
    # person doesn't have polish article in wikipedia - use google translation
    # add here also original answer - full name in english - it will be probably correct in most cases
    return {GOOGLE_ANSWER_TRANSLATIONS[answer_text], answer_text}


def translate_time_answer(answer_text) -> Set[str]:
    answer_text = answer_text.replace("the ", "").replace("a ", "")
    if answer_text in TIME_WORDS_TRANSLATION:
        return {TIME_WORDS_TRANSLATION[answer_text]}
    # answer is a time in a format: hh:mm [a.m.|p.m.|pm]?
    if " " not in answer_text:
        return {answer_text}
    hour, period = answer_text.split(" ")
    if period in ["p.m.", "pm", "p.m"]:
        # add 12 hours - polish time format usually doesn't use am/pm
        h, m = hour.split(":")
        return {f"{int(h) + 12}:{m}", hour}
    return {hour}


def translate_misc_answer(answer_text) -> Set[str]:
    google_translation = GOOGLE_ANSWER_TRANSLATIONS[answer_text]
    if answer_text.startswith("the "):  # only when it starts with small "t"
        answer_text = answer_text[4:]
    if answer_text.startswith("a "):  # only when it starts with small "a"
        answer_text = answer_text[2:]
    if answer_text.startswith("an "):  # only when it starts with small "a"
        answer_text = answer_text[3:]

    if (
        answer_text in TITLES_EN_PL
        and not (UNWANTED_CHARS_IN_MISC_TRANSLATION & set(TITLES_EN_PL[answer_text]))
        and answer_text != TITLES_EN_PL[answer_text]
    ):
        # answer_text != TITLES_EN_PL[answer_text] check is here because
        # we got many translations like:
        # German -> German, Spanish -> Spanish, Western -> Western,
        # Eastern -> Eastern, Olympic -> Olympic, English Civil War -> English Civil War
        # (but also: Bohemian Rhapsody -> Bohemian Rhapsody, Windows RT -> Windows RT which are correct)
        # in that cases it would be better to use other kind of translation (e.g. Google Translate API)
        return {TITLES_EN_PL[answer_text], google_translation, answer_text}
    return {google_translation, answer_text}


def translate_ordinal_answer(answer_text) -> Set[str]:
    answer_text = answer_text.lower().replace("the ", "").replace("a ", "")
    if answer_text in ORDINAL_WORDS_TRANSLATION:
        # answer is a letter-only text
        number_str = ORDINAL_WORDS_TRANSLATION[answer_text]
        number = int(number_str)
        result = {POLISH_ORDINALS[int(number_str)], number_str}
    else:
        try:
            number = int(answer_text[:-2])
        except ValueError:
            # one case - "17th-18th"
            return {GOOGLE_ANSWER_TRANSLATIONS[answer_text]}
        if number in POLISH_ORDINALS:
            # one-word number
            result = {POLISH_ORDINALS[number], str(number)}
        else:
            # two-word number
            result = {f"{POLISH_ORDINALS[(number // 10) * 10]} {POLISH_ORDINALS[number % 10]}", str(number)}

    # "16th" as a part of "16th century" case
    if number in ROMAN_NUMERALS and number > 1:  # I is very common in polish so it would be better to skip it
        result.add(ROMAN_NUMERALS[number])
    return result


def translate_organization_answer(answer_text) -> Set[str]:
    google_translation = GOOGLE_ANSWER_TRANSLATIONS[answer_text]
    if answer_text.startswith("the "):  # only when it starts with small "t"
        answer_text = answer_text[4:]
    if answer_text.startswith("a "):  # only when it starts with small "a"
        answer_text = answer_text[2:]
    if answer_text.startswith("an "):  # only when it starts with small "a"
        answer_text = answer_text[3:]

    if answer_text in TITLES_EN_PL and len(TITLES_EN_PL[answer_text]) <= 2 * len(answer_text):
        # len(TITLES_EN_PL[answer_text]) <= 2*len(answer_text) check is a filter for weird translations
        return {TITLES_EN_PL[answer_text], google_translation, answer_text}
    return {google_translation, answer_text}


def translate_date_answer(answer_text) -> Set[str]:
    google_translation = GOOGLE_ANSWER_TRANSLATIONS[answer_text]
    answer_text = answer_text.replace(",", "").lower()  # remove commas
    # date is translated by google with a good quality
    google_translation = google_translation.replace(",", "").lower()  # remove commas

    # if answer has year in it, google translation adds "r" at the end (e.g. "15 listopada 1994 r")
    # we don't want that letter in an answer so remove it
    if re.fullmatch(r".*\d{1,4} r$", google_translation):
        google_translation = google_translation[:-2].rstrip()

    answer_text = answer_text.replace("the ", "")

    # simplify some answers
    answer_text = (
        answer_text.replace("the ", "")
        .replace("mid-", "")
        .replace("mid ", "")
        .replace("late ", "")
        .replace("early ", "")
        .replace("fall of ", "fall ")
        .replace("autumn of ", "autumn ")
        .replace("summer of ", "summer ")
        .replace("spring of ", "spring ")
        .replace("first half of ", "")
        .replace("second half of ", "")
        .replace("first decade of ", "")
        .replace("last decade of ", "")
        .replace("latter half of ", "")
        .replace("end of ", "")
        .replace("century bc", "century")
        .replace("century ce", "century")
    )

    if " " not in answer_text:
        # it's one word
        if is_letter_only(answer_text):
            return {date_word_translation(answer_text), google_translation}
        # answer is a year (most cases) or something year related like '1980s' or '1935/36', '1519-1521'
        if re.fullmatch(r"^(\d\d)?\d\ds$", answer_text):
            # 1980s
            return {f"lata {answer_text[-3:-1]}"}
        if re.fullmatch(r"^(\d+)(st|nd|rd|th)$", answer_text):
            return {str(int(answer_text[: re.search(r"(st|nd|rd|th)", answer_text).start()]))}
        if re.fullmatch(r"^(\D+)(–|-)(\D+)$", answer_text):
            # "march–june" format, just a couple of cases
            words = answer_text.split("–") if "–" in answer_text else answer_text.split("-")
            return {f"{date_word_translation(words[0])}-{date_word_translation(words[1])}"}

        # no need for translation
        # almost all have "1994" or "1398–1402" format
        return {answer_text}

    date = answer_text.split(" ")
    try:
        if re.fullmatch(r"in (\w| )+", answer_text):
            # "in November 1994" case
            # here "w"/"we" at the beginning of a translation is optional
            start, shorter_google_translation = google_translation.split(" ", 1)
            assert start in ["w", "we"]
            translations = translate_date_answer(answer_text.split(" ", 1)[1])
            return (
                translations
                | {f"{start} {translation}" for translation in translations if translation}
                | {google_translation}
            )

        if len(date) == 3 and re.fullmatch(r"\d+ \D+ \d+", answer_text):
            # "number word number" format like '25 October 1922' or '1918 to 1920'
            return {f"{date[0]} {date_word_translation(date[1], True)} {date[2]}", google_translation}

        if len(date) == 3 and re.fullmatch(r"\D+ \d+ \d+", answer_text):
            # "word number number" format like 'December 7, 1941'
            # change order here
            return {f"{date[1]} {date_word_translation(date[0], True)} {date[2]}", google_translation}

        if len(date) == 2 and re.fullmatch(r"\d+ \D+", answer_text):
            # "number word" format like '19 November'
            return {f"{date[0]} {date_word_translation(date[1], True)}", google_translation}

        if len(date) == 2 and re.fullmatch(r"\D+ \d+", answer_text):
            # "word number" format like 'December 28', "June 1994"
            # let's assume that if number is bigger that 31 then it is year
            # and if it's year then it should be at the end of answer
            # otherwise number is at the beginning
            if int(date[1]) > 31:
                # it's a year
                return {f"{date_word_translation(date[0])} {date[1]}", google_translation}
            # change order
            return {f"{date[1]} {date_word_translation(date[0], True)}", google_translation}

        if re.fullmatch(r"\d+(th|rd|nd|st) century", answer_text):
            number = int(answer_text[: re.search(r"(th|rd|nd|st)", answer_text).start()])
            return {
                # f"{POLISH_ORDINALS[number]} wiek",  no one writes in this format in polish
                f"{ROMAN_NUMERALS[number]} wiek"
            }

        if re.fullmatch(r"\w+ century", answer_text):
            ordinal_word = answer_text.split(" ", 1)[0]
            return {
                # f"{POLISH_ORDINALS[number]} wiek",  no one writes in this format in polish
                f"{ROMAN_NUMERALS[int(ORDINAL_WORDS_TRANSLATION[ordinal_word])]} wiek"
            }

        if re.fullmatch(r"\w+ to \w+", answer_text):
            # "june to october" case
            words = [word.strip() for word in answer_text.split(" to ", 1)]
            return {f"{date_word_translation(words[0])} do {date_word_translation(words[1])}"}

        # there are 65 answers left in this case
        # use just google translation for them
        return {google_translation}

    except KeyError:
        if re.fullmatch(r"^(\d+) or (\d+)$", answer_text):
            # "1994 or 1999" format
            numbers = [int(number) for number in answer_text.split("or")]
            return {f"{numbers[0]} lub {numbers[1]}"}

        if re.fullmatch(r"^(\d+) until (\d+)$", answer_text):
            numbers = [int(number) for number in answer_text.split("until")]
            return {f"{numbers[0]} do {numbers[1]}"}

        # weird cases - there is only a couple of such cases
        translations = {
            "late may 1645": "maj 1645",
            "pril 25 1976": "25 kwiecień 1976",
            "1963 from 12 may to 29 june": "od 12 maja do 29 czerwca 1963",
            "july–august 1943": "lipiec-sierpień 1943",
            "1880s to 1914": "1914",
        }
        return {translations[answer_text]}


def translate_percent_answer(answer_text) -> Set[str]:
    if re.fullmatch(r"^\d+((\.|,)\d+)?%$", answer_text):
        # in polish we usually write 12,3% not 12.3%
        return {answer_text.replace(".", ",")}
    if re.fullmatch(r"^\d+((\.|,)\d+)? percent$", answer_text):
        return {f"{answer_text[:-8].replace('.', ',')}%"}
    if re.fullmatch(r"^\D+ percent$", answer_text):
        # 14 cases
        number_word = answer_text[:-8].lower()
        return {f"{number_word_to_int(number_word)}%"}
    try:
        number = float(answer_text)
        number = int(number) if isinstance(number, float) and number.is_integer() else number
        return {str(number).replace(".", ",")}
    except:
        pass

    if answer_text == "eighty-seven":
        return {"87"}
    if answer_text == "Fourteen":
        return {"14"}

    # cases like "10-45%" or "-40%" - here we can leave it without translation
    return {answer_text}


def translate_location_answer(answer_text) -> Set[str]:
    # Locations should be translated similarly to persons
    google_translation = GOOGLE_ANSWER_TRANSLATIONS[answer_text]
    if answer_text.startswith("the "):
        answer_text = answer_text[4:]
    if answer_text in TITLES_EN_PL and len(TITLES_EN_PL[answer_text]) <= 2 * len(answer_text):
        # len(TITLES_EN_PL[answer_text]) <= 2*len(answer_text) check is a filter for weird translations
        return {TITLES_EN_PL[answer_text], google_translation, answer_text}
    # Location doesn't have polish article in wikipedia so
    # here it would be better to use google translation
    return {google_translation, answer_text}


def translate_number_answer(answer_text) -> Set[str]:
    number = None
    google_translation = GOOGLE_ANSWER_TRANSLATIONS[answer_text]
    answer_text = answer_text.replace(",", "")  # remove separating commas
    answer_text = answer_text.replace("a ", "").replace("the ", "")

    if re.fullmatch(r"^(\d|,|\.| )+$", answer_text):  # plain number - most common case
        try:
            number = float(answer_text.replace(" ", ""))
        except ValueError:
            # invalid number format like "91.379.615" (this is actually the only case here) - don't translate it
            return {answer_text}
    elif re.fullmatch(r"^(\D+|(\d|\.)+) *(thousand|million|billion|b|bn|trillion)$", answer_text):
        match = list(re.finditer(r"(thousand|million|billion|b|bn|trillion)$", answer_text))[0]
        word_number = answer_text[: match.start()].rstrip()
        try:
            number: float = float(word_number)
        except ValueError:
            number: int = number_word_to_int(word_number)

        type = match.group()
        if type == "thousand":
            number *= 1000
        elif type == "million":
            number = int(1000000 * number)
        elif type in ["billion", "b", "bn"]:
            number = int(1000000000 * number)
        elif type == "trillion":
            number = int(1000000000000 * number)
        else:
            raise ValueError()
    else:
        try:
            # text case
            number = number_word_to_int(answer_text)
        except ValueError:
            pass

    if number is not None:
        # cast number to int if possible to not have ".0" in str()
        number = int(number) if isinstance(number, float) and number.is_integer() else number
        text_number = number_to_text_number(number)
        number_str = str(number).replace(".", ",")
        result = {number_str}

        # if number is int then create also formatted version of a number with spaces, e.g. "5 000 000"
        if isinstance(number, int):
            result.add("{:,}".format(number).replace(",", " "))

        if text_number:
            result.update(text_number)

        return result

    if re.fullmatch(r"^(\d|\.)+ ?m$", answer_text):
        # problematic case because we don't know whether "m" stands for "million" or "metres"
        # assume both cases
        number = float(answer_text[:-1])
        number = int(number) if number.is_integer() else number

        # meters case
        number_str = str(number).replace(".", ",")
        result = {f"{number_str} m", f"{number_str} metr"}
        # if number is int then create also formatted version of a number with spaces, e.g. "5 000 000"
        if isinstance(number, int):
            formatted_number_str = "{:,}".format(number).replace(",", " ")
            result.add(f"{formatted_number_str} m")
            result.add(f"{formatted_number_str} metr")

        # million case
        number = int(1000000 * number)
        result.add(str(number))
        result.add("{:,}".format(number).replace(",", " "))  # formatted version with extra spaces

        text_number = number_to_text_number(number)
        if text_number:
            result.update(text_number)
        return result

    if is_roman_numeral(answer_text):
        # only 3 cases
        return {answer_text}

    if answer_text[0] == "₹":
        # there are just couple of cases with this currency
        # remove the currency sign
        return translate_number_answer(answer_text[1:].lstrip())

    # some special cases
    translations = {
        "even-numbered": {"parzysty"},
        "1.5-mile": {"1,5 mila"},
        "21st": {"dwudziesty pierwszy", "21"},
        "41st": {"czterdziesty pierwszy", "41"},
        "over 1000": {"ponad 1000"},
        "one to four": {"1 do 4", "jeden do cztery"},
    }
    if answer_text in translations:
        return translations[answer_text]

    if re.fullmatch(r"^(\d+) to (\d+)$", answer_text):
        numbers = [int(number) for number in answer_text.split("to")]
        return {f"{numbers[0]} do {numbers[1]}"}

    if re.fullmatch(r"^(\d+) and (\d+)$", answer_text):
        numbers = [int(number) for number in answer_text.split("and")]
        return {f"{numbers[0]} i {numbers[1]}"}

    if re.fullmatch(r"^(\d\d)?\d\ds$", answer_text):
        # 1870s -> lata 70, 70s -> lata 70
        return {f"lata {answer_text[-3:-1]}"}

    if re.fullmatch(r"\d+(th|rd|nd|st) century", answer_text):
        # "74th" case
        number = int(answer_text[: re.search(r"(th|rd|nd|st)", answer_text).start()])
        result = {str(number)}
        if number in ROMAN_NUMERALS and number > 1:
            # "16th century" case
            # I is very common in polish so it would be better to skip it
            result.add(ROMAN_NUMERALS[number])
        return result

    if re.search(r"[a-z]", answer_text):
        # if answer contains something more than digits and "-"
        # then it probably has some words so use GCP translation
        return {google_translation}

    # 204 answers that remained can be left without translating
    # these are answers like: "5½", "90°", "1⁄3", "600-900", "56:79", "2010/2011", "1024×768"
    return {answer_text}


def translate_money_answer(answer_text) -> Set[str]:
    answer_text = answer_text.replace(",", "")

    currency, text, currency_at_end = split_from_currency(answer_text)

    # make 2 translations - with currency word and without
    # plus 2 different number formatting - so it gives max 4 translations

    if re.fullmatch(r"^(\d|,|\.)+$", text):  # plain number
        number: float = float(text)
        number = int(number) if number.is_integer() else number
        number_str = str(number).replace(".", ",").strip()
        formatted_number_str = "{:,}".format(number).replace(",", " ").replace(".", ",")  # additional spaces
        return {
            f"{number_str} {currency}".rstrip(),
            number_str,
            f"{formatted_number_str} {currency}".rstrip(),
            formatted_number_str,
        }

    elif re.fullmatch(r"^(\D+|(\d|\.)+) *(hundred|thousand|million|m|billion|b|bn|trillion)$", text):
        match = list(re.finditer(r"(hundred|thousand|million|m|billion|b|bn|trillion)$", text))[0]
        word_number = text[: match.start()].rstrip()
        try:
            number: float = float(word_number)
            number = int(number) if number.is_integer() else number
        except ValueError:
            number: int = number_word_to_int(word_number)

        type = match.group()
        number_str = str(number).replace(".", ",")
        if type == "hundred":
            return {f"{number * 100} {currency}".rstrip(), f"{number * 100}"}

        formatted_number_str = "{:,}".format(number).replace(",", " ").replace(".", ",")  # additional spaces
        translations = set()

        for number_str in (number_str, formatted_number_str):
            if type == "thousand":
                translations.update({f"{number_str} tysiąc {currency}".rstrip(), f"{number_str} tysiąc"})
            if type in ["million", "m"]:
                translations.update({f"{number_str} milion {currency}".rstrip(), f"{number_str} milion"})
            if type in ["billion", "b", "bn"]:
                translations.update({f"{number_str} miliard {currency}".rstrip(), f"{number_str} miliard"})
            if type == "trillion":
                translations.update({f"{number_str} trylion".rstrip(), f"{number_str} trylion"})
        return translations
    else:
        translations = {"ten": {"10", "dziesięć"}}
        if answer_text in translations:
            return translations[answer_text]

    # there is a couple (< 10) of weird cases
    # they are probably wrong but return GCP translation
    return {GOOGLE_ANSWER_TRANSLATIONS[answer_text]}


def translate_duration_answer(answer_text) -> Set[str]:
    google_translation = GOOGLE_ANSWER_TRANSLATIONS[answer_text]
    answer_text = answer_text.replace("the ", "").replace("a ", "").replace(",", "")
    duration = answer_text.split(" ")

    # special cases
    result = {
        "three hundred years": {"300 lat", "trzysta lat"},
        "10 minutes 48 seconds": {"10 minut 48 sekund"},
        "55 minutes 5 seconds": {"55 minut 5 sekund"},
        "60 minutes 5 seconds": {"60 minut 5 sekund"},
    }.get(answer_text)
    if result:
        return result

    if duration[-1] not in DURATION_WORDS_TRANSLATION:
        try:
            translations = translate_number_answer(answer_text)  # word or digit number
            if {tr for tr in translations if tr}:
                return translations
        except:
            pass

    else:
        end = google_translation.rpartition(" ")[-1]
        try:
            translations = translate_number_answer(answer_text.rpartition(" ")[0])
        except:
            pass
        else:
            if {tr for tr in translations if tr}:
                return {f"{trans} {end}" for trans in translations if trans}

    raise ValueError()


def translate_other_answer(answer_text, ner) -> Set[str]:
    categories = ner.split("|")
    google_translation = GOOGLE_ANSWER_TRANSLATIONS.get(answer_text)

    if answer_text.startswith("the "):  # only when it starts with small "t"
        answer_text = answer_text[4:]
    if answer_text.startswith("a "):  # only when it starts with small "a"
        answer_text = answer_text[2:]
    if answer_text.startswith("an "):  # only when it starts with small "a"
        answer_text = answer_text[3:]

    if len(categories) == 1:
        # answer is just a one word with "Other" category (which can be everything)
        assert ner == "O", "There is some NER category that you didn't handle"
        return {google_translation, answer_text}

    # not classified date case - remove "r" letter from the end of a translation
    # if answer has year in it, google translation adds "r" at the end (e.g. "15 listopada 1994 r")
    # we don't want that letter in an answer so remove it
    if re.fullmatch(r".*\d{1,4} r$", google_translation):
        # this not necessarily has to be a date answer,
        # e.g. "down from 69.6% in 1970" -> "spadek z 69,6% w 1970 r"
        google_translation = google_translation[:-2].rstrip()

    return {google_translation, answer_text}


def translate_answers(answers_ner: dict):
    translated_answers = defaultdict(dict)

    for qas_id, answers in answers_ner.items():
        for answer_text, ner in answers.items():
            if not answer_from_squad(answer_text):
                # don't translate non-SQuAD answers - we don't need them
                # GOOGLE_ANSWER_TRANSLATIONS has all SQuAD answers translated but apparently
                # file with CoreNLP annotations contains also some answers that finally weren't added to SQuAD
                continue

            if ner == "PERSON":
                trans_answers = translate_person_answer(answer_text)
            elif ner == "TIME":
                trans_answers = translate_time_answer(answer_text)
            elif ner == "MISC":
                trans_answers = translate_misc_answer(answer_text)
            elif ner == "ORDINAL":
                trans_answers = translate_ordinal_answer(answer_text)
            elif ner == "ORGANIZATION":
                trans_answers = translate_organization_answer(answer_text)
            elif ner == "DATE":
                # In case of date translation different permutations of a translated answer can also
                # be correct (these permutations are not in a result but should be later considered!)
                trans_answers = translate_date_answer(answer_text)
                trans_answers -= {"może", "marsz"}  # remove some weird date translations
            elif ner == "PERCENT":
                trans_answers = translate_percent_answer(answer_text)
            elif ner == "LOCATION":
                trans_answers = translate_location_answer(answer_text)
            elif ner == "NUMBER":
                trans_answers = translate_number_answer(answer_text)
            elif ner == "MONEY":
                trans_answers = translate_money_answer(answer_text)
            elif ner == "DURATION":
                trans_answers = translate_duration_answer(answer_text)
            else:
                # These are other answers without one specific NER category
                # In SQuAD 1.1 there are ? answers in this category (?? unique)
                trans_answers = translate_other_answer(answer_text, ner)

            # remove empty translations (answers from outside of SQuAD aren't translated by gcp)
            translated_answers[qas_id][answer_text] = {
                trans_answer.lower() for trans_answer in trans_answers if trans_answer
            }

    return translated_answers


def all_translated_answers(translated_answers):
    return [
        answer
        for id, answers in translated_answers.items()
        for answer, tr_answers in answers.items()
        if tr_answers and answer in GOOGLE_ANSWER_TRANSLATIONS
    ]


def all_distinct_translated_answers(translated_answers):
    return {
        answer
        for id, answers in translated_answers.items()
        for answer, tr_answers in answers.items()
        if tr_answers and answer in GOOGLE_ANSWER_TRANSLATIONS
    }


def answer_from_squad(answer):
    # only SQuAD answers were translated but there are some extra answers in CoreNLP annotation files
    # so use GOOGLE_ANSWER_TRANSLATIONS to filter them out
    return answer in GOOGLE_ANSWER_TRANSLATIONS


def analyze_translated_answers(answers_ner, translated_answers):
    """
    Print translated answers count grouped by:
    - NER categories
    - number of answer's words
    - number of possible translations
    """
    by_category_count = defaultdict(int)
    by_word_count = defaultdict(int)
    by_translation_count = defaultdict(int)

    for qas_id, answers in answers_ner.items():
        for answer_text, ner in answers.items():
            if answer_from_squad(answer_text):
                ner = "OTHER" if ner == "O" or "|" in ner else ner
                by_category_count[ner] += 1
                by_category_count["ALL"] += 1

    for id, answers in translated_answers.items():
        for answer, tr_answers in answers.items():
            if answer_from_squad(answer):
                for translated_answer in tr_answers:
                    by_word_count[len(translated_answer.split(" "))] += 1
                by_translation_count[len(tr_answers)] += 1

    print(f"Number of distinct (not lowered) answers in SQuAD: {len(GOOGLE_ANSWER_TRANSLATIONS)}")
    print()

    print(f"Answers that have some translation: {len(all_translated_answers(translated_answers))}")
    print()

    print(f"Distinct answers that have some translation: {len(all_distinct_translated_answers(translated_answers))}")
    print()

    print(f"Answers (with repetitions) count by NER category:")
    for ner, number in sorted(by_category_count.items(), key=lambda p: p[1], reverse=True):
        print(f"{ner}  -->  {number}")
    print()

    print("Answers (with repetitions) count by number of translations:")
    for count, number in sorted(by_translation_count.items()):
        print(f"{count} translation{'s' if count != 1 else ''}  -->  {number}")
    print()

    print("Translated answers (with repetitions of source answer) count by number of words:")
    for count, number in sorted(by_word_count.items()):
        print(f"{count} word{'s' if count != 1 else ''}  -->  {number}")


def save_translated_answers():
    global TITLES_EN_PL
    global GOOGLE_ANSWER_TRANSLATIONS
    TITLES_EN_PL = load_wiki_titles_translation()
    GOOGLE_ANSWER_TRANSLATIONS = load_google_answer_translations()
    answers_ner = get_answers_with_ner()
    translated_answers = translate_answers(answers_ner)
    analyze_translated_answers(answers_ner, translated_answers)
    with open(TRANSLATED_ANSWERS_PATH, "wb") as translated_answers_file:
        pickle.dump(translated_answers, translated_answers_file)


def load_translated_answers():
    with open(TRANSLATED_ANSWERS_PATH, "rb") as translated_answers_file:
        return pickle.load(translated_answers_file)


# translate answers using Google Translate API
# translate_answers_with_gcp(SQUAD_PATH)


if __name__ == "__main__":
    TITLES_EN_PL = load_wiki_titles_translation()
    GOOGLE_ANSWER_TRANSLATIONS = load_google_answer_translations()
    answers_ner = get_answers_with_ner()
    translated_answers = translate_answers(answers_ner)
    analyze_translated_answers(answers_ner, translated_answers)
    # save_translated_answers()

#
# NER category information (`answers_ner`) must be kept for later processing!
#

"""
Number of distinct (not lowered) answers in SQuAD: 79093

Answers that have some translation: 97626

Distinct answers that have some translation: 69847

Answers (with repetitions) count by NER category:
ALL  -->  97626
OTHER  -->  55672
PERSON  -->  9146
DATE  -->  9018
LOCATION  -->  6550
NUMBER  -->  5573
ORGANIZATION  -->  5192
MISC  -->  3493
PERCENT  -->  1314
MONEY  -->  733
DURATION  -->  623
ORDINAL  -->  275
TIME  -->  37

Answers (with repetitions) count by number of translations:
1 translation  -->  27227
2 translations  -->  67475
3 translations  -->  2353
4 translations  -->  553
5 translations  -->  13
6 translations  -->  3
7 translations  -->  2

Translated answers (with repetitions of source answer) count by number of words:
1 word  -->  57070
2 words  -->  51935
3 words  -->  27951
4 words  -->  14237
5 words  -->  8879
6 words  -->  5255
7 words  -->  3349
8 words  -->  1839
9 words  -->  903
10 words  -->  86
11 words  -->  21
12 words  -->  14
13 words  -->  2
14 words  -->  1
18 words  -->  1
19 words  -->  2
"""
