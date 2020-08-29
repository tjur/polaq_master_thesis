import csv
import re
from collections import defaultdict

from nameparser import HumanName
from names_dataset import NameDataset

from squad_pl import DATA_DIR
from squad_pl.doc2vec.utils import polish_nlp
from squad_pl.translate.translate_answers import load_google_answer_translations
from squad_pl.translate.utils import is_roman_numeral, is_sublist

NORMALIZED_NES = {
    "date": "DATE",
    "persName": "PERSON",
    "orgName": "ORGANIZATION",
    "placeName": "LOCATION",
    "geogName": "LOCATION",
    "time": "TIME",
}


def normalize_named_entity(named_entity):
    if named_entity == "O" or "|" in named_entity:
        named_entity = "OTHER"
    return NORMALIZED_NES.get(named_entity, named_entity)


def extend_date_question_answer(question, question_answer, extended):
    """Extend question-answer with a date type answer."""
    answer = question_answer["answer"]
    sentence = question_answer["sentence"]
    sentence_nes = question_answer["sentence_nes"]
    question_lemmas = question_answer["question_lemmas"]

    full_date_match = re.fullmatch(r"(?P<day>\d+) (?P<month>\D+) (?P<year>\d+)", answer)
    year_match = re.fullmatch(r"\d+", answer)

    if year_match:
        # "1994" format
        if question_lemmas[:3] == ["w", "który", "rok"]:
            # "W którym roku ..." --> "Kiedy ..."
            new_question_answer = question_answer.copy()
            new_question_answer["extended"] = True
            new_question = " ".join(["Kiedy"] + question.split(" ")[3:])
            extended[new_question].append(new_question_answer)

        if question_lemmas[0] == "kiedy":
            # "Kiedy ..." --> "W którym roku ..."
            new_question_answer = question_answer.copy()
            new_question_answer["extended"] = True
            new_question = " ".join(["W", "którym", "roku"] + question.split(" ")[1:])
            extended[new_question].append(new_question_answer)

    if full_date_match:
        # "15 listopada 1994" format
        day = full_date_match.group("day")
        month = full_date_match.group("month")
        year = full_date_match.group("year")

        if question_lemmas[0] == "kiedy":
            new_question_answer = question_answer.copy()
            new_question_answer["answer"] = year
            new_question_answer["answer_start"] += answer.index(year)
            new_question_answer["extended"] = True
            new_question = " ".join(["W", "którym", "roku"] + question.split(" ")[1:])
            extended[new_question].append(new_question_answer)

        # day and month can possibly be used for generating another answers and questions like:
        # "W jaki dzień", "W jakim miesiącu", "Jaka była data"

    full_date_from_sentence = next(
        (
            re.sub(r"(roku|r\.|(?<!gru)dniu|\*|)", "", text).strip()
            for text, ne in sentence_nes.items()
            if (
                ne == "date"
                and answer in text
                and answer != text
                and is_sublist(answer.lower().split(" "), text.lower().split(" "))
                and "latach" not in answer
                and "lata" not in answer
                and sentence.index(text) + text.index(answer) == sentence.index(answer)
            )
        ),
        None,
    )
    if full_date_from_sentence and question_lemmas[:3] == ["w", "który", "rok"]:
        # there is a full date with our date answer as a part of it in a sentence
        # use it as a new answer with a slightly changed question
        new_question_answer = question_answer.copy()
        new_question_answer["answer"] = full_date_from_sentence
        new_question_answer["answer_start"] -= full_date_from_sentence.index(answer)
        new_question_answer["extended"] = True
        new_question = " ".join(["Kiedy"] + question.split(" ")[3:])
        extended[new_question].append(new_question_answer)

    if full_date_from_sentence and question_lemmas[0] == "kiedy":
        # there is a full date with our date answer as a part of it in a sentence
        # use it as a new answer with the same question
        new_question_answer = question_answer.copy()
        new_question_answer["answer"] = full_date_from_sentence
        new_question_answer["answer_start"] -= full_date_from_sentence.index(answer)
        new_question_answer["extended"] = True
        extended[question].append(new_question_answer)


name_dataset = NameDataset()


def extend_person_question_answer(question, question_answer, extended):
    """Extend question-answer with a person type answer."""
    answer = question_answer["answer"]
    sentence = question_answer["sentence"]
    sentence_nes = question_answer["sentence_nes"]

    person = HumanName(answer)

    # double check first name with another third party library (names-dataset)
    if not name_dataset.search_first_name(person.first):
        return

    if len(answer.split(" ")) > 1 and not is_roman_numeral(person.last):
        # if person answer has more than 1 token
        # then use last name as a separate answer (if it's not a roman numeral)
        new_question_answer = question_answer.copy()
        new_question_answer["answer"] = person.last
        new_question_answer["answer_start"] += answer.index(person.last)
        new_question_answer["extended"] = True
        extended[question].append(new_question_answer)

    full_person = next(
        (
            text
            for text, ne in sentence_nes.items()
            if (
                ne == "persName"
                and answer in text
                and answer != text
                and is_sublist(answer.lower().split(" "), text.lower().split(" "))
                and sentence.index(text) + text.index(answer) == sentence.index(answer)
            )
        ),
        None,
    )

    if full_person:
        # current person answer is a part of a bigger person named entity
        # use it for a new answer with the same question
        new_question_answer = question_answer.copy()
        new_question_answer["answer"] = full_person
        new_question_answer["answer_start"] -= full_person.index(answer)
        new_question_answer["extended"] = True
        extended[question].append(new_question_answer)


CITIES_DATASET_PATH = DATA_DIR / "other/world-cities.csv"
GOOGLE_ANSWER_TRANSLATIONS = load_google_answer_translations()


def load_cities_dataset():
    cities = set()
    with open(CITIES_DATASET_PATH, "r") as cities_file:
        csv_reader = csv.reader(cities_file, delimiter=",")
        for city, country, subcountry, geonameid in csv_reader:
            cities.add(city)
            cities.add(GOOGLE_ANSWER_TRANSLATIONS.get(city, city))  # add translated city name
    return cities


CITIES = load_cities_dataset()


def is_city(text):
    return text in CITIES


def extend_location_question_answer(question, question_answer, extended):
    """Extend question-answer with a location type answer."""
    answer = question_answer["answer"]
    question_tokens = question_answer["question_tokens"]

    if is_city(answer) and question_tokens[0] == "Gdzie":
        new_question_answer = question_answer.copy()
        new_question_answer["extended"] = True
        new_question = " ".join(["W", "jakim", "mieście"] + question.split(" ")[1:])
        extended[new_question].append(new_question_answer)

    elif question_tokens[:3] == ["W", "jakim", "mieście"]:
        new_question_answer = question_answer.copy()
        new_question_answer["extended"] = True
        new_question = " ".join(["Gdzie"] + question.split(" ")[3:])
        extended[new_question].append(new_question_answer)


def extend_number_question_answer(question, question_answer, extended):
    """Extend question-answer with a number type answer."""
    answer = question_answer["answer"]
    question_tokens = question_answer["question_tokens"]
    question_lemmas = question_answer["question_lemmas"]
    sentence = question_answer["sentence"]

    if question_tokens[0] == "Ile":
        new_question_answer = question_answer.copy()
        new_question_answer["extended"] = True
        new_question = " ".join(["Jak", "wiele"] + question.split(" ")[1:])
        extended[new_question].append(new_question_answer)

        word_after_answer = re.sub(
            r"(,|\.|\)|:)", "", sentence[sentence.index(answer) + len(answer) + 1 :].lstrip().split(" ", 1)[0]
        )
        if not word_after_answer:
            return
        processed_word = polish_nlp(word_after_answer)[0]
        if processed_word.pos_ == "NOUN" and question_lemmas[1] == processed_word.lemma_:
            # if in a sentence the word right after the number (answer) is a noun and
            # this noun (its lemma) is a part of a question in a form
            # "Ile <noun> ...?" then answer "<number> <noun>" is also correct
            new_question_answer = question_answer.copy()
            new_question_answer["answer"] = f"{answer} {word_after_answer}"
            new_question_answer["extended"] = True
            extended[question].append(new_question_answer)


# do not use "Jaką" - "Którą" - gives bad results
SIMILAR_ADJECTIVES = {
    "Jaki": "Który",
    "Jaka": "Która",
    "Jakie": "Które",
    "Jakiego": "Którego",
    "Jakiej": "Której",
    "Jacy": "Którzy",
    "Który": "Jaki",
    "Która": "Jaka",
    "Które": "Jakie",
    "Którego": "Jakiego",
    "Której": "Jakiej",
    "Którzy": "Jacy",
}


def extend_other_question_answer(question, question_answer, extended):
    """Extend question-answer with an other type answer."""
    answer = question_answer["answer"]
    question_tokens = question_answer["question_tokens"]
    question_lemmas = question_answer["question_lemmas"]
    question_poss = question_answer["question_poss"]
    processed_answer = polish_nlp(answer)

    if len(processed_answer) == 2 and question_lemmas[0] in ["jaki", "który"]:
        answer_poss = {token.lemma_: token.pos_ for token in processed_answer}
        if answer_poss.get(question_lemmas[1]) == "NOUN" and (
            "ADJ" in answer_poss.values() or len(set(answer_poss.values())) == 1
        ):
            # if answer has 2 tokens and one is a noun and the other one
            # is either noun or adjective and question starts
            # with a word similar to "Jaki", "Który"
            # and then has first word (lemma) from answer
            # then second word can be separate answer
            if processed_answer[0].lemma_ == question_lemmas[1]:
                new_answer = processed_answer[1].text
            else:
                new_answer = processed_answer[0].text

            new_question_answer = question_answer.copy()
            new_question_answer["answer"] = new_answer
            new_question_answer["answer_start"] += answer.index(new_answer)
            new_question_answer["extended"] = True
            extended[question].append(new_question_answer)

    if question_tokens[0] in SIMILAR_ADJECTIVES and question_poss[1] == "NOUN":
        new_question_answer = question_answer.copy()
        new_question_answer["extended"] = True
        new_question = " ".join([SIMILAR_ADJECTIVES[question_tokens[0]]] + question.split(" ")[1:])
        extended[new_question].append(new_question_answer)


def extend_questions_answers(questions_answers, manual):
    extended = defaultdict(list)
    for question, question_answers in questions_answers.items():
        processed_question = polish_nlp(question)
        question_tokens = [token.text for token in processed_question]
        question_lemmas = [token.lemma_ for token in processed_question]
        question_poss = [token.pos_ for token in processed_question]
        for question_answer in question_answers:
            if question_answer.get("extended"):
                continue

            extended[question].append(question_answer)

            answer = question_answer["answer"]
            answer_ne = question_answer["answer_ne"]
            processed_sentence = polish_nlp(question_answer["sentence"])

            sentence_nes = {entity.text: entity.label_ for entity in processed_sentence.ents}
            for text, named_entity in sentence_nes.items():
                if (
                    (answer_ne == "O" or "|" in answer_ne)
                    and answer in text
                    and is_sublist(answer.lower().split(" "), text.lower().split(" "))
                ):
                    # change answer's named entity only if we found category for the answer
                    # using polish spaCy model and answer doesn't have a named entity yet
                    # (so it's "O" or something like "O|O|O")
                    answer_ne = named_entity
                    break
            # both CoreNLP and polish spacy model are used for named entity recognition
            # and they use different names for the same categories
            answer_ne = normalize_named_entity(answer_ne)

            question_answer["answer_ne"] = answer_ne
            question_answer["sentence_nes"] = sentence_nes
            question_answer["question_tokens"] = question_tokens
            question_answer["question_lemmas"] = question_lemmas
            question_answer["question_poss"] = question_poss

            if not manual and question_answer["similarity"] < 0.8:
                # skip extending in non manual (generated) generation mode unless similarity is high
                continue

            if answer_ne == "DATE":
                extend_date_question_answer(question, question_answer, extended)
            elif answer_ne == "PERSON":
                extend_person_question_answer(question, question_answer, extended)
            elif answer_ne == "LOCATION":
                extend_location_question_answer(question, question_answer, extended)
            elif answer_ne == "NUMBER":
                extend_number_question_answer(question, question_answer, extended)
            elif answer_ne == "OTHER":
                extend_other_question_answer(question, question_answer, extended)

    return extended
