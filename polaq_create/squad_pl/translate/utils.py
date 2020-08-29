import re
from typing import Dict, Union, Optional, Tuple, List, Set

import google
from google.cloud import translate as gcp_translate

from squad_pl import DATA_DIR, tranlator, logger

# set GOOGLE_APPLICATION_CREDENTIALS env variable to a path to a json file with credentials
try:
    GCP_CLIENT = gcp_translate.TranslationServiceClient()
except google.auth.exceptions.DefaultCredentialsError as err:
    logger.warning(err)
    GCP_CLIENT = None


def translate_googletrans(text: Union[str, List[str]]):
    result = tranlator.translate(text, src="en", dest="pl")
    if isinstance(text, list):
        return [translation.text for translation in result]
    return result.text


def translate_gcp(text: List[str]):
    result = GCP_CLIENT.translate_text(
        text,
        source_language_code="en",
        target_language_code="pl",
        parent="projects/uplifted-nuance-269009/locations/us-central1",
        model="projects/uplifted-nuance-269009/locations/us-central1/models/general/nmt",
        mime_type="text/plain",
    )
    return [translation.translated_text for translation in result.translations]


GOOGLETRANS_REQUEST_LENGTH_LIMIT = 14500  # according to googletrans docs limit is 15000 characters per text

GCP_REQUEST_LENGTH_LIMIT = 29500  # GCP Cloud Translation limit per 100 seconds is 30k characters
GCP_SUBREQUEST_LIMIT = 1020  # GCP Cloud Translation subrequests limit per one request (length of a sent list) is 1024


POLISH_ORDINALS = {
    1: "pierwszy",
    2: "drugi",
    3: "trzeci",
    4: "czwarty",
    5: "piąty",
    6: "szósty",
    7: "siódmy",
    8: "ósmy",
    9: "dziewiąty",
    10: "dziesiąty",
    11: "jedenasty",
    12: "dwunasty",
    13: "trzynasty",
    14: "czternasty",
    15: "piętnasty",
    16: "szesnasty",
    17: "siedemnasty",
    18: "osiemnasty",
    19: "dziewiętnasty",
    20: "dwudziesty",
    30: "trzydziesty",
    40: "czterdziesty",
    50: "pięćdziesiąty",
    60: "sześćdziesiąty",
    70: "siedemdziesiąty",
    80: "osiemdziesiąty",
    90: "dziewięćdziesiąty",
}

POLISH_NUMBERS = {
    0: "zero",
    1: "jeden",
    2: "dwa",
    3: "trzy",
    4: "cztery",
    5: "pięć",
    6: "sześć",
    7: "siedem",
    8: "osiem",
    9: "dziewięć",
    10: "dziesięć",
    11: "jedenaście",
    12: "dwanaście",
    13: "trzynaście",
    14: "czternaście",
    15: "piętnaście",
    16: "szesnaście",
    17: "siedemnaście",
    18: "osiemnaście",
    19: "dziewiętnaście",
    20: "dwadzieścia",
    30: "trzydzieści",
    40: "czterdzieści",
    50: "pięćdziesiąt",
    60: "sześćdziesiąt",
    70: "siedemdziesiąt",
    80: "osiemdziesiąt",
    90: "dziewięćdziesiąt",
    100: "sto",
    200: "dwieście",
    300: "trzysta",
    400: "czterysta",
    500: "pięćset",
    600: "sześćset",
    700: "siedemset",
    800: "osiemset",
    900: "dziewięćset",
}


ROMAN_NUMERALS = {
    1: "I",
    2: "II",
    3: "III",
    4: "IV",
    5: "V",
    6: "VI",
    7: "VII",
    8: "VIII",
    9: "IX",
    10: "X",
    11: "XI",
    12: "XII",
    13: "XIII",
    14: "XIV",
    15: "XV",
    16: "XVI",
    17: "XVII",
    18: "XVIII",
    19: "XIX",
    20: "XX",
    21: "XXI",
}


UNWANTED_CHARS_IN_PERSON_TRANSLATION = {":", "/", ","}


# TIME NER category has only few words
TIME_WORDS_TRANSLATION = {
    "evening": "wieczór",
    "night": "noc",
    "midnight": "północ",
    "morning": "ranek",
    "day": "dzień",
    "sundown": "zachód słońca",
    "late afternoons": "popołudnia",
    "12 minutes past 2": "2:12",
}


UNWANTED_CHARS_IN_MISC_TRANSLATION = {":", "/", ","}


# ORDINAL NER category has only few words (the others are just numbers)
ORDINAL_WORDS_TRANSLATION = {
    "first": "1",
    "second": "2",
    "third": "3",
    "fourth": "4",
    "fifth": "5",
    "sixth": "6",
    "seventh": "7",
    "eighth": "8",
    "ninth": "9",
    "tenth": "10",
    "eleventh": "11",
    "twelfth": "12",
    "thirteenth": "13",
    "fourteenth": "14",
    "fifteenth": "15",
    "sixteenth": "16",
    "seventeenth": "17",
    "eighteenth": "18",
    "nineteenth": "19",
    "twentieth": "20",
}


DATE_WORDS_TRANSLATION = {
    **{  # all in lemma form
        "monday": "poniedziałek",
        "tuesday": "wtorek",
        "wednesday": "środa",
        "wednesdays": "środy",
        "thursday": "czwartek",
        "friday": "piątek",
        "saturday": "sobota",
        "sunday": "niedziela",
        "sundays": "niedziele",
        "january": "styczeń",
        "february": "luty",
        "march": "marzec",
        "april": "kwiecień",
        "may": "maj",
        "june": "czerwiec",
        "july": "lipiec",
        "august": "sierpień",
        "september": "wrzesień",
        "october": "październik",
        "oct.": "październik",
        "november": "listopad",
        "nov.": "listopad",
        "december": "grudzień",
        "century": "wiek",
        "winter": "zima",
        "spring": "wiosna",
        "summer": "lato",
        "autumn": "jesień",
        "fall": "jesień",
        "weekend": "weekend",
        "christmas": "Boże Narodzenie",
        "to": "do",
        "and": "i",
        "in": "w",
        "-": "-",
        "through": "-",
    },
    **ORDINAL_WORDS_TRANSLATION  # add ordinal words translation -
    # in polish we usually write "1 listopad" not "pierwszy listopad"
}


POLISH_MONTHS_DATE_FORM = {
    "styczeń": "stycznia",
    "luty": "lutego",
    "marzec": "marca",
    "kwiecień": "kwietnia",
    "maj": "maja",
    "czerwiec": "czerwca",
    "lipiec": "lipca",
    "sierpień": "sierpnia",
    "wrzesień": "września",
    "październik": "października",
    "listopad": "listopada",
    "grudzień": "grudnia",
}


POLISH_MONTHS = {
    "styczeń",
    "luty",
    "marzec",
    "kwiecień",
    "maj",
    "czerwiec",
    "lipiec",
    "sierpień",
    "wrzesień",
    "październik",
    "listopad",
    "grudzień",
}


def is_month(word):
    return word in POLISH_MONTHS


def date_word_translation(word, month_date_form=False):
    translation = DATE_WORDS_TRANSLATION[word]
    if not is_month(translation) or not month_date_form:
        return translation

    return POLISH_MONTHS_DATE_FORM[translation]


DURATION_WORDS_TRANSLATION = {
    "years": "lat",
    "year": "rok",
    "minutes": "minut",
    "seconds": "sekundy",
    "second": "sekunda",
    "decades": "dekad",
    "hours": "godzin",
    "days": "dni",
    "months": "miesięcy",
    "weeks": "tygodnie",
    "week": "tydzień",
    "day": "dzień",
    "hour": "godzina",
}


NUMBER_WORDS_VALUES = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
    "hundred": 100,
    "thousand": 1000,
    "million": 1000000,
    "b": 1000000000,
    "bn": 1000000000,
    "trillion": 1000000000000,
}


def number_word_to_int(text: str) -> int:
    parts = [word for word in text.replace("-", " ").lower().split(" ")]
    try:
        if len(parts) == 1:
            return NUMBER_WORDS_VALUES[parts[0]]
        if len(parts) == 2:
            if parts[1] == "hundred":
                return NUMBER_WORDS_VALUES[parts[0]] * 100
            if parts[1] == "thousand":
                return NUMBER_WORDS_VALUES[parts[0]] * 1000
            if parts[1] == "million":
                return NUMBER_WORDS_VALUES[parts[0]] * 1000000

            return NUMBER_WORDS_VALUES[parts[0]] + NUMBER_WORDS_VALUES[parts[1]]
        if len(parts) == 3:
            if parts[1] == "hundred":
                return NUMBER_WORDS_VALUES[parts[0]] * 100 + NUMBER_WORDS_VALUES[parts[2]]
            if NUMBER_WORDS_VALUES[parts[0]] < 20:
                # "nineteen eighty four" case
                return (
                    NUMBER_WORDS_VALUES[parts[0]] * 100 + NUMBER_WORDS_VALUES[parts[1]] + NUMBER_WORDS_VALUES[parts[2]]
                )
            raise ValueError(f"Unexpected value {text!r}")
        if "and" in text:
            # "one hundred and fifty-seven" case
            left, right = text.split("and")
            return number_word_to_int(left.strip()) + number_word_to_int(right.strip())

    except KeyError:
        raise ValueError(f"Unexpected value {text!r}")
    raise ValueError("Expected either one, two or three words number.")


def number_to_text_number(number: Union[float, int]) -> Set[str]:
    number = int(number) if isinstance(number, float) and number.is_integer() else number
    # use polish lemma form of "milion"
    # all possible polish forms here: "miliona", "miliony", "milionów"
    # have the same lemma form "milion" using `squad_pl.doc2vec.preprocess.polish_lemmatization`function
    # (same reasoning for "tysiąc", "miliard")
    text_number = None
    if 1000 <= number < 1000000 and number % 100 == 0:
        text_number = f"{number / 1000} tysiąc"

        # create another version with word instead of a number
        if number % 1000 == 0:
            another_text_number = None
            thousand_number = number // 1000
            if thousand_number in POLISH_NUMBERS:
                another_text_number = f"{POLISH_NUMBERS[thousand_number]} tysiąc"
            elif thousand_number < 100:
                another_text_number = (
                    f"{POLISH_NUMBERS[(thousand_number // 10) * 10]} {POLISH_NUMBERS[thousand_number % 10]} tysiąc"
                )
            if another_text_number:
                return {
                    text_number.replace(".", ",").replace(",0", ""),
                    another_text_number.replace(".", ",").replace(",0", ""),
                }

    if 1000000 <= number < 1000000000 and number % 100000 == 0:
        text_number = f"{number / 1000000} milion"

    if 1000000000 <= number < 1000000000000 and number % 100000000 == 0:
        text_number = f"{number / 1000000000} miliard"

    if 1000000000000 <= number < 1000000000000000 and number % 100000000000 == 0:
        text_number = f"{number / 1000000000000} trylion"

    if isinstance(number, int):
        if number in POLISH_NUMBERS:
            return {POLISH_NUMBERS[number]}
        elif number < 100:
            return {f"{POLISH_NUMBERS[(number // 10) * 10]} {POLISH_NUMBERS[number % 10]}"}

    if text_number:
        return {text_number.replace(".", ",").replace(",0", "")}

    return set()


LETTER_ONLY_REGEX = r"[A-Za-z ]+"


def is_letter_only(word):
    return bool(re.fullmatch(LETTER_ONLY_REGEX, word))


ROMAN_NUMERAL_REGEX = r"^(I|V|X|L|C|D|M)+$"


def is_roman_numeral(word):
    return bool(re.fullmatch(ROMAN_NUMERAL_REGEX, word))


TITLES_TRANSLATION_FILENAME = DATA_DIR / "wikipedia/raw/pl/titles_pl_en.txt"


CURRENCIES = {
    "us$": "dolar",
    "£": "funt",
    "€": "euro",
    "¥": "juan",
    "₹": "rupia",
    "euros": "euro",
    "us dollars": "dolar",
    "USD": "dolar",
    "dollars": "dolar",
    "pounds": "funt",
    "pesos": "pesos",
    "yuan": "juan",
    "francs": "franków",
    "swiss francs": "franków",
    "Swiss francs": "franków",
    "cents": "cent",
    "reais": "reais",  # real brazylijski
    "nt$": "dolar",  # dolar tajwański
    "s$": "dolar",  # dolar singapurski
    # invalid "currencies"
    "#": "",
    "tonnes": "ton",
    "litres": "litr",
    "barrels": "baryłka",
    "hectares": "hektar",
    "gallons": "galon",
}


def split_from_currency(text: str) -> Tuple[str, str, Optional[bool]]:
    # return triple: (currency sign, rest of the text, is_currency_sign_at_end)
    text = text.lower()

    for currency in CURRENCIES.keys():
        if text[: len(currency)] == currency:
            return CURRENCIES[currency], text[len(currency) :].lstrip(), False
        if text[-len(currency) :] == currency:
            return CURRENCIES[currency], text[: -len(currency)].rstrip(), True
    return "", text, None


def load_wiki_titles_translation() -> Dict[str, str]:
    """
    Load wiki titles translation from a file (taken from wikipedia langlinks file using wikipedia-parallel-titles tool).
    """
    titles_en_pl = {}  # english title -> polish title
    with open(TITLES_TRANSLATION_FILENAME, "r") as titles_translation_file:
        for line in titles_translation_file.readlines():
            title_pl, title_en = [title.strip() for title in line.split("|||")]
            titles_en_pl[title_en] = title_pl
    return titles_en_pl


def is_sublist(l1, l2, pos=False):
    """
    Check if l1 is a sublist of l2 (it contains all elements from l1 in the same order).

    if pos is True, return position where sublist starts, not a boolean value
    """
    if l1 == []:
        return 0 if pos else True
    elif l1 == l2:
        return 0 if pos else True
    elif len(l1) > len(l2):
        return -1 if pos else False

    for i in range(len(l2)):
        if l2[i] == l1[0]:
            n = 1
            while (n < len(l1)) and (i + n < len(l2)) and (l2[i + n] == l1[n]):
                n += 1

            if n == len(l1):
                return i if pos else True
    return -1 if pos else False
