import json
import os
import pickle
from time import sleep

from squad_pl import DATA_DIR, logger, SQUAD_PATH
from squad_pl.translate.utils import (
    translate_googletrans,
    GOOGLETRANS_REQUEST_LENGTH_LIMIT,
    translate_gcp,
    GCP_REQUEST_LENGTH_LIMIT,
)

TRANSLATED_QUESTIONS_PATH = DATA_DIR / f"squad/translated/translated_questions"


def translate_questions(
    squad_path,
    translated_questions_path,
    translate=translate_googletrans,
    length_limit=GOOGLETRANS_REQUEST_LENGTH_LIMIT,
    wait=60,
):
    """Translate SQuAD questions using provided translator."""
    with open(squad_path, "r") as squad_file:
        squad = json.load(squad_file)["data"]
        questions_to_translate = []
        question_ids = []
        questions_length = 0

        if os.path.exists(translated_questions_path):
            with open(translated_questions_path, "rb") as translated_questions_file:
                already_translated_question_ids = set(pickle.load(translated_questions_file).keys())
        else:
            already_translated_question_ids = set()

        logger.info(f"Starting with {len(already_translated_question_ids)} already translated questions.")

        for article in squad:
            for paragraph in article["paragraphs"]:
                for question_answers in paragraph["qas"]:
                    question_id = question_answers["id"]
                    if question_id in already_translated_question_ids:
                        continue
                    question = question_answers["question"].strip()
                    questions_to_translate.append(question)
                    question_ids.append(question_id)
                    questions_length += len(question)

                if questions_length >= length_limit:
                    if os.path.exists(translated_questions_path):
                        with open(translated_questions_path, "rb") as translated_questions_file:
                            translated_questions = pickle.load(translated_questions_file)
                    else:
                        translated_questions = {}

                    result = translate(questions_to_translate)

                    for i, id in enumerate(question_ids):
                        translated_questions[id] = result[i]

                    with open(translated_questions_path, "wb") as translated_questions_file:
                        pickle.dump(translated_questions, translated_questions_file)

                    logger.info(f"Translated {len(questions_to_translate)} questions in a request")
                    logger.info(f"Translated {len(translated_questions.keys())} in total")
                    questions_to_translate = []
                    question_ids = []
                    questions_length = 0
                    logger.info(f"Wait {wait} seconds before next request")
                    sleep(wait)

        #  iteration is over but there might be some questions left to translate
        if questions_to_translate:
            with open(translated_questions_path, "rb") as translated_questions_file:
                translated_questions = pickle.load(translated_questions_file)

            result = translate(questions_to_translate)

            for i, id in enumerate(question_ids):
                translated_questions[id] = result[i]

            with open(translated_questions_path, "wb") as translated_questions_file:
                pickle.dump(translated_questions, translated_questions_file)

            logger.info(f"Translated {len(questions_to_translate)} questions in a request")
            logger.info(f"Translated {len(translated_questions.keys())} in total")


def load_translated_questions():
    with open(TRANSLATED_QUESTIONS_PATH, "rb") as translated_questions_file:
        return pickle.load(translated_questions_file)


# translate_questions(
#     SQUAD_PATH,
#     TRANSLATED_QUESTIONS_PATH,
#     translate=translate_gcp,
#     length_limit=GCP_REQUEST_LENGTH_LIMIT,
#     wait=101
# )
