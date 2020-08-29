import json
import re
from collections import defaultdict
from typing import List

from squad_pl import DATA_DIR
from squad_pl.doc2vec.preprocess import load_squad_wikipedia, Tag
from squad_pl.doc2vec.utils import remove_headers, H_REG, split_article_into_sentences, polish_nlp
from squad_pl.generate_squad.extend_squad import extend_questions_answers
from squad_pl.generate_squad.process_similar_wiki_sentences import (
    FILTERED_CORRECT_SIMILAR_WIKI_SENTENCES_PATH,
    process_wiki_similar_sentences,
)


def split_article_into_paragraphs(text) -> List[str]:
    """Take a wikipedia article and return all its paragraphs."""

    # handle special case of a list header in an article
    to_replace = []
    for header in re.finditer(H_REG, text):
        if text[header.end() : header.end() + 10].lstrip()[:2] == "* ":
            to_replace.append((text[header.start() : header.end()], header.group("header").strip() + ": "))
    for header, header_text in to_replace:
        text = text.replace(header, header_text)

    text = remove_headers(text)
    text = re.sub(r"(\n){2,}", "***paragraph-break***", text)

    paragraphs = [paragraph for paragraph in text.split("***paragraph-break***") if paragraph.strip()]
    result = []
    # handle special case of a list in an article - this should be one paragraph with header as a first sentence
    i = 0
    while i < len(paragraphs):
        if i < len(paragraphs) - 1 and paragraphs[i].rstrip()[-1:] == ":" and paragraphs[i + 1][:2].lstrip()[:1] == "*":
            # merge list elements with a first paragraph (that might be old header or just a paragraph)
            list_paragraph = [paragraphs[i]]
            i += 1
            while i < len(paragraphs) and paragraphs[i][:2].lstrip()[:1] == "*":
                list_paragraph.append(paragraphs[i])
                i += 1
            result.append(" ".join(list_paragraph))
            continue

        result.append(paragraphs[i])
        i += 1

    # remove all newlines and extra spaces from text
    result = [re.sub(r"\n", " ", paragraph).strip() for paragraph in result]
    result = [re.sub(r"( ){2,}", " ", paragraph) for paragraph in result]
    return result


def get_answer_position(paragraph, sentence, answer):
    """Get answer start in a paragraph."""
    sentence_lower = sentence.lower()
    answer_lower = answer.lower()

    # Calculating an answer position in a whole paragraph
    # Getting an answer position for sentence only
    # cannot be done with just a simple:
    # answer_pos_in_sentence = sentence_lower.index(answer_lower)
    # because the answer might be a substring of some word, e.g. "3" is in "1793"
    # that's why separate tokens need to be checked

    if " " in answer or sentence_lower.count(answer_lower) == 1:
        # if answer has more than one word or there is only one such a substring in a sentence
        # then position can be simply found
        answer_pos_in_sentence = sentence_lower.index(answer_lower)
    else:
        sentence_tokens = [token.text.lower() for token in polish_nlp(sentence)]
        real_answer_number = 0
        for token in sentence_tokens:
            if token == answer_lower:
                break
            if answer_lower in token:
                real_answer_number += 1

        answer_pos_in_sentence = sentence_lower.index(answer_lower)
        for i in range(real_answer_number):
            answer_pos_in_sentence = sentence_lower.index(answer_lower, answer_pos_in_sentence + len(answer))

    # REMOVE THIS LINE FOR FINAL PROCESSING AND UNCOMMENT CODE ABOVE
    # answer_pos_in_sentence = sentence_lower.index(answer.lower())

    try:
        sentence_pos_in_paragraph = paragraph.index(sentence)
    except ValueError:
        # sentence is in fact 2 merged sentences
        # rare case in which sentence wasn't processed correctly because sentences with less than 3 tokens
        # were rejected (which was a bad idea but didn't have much impact on a result)
        sentence_pos_in_paragraph = paragraph.lower().index(split_article_into_sentences(sentence)[0].lower())

    answer_pos = sentence_pos_in_paragraph + answer_pos_in_sentence
    return answer_pos


def generate_squad_from_similar_sentences(data, manual):

    page_ids = set()
    for question_answer in data:
        page_ids.add(str(Tag.from_str(question_answer["tag"]).page_id))

    articles = load_squad_wikipedia(page_ids)
    paragraphs = {page_id: split_article_into_paragraphs(articles[page_id]) for page_id in page_ids}

    result = []
    incorrect_sentences = 0
    different_answers = 0
    for question_answer in data:
        question = question_answer["question"]
        answer = question_answer["answer"]

        if not manual and answer.lower() != question_answer["original_answer"].lower():
            # case in which the answer wasn't an exact match
            # but a lemmatized one (manual dataset was based on exact matches only - no need to check)
            original_answer = question_answer["original_answer"]

            # check that real and original answer are the same when lemmatized
            answer_lemma = [token.lemma_.lower() for token in polish_nlp(answer)]
            original_answer_lemma = [token.lemma_.lower() for token in polish_nlp(original_answer)]
            if answer_lemma != original_answer_lemma:
                if len(answer_lemma) != len(original_answer_lemma):
                    print(
                        f"Skipping because lemmatized answers are different: "
                        f"{answer_lemma} != {original_answer_lemma}"
                    )
                    different_answers += 1
                    continue

                skip = False
                for i in range(len(answer_lemma)):
                    if answer_lemma[i] != original_answer_lemma:
                        # lemmatizer could fail but words are still equal
                        # (particular words have the same base form - check the beginning of each word)
                        if answer_lemma[i][:3] != original_answer_lemma[i][:3]:
                            if len(answer_lemma[i]) <= 4:
                                if answer_lemma[i][:2] != original_answer_lemma[i][:2]:
                                    print(
                                        f"Skipping because lemmatized answers are different: "
                                        f"{answer_lemma} != {original_answer_lemma}"
                                    )
                                    skip = True
                                    break

                if skip:
                    continue

        # "sentence" might be actually 2 sentences
        sentence = question_answer["sentence"]
        tag = Tag.from_str(question_answer["tag"])
        answer_ne = question_answer["answer_ner"]

        sentence = re.sub(r"( ){2,}", " ", sentence)
        sentence_paragraph = None
        found = False

        for paragraph in paragraphs[str(tag.page_id)]:
            if sentence in paragraph:
                sentence_paragraph = paragraph
                found = True
                break

        if not found:
            # not found
            # sentence might be in 2 sentences from different paragraphs
            # try again merging adjacent paragraphs into one paragraph
            for i in range(len(paragraphs[str(tag.page_id)]) - 1):
                paragraph = paragraphs[str(tag.page_id)][i] + " " + paragraphs[str(tag.page_id)][i + 1]
                if sentence in paragraph:
                    sentence_paragraph = paragraph
                    found = True
                    break

        if not found:
            print(f"Incorrect sentence: {sentence}")
            incorrect_sentences += 1
            continue

        assert sentence_paragraph
        answer_pos = get_answer_position(sentence_paragraph, sentence, answer)
        assert sentence_paragraph[answer_pos : answer_pos + len(answer)].lower() == answer.lower()
        answer = sentence_paragraph[answer_pos : answer_pos + len(answer)]

        result.append(
            {
                # "_(sentence_1)" or "_(sentence_2)" are at the end of
                # every title when join=True
                "title": tag.title.replace(" ", "_").replace("_(sentence_1)", "").replace("_(sentence_2)", ""),
                "context": sentence_paragraph,
                "question": question,
                "answer": answer,
                "answer_start": answer_pos,
                "page_id": tag.page_id,
                "sentence": sentence,  # for later processing
                "answer_ne": answer_ne,  # for later processing
                "similarity": question_answer.get("similarity", -1),  # for later processing
            }
        )

    print(f"Incorrect sentences: {incorrect_sentences}")
    print(f"Different answers: {different_answers}")
    return result


def generate_squad(manual, depth, join):
    """Generate polish SQuAD with report."""
    if manual:
        # if we want exact (manual) question answers then use manually processed data
        generated_from_sentences = []
        for sentence_num in (1, 2):
            correct_sentences_path = FILTERED_CORRECT_SIMILAR_WIKI_SENTENCES_PATH.format(0, sentence_num)
            with open(correct_sentences_path, "r") as correct_sentences_file:
                data = json.load(correct_sentences_file)
            generated_from_sentences += generate_squad_from_similar_sentences(data, manual)
    else:
        # use automatically processed data otherwise
        data = process_wiki_similar_sentences(depth, join)
        generated_from_sentences = generate_squad_from_similar_sentences(data, manual)

    grouped = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for i, item in enumerate(generated_from_sentences):
        grouped[item["title"]][item["context"]][item["question"]].append(item)

    polish_squad = []
    report = {"all": defaultdict(int), "original": defaultdict(int), "extended": defaultdict(int)}

    question_id = 1
    for title, paragraphs in grouped.items():
        title_elem = {"title": title, "paragraphs": []}
        for context, questions in paragraphs.items():
            paragraph_elem = {"context": context, "qas": []}
            extended_questions = extend_questions_answers(questions, manual)
            for question, question_answers in extended_questions.items():
                question_elem = {"question": question, "answers": [], "id": question_id}
                question_id += 1
                answers = set()
                for question_answer in question_answers:
                    if question_answer["answer"] not in answers:
                        # remove duplicated answers
                        answers.add(question_answer["answer"])
                        answer = question_answer["answer"]
                        answer_start = question_answer["answer_start"]
                        question_elem["answers"].append(
                            {
                                "text": answer,
                                "answer_start": answer_start,
                                # "answer_ne": question_answer["answer_ne"],
                                # "sentence_nes": question_answer["sentence_nes"]
                            }
                        )
                        if question_answer.get("extended"):
                            report["extended"]["ALL_CATEGORIES"] += 1
                            report["extended"][question_answer["answer_ne"]] += 1
                        else:
                            report["original"]["ALL_CATEGORIES"] += 1
                            report["original"][question_answer["answer_ne"]] += 1

                        report["all"]["ALL_CATEGORIES"] += 1
                        report["all"][question_answer["answer_ne"]] += 1

                        # check it just to be sure
                        assert context[answer_start : answer_start + len(answer)] == answer

                        # if question_answer.get("extended") and question_answer["answer_ne"] == "PERSON":
                        #     print()
                        #     print(question)
                        #     print(question_answer["answer"])
                        #     print()

                paragraph_elem["qas"].append(question_elem)
            title_elem["paragraphs"].append(paragraph_elem)
        polish_squad.append(title_elem)

    # transform the result into the form in which every question has exactly one answer
    # unfortunately this is needed because SQuAD train dataset must comply with this rule
    result = []
    question_id = 1
    for article in polish_squad:
        new_article = {"title": article["title"], "paragraphs": []}
        for paragraph in article["paragraphs"]:
            new_paragraph = {"context": paragraph["context"], "qas": []}
            for question_answers in paragraph["qas"]:
                for answer in question_answers["answers"]:
                    new_paragraph["qas"].append(
                        {
                            "question": question_answers["question"],
                            "answers": [{"text": answer["text"], "answer_start": answer["answer_start"]}],
                            "id": str(question_id),
                        }
                    )
                    question_id += 1
            new_article["paragraphs"].append(new_paragraph)
        result.append(new_article)

    return {"data": result}, report


POLISH_SQUAD_MANUAL_PATH = DATA_DIR / "squad/pl/final/polaq_dataset_manual.json"
POLISH_SQUAD_REPORT_MANUAL_PATH = DATA_DIR / "squad/pl/final/polaq_dataset_manual_report.txt"

POLISH_SQUAD_GENERATED_JOINED_PATH = str(DATA_DIR / "squad/pl/final/polaq_dataset_generated_joined_depth_{}.json")
POLISH_SQUAD_GENERATED_JOINED_REPORT_PATH = str(
    DATA_DIR / "squad/pl/final/polaq_dataset_generated_joined_depth_{}_report.txt"
)

POLISH_SQUAD_GENERATED_SEPARATE_PATH = str(DATA_DIR / "squad/pl/final/polaq_dataset_generated_separate_depth_{}.json")
POLISH_SQUAD_GENERATED_SEPARATE_REPORT_PATH = str(
    DATA_DIR / "squad/pl/final/polaq_dataset_generated_separate_depth_{}_report.txt"
)


def generate_squad_and_report(manual, depth=0, join=False):
    polish_squad, report = generate_squad(manual, depth, join)
    if manual:
        squad_file = POLISH_SQUAD_MANUAL_PATH
        squad_report_file = POLISH_SQUAD_REPORT_MANUAL_PATH
    elif join:
        squad_file = POLISH_SQUAD_GENERATED_JOINED_PATH.format(depth)
        squad_report_file = POLISH_SQUAD_GENERATED_JOINED_REPORT_PATH.format(depth)
    else:
        squad_file = POLISH_SQUAD_GENERATED_SEPARATE_PATH.format(depth)
        squad_report_file = POLISH_SQUAD_GENERATED_SEPARATE_REPORT_PATH.format(depth)

    with open(squad_file, "w") as polish_squad_file:
        json.dump(polish_squad, polish_squad_file, indent=2)
    with open(squad_report_file, "w") as polish_squad_report_file:
        for type, categories in report.items():
            polish_squad_report_file.write(f"{type.upper()}:\n")
            for category, count in sorted(categories.items(), key=lambda item: item[1], reverse=True):
                polish_squad_report_file.write(f"    {category}: {count}\n")
            polish_squad_report_file.write("\n\n")


if __name__ == "__main__":
    # generate_squad_and_report(manual=True)
    # generate_squad_and_report(manual=False, depth=0, join=False)
    # generate_squad_and_report(manual=False, depth=1, join=False)
    # generate_squad_and_report(manual=False, depth=0, join=True)
    generate_squad_and_report(manual=False, depth=1, join=True)
