import json

from squad_pl import DATA_DIR

VERSION1_1 = "1.1"
VERSION2_0 = "2.0"


def join_squad_files(version):
    dev_squad_json_filename = DATA_DIR / f"squad/raw/dev-v{version}.json"
    train_squad_json_filename = DATA_DIR / f"squad/raw/train-v{version}.json"
    joined_squad_json_filename = DATA_DIR / f"squad/raw/squad_full_dev_and_train-v{version}.json"

    with open(joined_squad_json_filename, "w+", encoding="utf8") as joined_squad_json_file, open(
        dev_squad_json_filename, "r", encoding="utf8"
    ) as dev_squad_json_file, open(train_squad_json_filename, "r", encoding="utf8") as train_squad_json_file:
        dev_squad_data = json.load(dev_squad_json_file)["data"]
        train_squad_data = json.load(train_squad_json_file)["data"]
        joined_squad_data = {"data": dev_squad_data + train_squad_data, "version": version}

        json.dump(joined_squad_data, joined_squad_json_file, indent=4)


def join_both_squad_versions():
    squad_1_1_json_filename = DATA_DIR / f"squad/raw/squad_full_dev_and_train-v{VERSION1_1}.json"
    squad_2_0_json_filename = DATA_DIR / f"squad/raw/squad_full_dev_and_train-v{VERSION2_0}.json"
    joined_squad_json_filename = DATA_DIR / f"squad/raw/squad_full_dev_and_train.json"

    with open(joined_squad_json_filename, "w+", encoding="utf8") as joined_squad_json_file, open(
        squad_1_1_json_filename, "r", encoding="utf8"
    ) as squad_1_1_json_file, open(squad_2_0_json_filename, "r", encoding="utf8") as squad_2_0_json_file:
        squad_1_1_data = {}
        for article in json.load(squad_1_1_json_file)["data"]:
            title = article["title"]
            squad_1_1_data[title] = {}
            for paragraph in article["paragraphs"]:
                context = paragraph["context"]
                if context not in squad_1_1_data[title]:  # context sometimes repeats so don't always set to {}
                    squad_1_1_data[title][context] = {}
                for question_answers in paragraph["qas"]:
                    question_answers["is_impossible"] = False  # SQuAD 1.1 doesn't have this property
                    squad_1_1_data[title][context][question_answers["id"]] = question_answers

        squad_full_data = squad_1_1_data
        new_question_answers = []
        common_qas_count = 0
        for article in json.load(squad_2_0_json_file)["data"]:
            title = article["title"]
            assert title in squad_1_1_data  # every title from SQuAD 2.0 is in SQuAD 1.1 somehow
            for paragraph in article["paragraphs"]:
                context = paragraph["context"]
                if context not in squad_1_1_data[title]:
                    squad_full_data[title][context] = {}
                for question_answers in paragraph["qas"]:
                    if question_answers["id"] not in squad_full_data[title][context]:
                        new_question_answers.append(question_answers)
                    else:
                        common_qas_count += 1
                        # check that SQuAD 2.0 doesn't change the old ones
                        assert squad_full_data[title][context][question_answers["id"]] == question_answers
                    squad_full_data[title][context][question_answers["id"]] = question_answers

        data = {"data": []}
        for title, contexts in squad_full_data.items():
            paragraphs = {"title": title, "paragraphs": []}
            for context, questions_answers in contexts.items():
                qas = {"context": context, "qas": []}
                for id, question_answers in questions_answers.items():
                    qas["qas"].append(question_answers)
                paragraphs["paragraphs"].append(qas)
            data["data"].append(paragraphs)

        json.dump(data, joined_squad_json_file, indent=4)


def analyze_full_squad_dataset():
    """Analyze full SQuAD dataset (joined both versions)."""
    squad_1_1_json_filename = DATA_DIR / f"squad/raw/squad_full_dev_and_train-v{VERSION1_1}.json"
    squad_2_0_json_filename = DATA_DIR / f"squad/raw/squad_full_dev_and_train-v{VERSION2_0}.json"

    all_qas_count = 0
    with open(squad_1_1_json_filename, "r", encoding="utf8") as squad_1_1_json_file, open(
        squad_2_0_json_filename, "r", encoding="utf8"
    ) as squad_2_0_json_file:
        squad_1_1_data = {}
        for article in json.load(squad_1_1_json_file)["data"]:
            title = article["title"]
            squad_1_1_data[title] = {}
            for paragraph in article["paragraphs"]:
                context = paragraph["context"]
                if context not in squad_1_1_data[title]:  # context sometimes repeats so don't always set to {}
                    squad_1_1_data[title][context] = {}
                for question_answers in paragraph["qas"]:
                    question_answers["is_impossible"] = False  # SQuAD 1.1 doesn't have this property
                    squad_1_1_data[title][context][question_answers["id"]] = question_answers
                    all_qas_count += 1

        squad_full_data = squad_1_1_data
        new_question_answers = []
        common_qas_count = 0
        for article in json.load(squad_2_0_json_file)["data"]:
            title = article["title"]
            assert title in squad_1_1_data  # every title from SQuAD 2.0 is in SQuAD 1.1 somehow
            for paragraph in article["paragraphs"]:
                context = paragraph["context"]
                if context not in squad_1_1_data[title]:
                    squad_full_data[title][context] = {}
                for question_answers in paragraph["qas"]:
                    if question_answers["id"] not in squad_full_data[title][context]:
                        new_question_answers.append(question_answers)
                        all_qas_count += 1
                    else:
                        common_qas_count += 1
                        # check that SQuAD 2.0 doesn't change the old ones
                        assert squad_full_data[title][context][question_answers["id"]] == question_answers
                    squad_full_data[title][context][question_answers["id"]] = question_answers

        new_possible_qas_count = 0
        new_impossible_qas_count = 0
        for question_answers in new_question_answers:
            if question_answers["is_impossible"]:
                new_impossible_qas_count += 1
            else:
                new_possible_qas_count += 1

        print(f"All questions: {all_qas_count}")
        print(f"Common questions: {common_qas_count}")
        print(f"New possible questions: {new_possible_qas_count}")
        print(f"New impossible questions: {new_impossible_qas_count}")


if __name__ == "__main__":
    analyze_full_squad_dataset()
