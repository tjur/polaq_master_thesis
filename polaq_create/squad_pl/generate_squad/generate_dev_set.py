import json
from collections import defaultdict

from squad_pl import DATA_DIR

SQUAD_DEV_TEXT_PATH = DATA_DIR / "squad/pl/dev_set/polaq_dev_set.txt"
SQUAD_DEV_JSON_PATH = DATA_DIR / "squad/pl/final/polaq_dev_set.json"
SQUAD_DEV_REPORT_PATH = DATA_DIR / "squad/pl/final/polaq_dev_set_report.txt"


def read_lines_until_non_empty(file):
    while True:
        line = file.readline()
        if not line:
            raise EOFError
        line = line.strip()
        if line and line[0] not in ["#", "="]:
            return line


def generate_squad_dev_set_from_file(filename):
    """
    Generate SQuAD set from a specified, manually prepared file (with questions and answers in a specific format).
    """

    result = []
    report = defaultdict(int)
    counter = 1000001
    with open(filename, "r") as file:
        line = read_lines_until_non_empty(file)
        while True:
            assert line.startswith("T:")
            title = line[2:].strip()

            paragraphs = []
            line = read_lines_until_non_empty(file)
            while True:
                assert line.startswith("C:")
                context = line[2:].strip()
                assert len(context) >= 500

                qas = []
                line = read_lines_until_non_empty(file)
                while True:
                    assert line.startswith("Q:")
                    question = line[2:].strip()

                    line = file.readline()
                    assert line.startswith("A:")
                    answers = line[2:].strip()

                    line = file.readline()
                    assert line.startswith("S:")
                    sentence = line[2:].strip()

                    line = file.readline()
                    assert line.startswith("A_NE:")
                    ne = line[5:].strip()

                    question_answers = {"question": question, "answers": [], "id": str(counter)}
                    counter += 1
                    for answer in answers.split(";"):
                        answer = answer.strip()
                        answer_start = context.index(sentence) + sentence.index(answer)
                        assert context[answer_start : answer_start + len(answer)] == answer

                        question_answers["answers"].append({"text": answer, "answer_start": answer_start})
                    report[ne] += 1
                    report["ALL"] += 1

                    qas.append(question_answers)

                    try:
                        line = read_lines_until_non_empty(file)
                    except EOFError:
                        paragraphs.append({"context": context, "qas": qas})
                        result.append({"title": title, "paragraphs": paragraphs})
                        return {"data": result}, sorted(report.items(), key=lambda x: x[1], reverse=True)

                    if line.startswith("C:"):
                        paragraphs.append({"context": context, "qas": qas})
                        title_line = False
                        break
                    elif line.startswith("T:"):
                        paragraphs.append({"context": context, "qas": qas})
                        result.append({"title": title, "paragraphs": paragraphs})
                        title_line = True
                        break
                    else:
                        continue

                if title_line:
                    break


if __name__ == "__main__":
    squad_dev, report = generate_squad_dev_set_from_file(SQUAD_DEV_TEXT_PATH)
    with open(SQUAD_DEV_JSON_PATH, "w") as squad_dev_json_file:
        json.dump(squad_dev, squad_dev_json_file, indent=2)
    with open(SQUAD_DEV_REPORT_PATH, "w") as squad_dev_report_file:
        for category, count in report:
            squad_dev_report_file.write(f"{category}: {count}\n")
        squad_dev_report_file.write("\n")
    print("Generated polish SQuAD development json file")
    print(f"Report: {report}")
