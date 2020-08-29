import argparse
import json
import os
from concurrent.futures.thread import ThreadPoolExecutor
from logging import getLogger

from stanfordnlp.protobuf import Document
from stanfordnlp.server import CoreNLPClient
from google.protobuf.pyext._message import SetAllowOversizeProtos

from squad_pl import CORENLP_HOME, DATA_DIR
from squad_pl.proto.dataset_pb2 import Article, Paragraph, QuestionAnswer
from squad_pl.proto.io import write_article
from squad_pl.utils import get_annotated_article_titles

logger = getLogger()

annotated_squad_data_filename = os.path.join(DATA_DIR, "squad/annotated/squad_data_core_nlp")


test_paragraph = (
    "Super Bowl 50 was an American football game to determine the champion of the National "
    "Football League (NFL) for the 2015 season. The American Football Conference (AFC) "
    "champion Denver Broncos defeated the National Football Conference (NFC) champion Carolina "
    "Panthers 24\342\200\22310 to earn their third Super Bowl title. The game was played on February 7, "
    "2016, at Levi's Stadium in the San Francisco Bay Area at Santa Clara, California. As this was the "
    '50th Super Bowl, the league emphasized the "golden anniversary" with various gold-themed '
    "initiatives, as well as temporarily suspending the tradition of naming each Super Bowl game "
    'with Roman numerals (under which the game would have been known as "Super Bowl L"), '
    "so that the logo could prominently feature the Arabic numerals 50."
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-json", required=True, help="Path to the dataset in JSON format.")
    parser.add_argument(
        "--output-proto", required=True, help="Where to output the annotated dataset as a serialized protocol buffer.",
    )
    parser.add_argument("--ram", default="8G", help="Amount of RAM available for running the CoreNLP " "server.")
    parser.add_argument("--threads", default="8", type=int, help="Number of threads CoreNLP will be run on.")
    parser.add_argument("--corenlp-home", help="Location of your CoreNLP.")
    args = parser.parse_args()

    if args.corenlp_home:
        os.environ["CORENLP_HOME"] = args.corenlp_home
    else:
        os.environ["CORENLP_HOME"] = str(CORENLP_HOME)

    # Fix for: https://github.com/stanfordnlp/stanfordnlp/issues/154
    SetAllowOversizeProtos(True)

    annotated_article_titles = get_annotated_article_titles(args.output_proto)

    with CoreNLPClient(
        annotators=[
            "tokenize",
            "ssplit",
            "pos",
            "lemma",
            "ner",
            "depparse",
            # "coref", - quote requires coref
            # "quote" - quote doesn't work, throws error
            "regexner",
            "openie",
            "natlog",
            "entitylink",
            "relation",
            "kbp",
            "tokensregex",
        ],
        timeout=1200000,
        memory=args.ram,
        threads=args.threads,
        input_format="text",
        output_format="serialized",
        be_quiet=True,
    ) as client:

        def annotate(text: str) -> Document:
            return client.annotate(text)

        def annotate_paragraph(input_paragraph, output_paragraph):
            paragraph_annotated: Document = annotate(input_paragraph["context"])
            output_paragraph.context.CopyFrom(paragraph_annotated)

            for input_qa in input_paragraph["qas"]:
                output_qa: QuestionAnswer = output_paragraph.qas.add()
                output_qa.id = input_qa["id"]
                question_annotated: Document = annotate(input_qa["question"])
                output_qa.question.CopyFrom(question_annotated)

                # answers in SQuAD are often the same - use cache
                answer_cache = {}

                for answer in input_qa["answers"]:
                    # Fix extra characters.
                    answer_text = answer["text"]
                    answer_offset = answer["answer_start"]

                    extra_chars = [" ", ".", ",", "!", ":", ";", "?", "`", "'", "$"]
                    while len(answer_text) > 0 and answer_text[-1] in extra_chars:
                        answer_text = answer_text[:-1]
                    while len(answer_text) > 0 and answer_text[0] in extra_chars:
                        answer_text = answer_text[1:]
                        answer_offset += 1

                    if answer_text not in answer_cache:
                        answer_cache[answer_text] = annotate(answer_text)
                    answer_annotated: Document = answer_cache[answer_text]
                    output_qa.answers.append(answer_annotated)
                    output_qa.answerOffsets.append(answer_offset)

        # load SQuAD data
        with open(args.input_json, "r") as f:
            input_data = json.loads(f.read())

        with open(args.output_proto, "ab+") as proto_output:
            articles = len(annotated_article_titles)
            for input_article in input_data:
                if input_article["title"] in annotated_article_titles:
                    continue

                logger.info(f"Annotating: {input_article['title']}")
                output_article: Article = Article()
                output_article.title = input_article["title"]
                output_article.en_wiki_page_id = input_article.get("en_wiki_page_id", None)
                output_article.pl_wiki_page_id = input_article.get("pl_wiki_page_id", None)

                with ThreadPoolExecutor(max_workers=args.threads) as executor:
                    thread_args = []
                    for input_paragraph in input_article["paragraphs"]:
                        output_paragraph: Paragraph = output_article.paragraphs.add()
                        thread_args.append((input_paragraph, output_paragraph))
                    try:
                        list(
                            executor.map(lambda args: annotate_paragraph(*args), thread_args)
                        )  #  eagerly compute generator with list(), wait for result
                    except Exception as ex:
                        logger.error(f"Error: {ex}. Skipping article.")
                        continue

                write_article(output_article, proto_output)
                articles += 1
                annotated_article_titles.add(output_article.title)
                logger.info(f"Annotated {articles}/{len(input_data)} articles")
