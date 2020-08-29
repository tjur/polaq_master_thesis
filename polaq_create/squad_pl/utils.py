import os
import pickle
import resource
import sys
from typing import List, Set

from pyplwnxml import PlwnxmlParser

from squad_pl import DATA_DIR, PLWORDNET_PATH
from squad_pl.proto.io import read_article


def get_squad_data(squad_data_filename=os.path.join(DATA_DIR, "squad/processed/squad_data")) -> List[dict]:
    "Open pickled squad data and return it."
    with open(squad_data_filename, "rb") as squad_data_file:
        squad_data = pickle.load(squad_data_file)
        return squad_data


def get_annotated_article_titles(filename) -> Set[str]:
    article_titles = []
    with open(filename, "rb") as f:
        while True:
            article = read_article(f)
            if article is None:
                return set(article_titles)
            article_titles.append(article.title)


def parse_plwordnet(plwordnet_path=PLWORDNET_PATH):
    """Parse polish wordnet xml file and pickle result."""
    max_rec = 100000

    # May segfault without this line. 0x100 is a guess at the size of each stack frame.
    resource.setrlimit(resource.RLIMIT_STACK, [100 * max_rec, resource.RLIM_INFINITY])
    sys.setrecursionlimit(max_rec)

    with open("wordnet", "wb") as wordnet_file:
        wordnet = PlwnxmlParser(plwordnet_path).read_wordnet()
        pickle.dump(wordnet, wordnet_file)
