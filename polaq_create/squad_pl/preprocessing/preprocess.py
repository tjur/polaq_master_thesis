import csv
import json
import queue
import pickle
from ast import literal_eval
from collections import defaultdict
from random import shuffle
from typing import Tuple, Generator, Optional, Set, Dict, Union, List, Iterable
from urllib.parse import unquote

from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence

from squad_pl import SQUAD_PATH, DATA_DIR

# these files were at first preprocessed with `add_new_lines_to_sql_values.sh` script
ENWIKI_PAGE_SQL_PATH = DATA_DIR / "wikipedia/raw/eng/enwiki-20200401-page.sql"
PLWIKI_PAGE_SQL_PATH = DATA_DIR / "wikipedia/raw/pl/plwiki-20200401-page.sql"
ENWIKI_LANGLINKS_SQL_PATH = DATA_DIR / "wikipedia/raw/eng/enwiki-20200401-langlinks.sql"

# preprocessed at first with `filter_wiki_pagelinks.sh` script
PLWIKI_PAGELINKS_SQL_PATH = DATA_DIR / "wikipedia/raw/pl/plwiki-20200401-pagelinks.sql"

ENWIKI_PAGE_CSV_PATH = DATA_DIR / "wikipedia/raw/eng/enwiki-page.csv"
PLWIKI_PAGE_CSV_PATH = DATA_DIR / "wikipedia/raw/pl/plwiki-page.csv"
ENWIKI_LANGLINKS_CSV_PATH = DATA_DIR / "wikipedia/raw/eng/enwiki-langlinks.csv"
PLWIKI_PAGELINKS_CSV_PATH = DATA_DIR / "wikipedia/raw/pl/plwiki-pagelinks.csv"
PLWIKI_PAGELINKS_DICT_PATH = DATA_DIR / "wikipedia/raw/pl/plwiki-pagelinks"
PLWIKI_PAGELINKS_ID_PAIRS_CSV = DATA_DIR / "wikipedia/raw/pl/plwiki-pagelinks-id-pairs.csv"

# this is PLWIKI_PAGELINKS_ID_PAIRS_CSV shuffled with shuf linux command
# shuffled data is better for training
PLWIKI_PAGELINKS_IDS_SHUFFLED_CSV = DATA_DIR / "wikipedia/raw/pl/pagelinks-ids.csv"
PLWIKI_PAGELINKS_IDS_SHUFFLED_FILE = DATA_DIR / "wikipedia/raw/pl/pagelinks-ids"


def extract_data_from_wiki_page_sql(wiki_page_sql_path, wiki_page_csv_output_path):
    """Leave only data for pages that are from wikipedia (namespace 0) and are not redirect."""
    insert_line = "INSERT INTO `page` VALUES\n"
    with open(wiki_page_sql_path, "r") as wiki_page_sql_file, open(
        wiki_page_csv_output_path, "w", newline=""
    ) as wiki_page_csv_output_file:
        csv_writer = csv.writer(wiki_page_csv_output_file, delimiter=" ", quoting=csv.QUOTE_MINIMAL)
        while True:
            line = wiki_page_sql_file.readline()
            if line == insert_line:
                break
        while True:
            line = wiki_page_sql_file.readline()
            if line == insert_line:
                continue
            try:
                value = literal_eval(line.replace("NULL", "None")[:-2])
                # https://www.mediawiki.org/wiki/Manual:Page_table
                # +--------------------+---------------------+------+-----+----------------+----------------+
                # | Field              | Type                | Null | Key | Default        | Extra          |
                # +--------------------+---------------------+------+-----+----------------+----------------+
                # | page_id            | int(10) unsigned    | NO   | PRI | NULL           | auto_increment |
                # | page_namespace     | int(11)             | NO   | MUL | NULL           |                |
                # | page_title         | varbinary(255)      | NO   |     | NULL           |                |
                # | page_restrictions  | tinyblob            | NO   |     | NULL           |                |
                # | page_is_redirect   | tinyint(3) unsigned | NO   | MUL | 0              |                |
                # | page_is_new        | tinyint(3) unsigned | NO   |     | 0              |                |
                # | page_random        | double unsigned     | NO   | MUL | NULL           |                |
                # | page_touched       | binary(14)          | NO   |     |                |                |
                # | page_links_updated | varbinary(14)       | YES  |     | NULL           |                |
                # | page_latest        | int(10) unsigned    | NO   |     | NULL           |                |
                # | page_len           | int(10) unsigned    | NO   | MUL | NULL           |                |
                # | page_content_model | varbinary(32)       | YES  |     | NULL           |                |
                # | page_lang          | varbinary(35)       | YES  |     | NULL           |                |
                # +--------------------+---------------------+------+-----+----------------+----------------+
                page_id, namespace, title, _, is_redirect, _, _, _, _, _, _, _, _ = value
                if is_redirect or namespace != 0:
                    # namespace 0 is what we want (wikipedia)
                    continue
                csv_writer.writerow([page_id, title])
            except SyntaxError:
                # all values were read
                break


# extract_data_from_wiki_page_sql(ENWIKI_PAGE_SQL_PATH, ENWIKI_PAGE_CSV_PATH)
# extract_data_from_wiki_page_sql(PLWIKI_PAGE_SQL_PATH, PLWIKI_PAGE_CSV_PATH)


def extract_data_from_wiki_langlinks_sql(wiki_langlinks_sql_path, wiki_langlinks_csv_output_path):
    """Leave only data for links redirecting into polish articles."""
    insert_line = "INSERT INTO `langlinks` VALUES\n"
    is_polish = False
    with open(wiki_langlinks_sql_path, "r") as wiki_langlinks_sql_file, open(
        wiki_langlinks_csv_output_path, "w", newline=""
    ) as wiki_langlinks_csv_output_file:
        csv_writer = csv.writer(wiki_langlinks_csv_output_file, delimiter=" ", quoting=csv.QUOTE_MINIMAL)
        while True:
            line = wiki_langlinks_sql_file.readline()
            if line == insert_line:
                break
        while True:
            try:
                line = wiki_langlinks_sql_file.readline()
            except UnicodeDecodeError:
                continue
            if line == insert_line:
                continue
            try:
                value = literal_eval(line[:-2])
                # https://www.mediawiki.org/wiki/Manual:Langlinks_table
                # +----------+------------------+------+-----+---------+-------+
                # | Field    | Type             | Null | Key | Default | Extra |
                # +----------+------------------+------+-----+---------+-------+
                # | ll_from  | int(10) unsigned | NO   | PRI | 0       |       |
                # | ll_lang  | varbinary(20)    | NO   | PRI |         |       |
                # | ll_title | varbinary(255)   | NO   |     |         |       |
                # +----------+------------------+------+-----+---------+-------+
                en_page_id, lang, title = value
                if lang != "pl":
                    if is_polish:
                        # polish rows finished
                        break
                    continue
                is_polish = True
                csv_writer.writerow([en_page_id, title])
            except SyntaxError:
                continue


# extract_data_from_wiki_langlinks_sql(ENWIKI_LANGLINKS_SQL_PATH, ENWIKI_LANGLINKS_CSV_PATH)


# there are 19 titles from SQuAD which corresponding articles have redirects
SQUAD_ENWIKI_REDIRECTS = {
    "Universal_Studios": "Universal_Pictures",
    "Cardinal_(Catholicism)": "Cardinal_(Catholic_Church)",
    "BeiDou_Navigation_Satellite_System": "BeiDou",
    "Sino-Tibetan_relations_during_the_Ming_dynasty": "Ming–Tibet_relations",
    "Sony_Music_Entertainment": "Sony_Music",
    "Multiracial_American": "Multiracial_Americans",
    "Imamah_(Shia_doctrine)": "Imamate_in_Shia_doctrine",
    "Videoconferencing": "Videotelephony",
    "Mary_(mother_of_Jesus)": "Mary,_mother_of_Jesus",
    "Swaziland": "Eswatini",
    "Gramophone_record": "Phonograph_record",
    "United_States_presidential_election,_2004": "2004_United_States_presidential_election",
    "Private_school": "Independent_school",
    "Modern_history": "History_of_the_world",
    "War_on_Terror": "War_on_terror",
    "Huguenot": "Huguenots",
    "Endangered_Species_Act": "Endangered_Species_Act_of_1973",
    "Sky_(United_Kingdom)": "Sky_UK",
    "Antibiotics": "Antibiotic",
}


def load_wiki_pages(wiki_page_csv_path, title_to_page_id=True):
    wiki_pages = {}
    with open(wiki_page_csv_path, "r", newline="") as wiki_page_csv_file:
        csv_reader = csv.reader(wiki_page_csv_file, delimiter=" ")
        for page_id, title in csv_reader:
            if title_to_page_id:
                wiki_pages[title] = int(page_id)
            else:
                wiki_pages[int(page_id)] = title
    return wiki_pages


def load_wiki_langlinks(wiki_langlinks_csv_path):
    langlinks = {}
    with open(wiki_langlinks_csv_path, "r", newline="") as wiki_langlinks_csv_file:
        csv_reader = csv.reader(wiki_langlinks_csv_file, delimiter=" ")
        for en_page_id, pl_title in csv_reader:
            langlinks[int(en_page_id)] = pl_title
    return langlinks


SQUAD_PLWIKI_CORRESPONDING_TITLES = {
    "Pharmaceutical_industry": "Farmacja",  # inaccurate
    "Computational_complexity_theory": "Złożoność_obliczeniowa",
    "Orthodox_Judaism": "Judaizm",
    "National_Archives_and_Records_Administration": "Biblioteki_prezydenckie_w_Stanach_Zjednoczonych",  # inaccurate
    "Copyright_infringement": "Prawo_autorskie",  # inaccurate
    "Military_history_of_the_United_States": "Historia_Stanów_Zjednoczonych",  # inaccurate
    "Nutrition": "Odżywianie",  # inaccurate
    "Videoconferencing": "Wideotelefonia",
    "Railway_electrification_system": "Przewody_trakcyjne",
    "Ministry_of_Defence_(United_Kingdom)": "Ministrowie_obrony_Wielkiej_Brytanii",  # inaccurate
    "Affirmative_action_in_the_United_States": "Akcja_afirmatywna",  # inaccurate
    "Communications_in_Somalia": "Somalia",  # inaccurate
    "Royal_assent": "Monarchia",  # inaccurate
    "Georgian_architecture": "Epoka_georgiańska",  # inaccurate
    "Umayyad_Caliphate": "Umajjadzi",
    "Party_leaders_of_the_United_States_House_of_Representatives": "Izba_Reprezentantów_Stanów_Zjednoczonych",  # inaccurate
    "Philosophy_of_space_and_time": "-",
    "Muslim_world": "Islam",  # inaccurate
    "List_of_numbered_streets_in_Manhattan": "Manhattan",  # inaccurate
    "Southern_California": "Kalifornia",
    "Black_people": "Czarna_odmiana_człowieka",  # inaccurate
    "Separation_of_powers_under_the_United_States_Constitution": "-",
    "Renewable_energy_commercialization": "Odnawialne_źródła_energii",
    "Race_and_ethnicity_in_the_United_States_Census": "-",
    "Separation_of_church_and_state_in_the_United_States": "Rozdział_Kościoła_od_państwa",  # inaccurate
    "Private_school": "-",
    "Hindu_philosophy": "Filozofia_indyjska",
    "BBC_Television": "BBC",
    "Multiracial_American": "-",
    "Comprehensive_school": "-",
    "Sino-Tibetan_relations_during_the_Ming_dynasty": "-",
    "Letter_case": "-",
    "Geological_history_of_Earth": "Historia_Ziemi",
}


# there are only few accurate (found manually in wikipedia)
SQUAD_PLWIKI_ACCURATE_CORRESPONDING_TITLES = {
    "Computational_complexity_theory": "Złożoność_obliczeniowa",
    "Videoconferencing": "Wideotelefonia",
    "Railway_electrification_system": "Przewody_trakcyjne",
    "Umayyad_Caliphate": "Umajjadzi",
    "Southern_California": "Kalifornia",
    "Renewable_energy_commercialization": "Odnawialne_źródła_energii",
    "Hindu_philosophy": "Filozofia_indyjska",
    "BBC_Television": "BBC",
    "Orthodox_Judaism": "Judaizm",
    "Geological_history_of_Earth": "Historia_Ziemi",
}


def find_corresponding_polish_articles() -> Generator[Tuple[int, str, Optional[int], Optional[str]], None, None]:
    """
    Find corresponding polish articles.

    Using polish and english wikipedia `pages` and `langlinks` files connect english wikipedia
    articles related to titles from SQuAD with articles from polish wikipedia.
    """
    # use all SQuAD 1.1 titles - SQuAD 2.0 titles are just a subset of SQuAD 1.1 titles
    with open(SQUAD_PATH, "r") as squad_file:
        squad_titles = {unquote(article["title"]) for article in json.load(squad_file)["data"]}

    enwiki_pages = load_wiki_pages(ENWIKI_PAGE_CSV_PATH)
    plwiki_pages = load_wiki_pages(PLWIKI_PAGE_CSV_PATH)
    enwiki_langlinks = load_wiki_langlinks(ENWIKI_LANGLINKS_CSV_PATH)

    for en_title in squad_titles:
        if en_title in enwiki_pages:
            en_page_id = enwiki_pages[en_title]
        else:
            en_page_id = enwiki_pages[SQUAD_ENWIKI_REDIRECTS[en_title]]

        if en_page_id in enwiki_langlinks:
            # corresponding polish article exists
            pl_title = enwiki_langlinks[en_page_id].replace(" ", "_")
            pl_page_id = plwiki_pages[pl_title]
        else:
            # use manually found corresponding polish article if exists
            pl_title = SQUAD_PLWIKI_ACCURATE_CORRESPONDING_TITLES.get(en_title)
            pl_page_id = plwiki_pages.get(pl_title)

        yield (en_page_id, en_title, pl_page_id, pl_title)


# SQuAD file with english and polish (if exists, null otherwise) corresponding wikipedia article data -
# page_id and title for every SQuAD article item
SQUAD_EXTENDED_PATH = DATA_DIR / "squad/raw/squad_full_dev_and_train_extended.json"


def extend_squad_data_with_wikipedia_info():
    with open(SQUAD_PATH, "r") as squad_file:
        squad = json.load(squad_file)["data"]

    wiki_data = {}
    for en_page_id, en_title, pl_page_id, pl_title in find_corresponding_polish_articles():
        wiki_data[en_title] = (en_page_id, pl_page_id, pl_title)

    for article in squad:
        en_title = unquote(article["title"])
        en_page_id, pl_page_id, pl_title = wiki_data[en_title]
        article["enwiki_page_id"] = en_page_id
        article["enwiki_title"] = en_title
        article["plwiki_page_id"] = pl_page_id
        article["plwiki_title"] = pl_title

    with open(SQUAD_EXTENDED_PATH, "w") as squad_extended_file:
        json.dump({"data": squad}, squad_extended_file, indent=4)


# extend_squad_data_with_wikipedia_info()


def extract_data_from_wiki_pagelinks_sql(wiki_pagelinks_sql_path, wiki_pagelinks_csv_output_path):
    """Leave only links to namespace 0."""
    # the file is here already preprocessed and all lines are valid
    # preprocessed file is still very big
    with open(wiki_pagelinks_sql_path, "r") as wiki_pagelinks_sql_file, open(
        wiki_pagelinks_csv_output_path, "w"
    ) as wiki_pagelinks_csv_output_file:
        csv_writer = csv.writer(wiki_pagelinks_csv_output_file, delimiter=" ", quoting=csv.QUOTE_MINIMAL)
        rows = []
        batch_num = 1
        while True:
            line = wiki_pagelinks_sql_file.readline()
            if not line:
                break
            value = literal_eval(line[:-2])
            # https://www.mediawiki.org/wiki/Manual:Pagelinks_table
            # +-------------------+------------------+------+-----+---------+-------+
            # | Field             | Type             | Null | Key | Default | Extra |
            # +-------------------+------------------+------+-----+---------+-------+
            # | pl_from           | int(10) unsigned | NO   | PRI | 0       |       |
            # | pl_from_namespace | int(11)          | NO   | MUL | 0       |       |
            # | pl_title          | varbinary(255)   | NO   | PRI |         |       |
            # | pl_namespace      | int(11)          | NO   | PRI | 0       |       |
            # +-------------------+------------------+------+-----+---------+-------+
            source_page_id, source_namespace, target_title, target_namespace = value
            rows.append([source_page_id, target_title])
            if len(rows) == 100000:
                csv_writer.writerows(rows)
                rows = []
                batch_num += 1
        csv_writer.writerows(rows)


# extract_data_from_wiki_pagelinks_sql(PLWIKI_PAGELINKS_SQL_PATH, PLWIKI_PAGELINKS_CSV_PATH)


def load_wiki_pagelinks_from_csv(wiki_pagelinks_csv_path):
    # get all polish articles that aren't a redirect
    plwiki_pages = load_wiki_pages(PLWIKI_PAGE_CSV_PATH)
    plwiki_page_ids = set(plwiki_pages.values())
    pagelinks = defaultdict(set)
    with open(wiki_pagelinks_csv_path, "r", newline="") as wiki_pagelinks_csv_file:
        csv_reader = csv.reader(wiki_pagelinks_csv_file, delimiter=" ")
        for source_page_id, target_title in csv_reader:
            source_page_id = int(source_page_id)
            if source_page_id in plwiki_page_ids and target_title in plwiki_pages:
                # add if source and target page are not a redirect
                pagelinks[source_page_id].add(plwiki_pages[target_title])
    return pagelinks


# pagelinks = load_wiki_pagelinks_from_csv(PLWIKI_PAGELINKS_CSV_PATH)
# with open(PLWIKI_PAGELINKS_DICT_PATH, "wb") as wiki_pagelinks_dict_file:
#     pickle.dump(pagelinks, wiki_pagelinks_dict_file)


def load_wiki_pagelinks(wiki_pagelinks_dict_path):
    with open(wiki_pagelinks_dict_path, "rb") as wiki_pagelinks_dict_file:
        pagelinks = pickle.load(wiki_pagelinks_dict_file)
    return pagelinks


def find_similar_pages(depth=0, mapping=False) -> Union[Set, Dict[int, List[int]]]:
    """
    Depending on `depth` parameter, for every article return its most similar pages (for depth 0 this just that article).
    """
    with open(SQUAD_EXTENDED_PATH, "r") as squad_file:
        # get page IDs of all corresponding polish SQuAD articles
        page_ids = {article["plwiki_page_id"] for article in json.load(squad_file)["data"] if article["plwiki_page_id"]}
    if depth == 0:
        if mapping:
            return {page_id: [page_id] for page_id in page_ids}
        return page_ids

    if depth == 1:
        most_similar = load_most_similar_squad_articles()
        if mapping:
            return {page_id: [sim_page_id for sim_page_id, sim in most_similar[page_id]] for page_id in page_ids}
        return {sim_page_id for page_id in page_ids for sim_page_id, sim in most_similar[page_id]}

        # this is an old code
        # pagelinks = load_wiki_pagelinks(PLWIKI_PAGELINKS_DICT_PATH)
        # q = queue.Queue()
        # result = defaultdict(set)
        # for page_id in page_ids:
        #     q.put_nowait((0, page_id, page_id))  # (depth, page_id, first_source_page_id)
        #
        # while q.qsize() > 0:
        #     d, page_id, source_page_id = q.get_nowait()
        #     result[source_page_id].add(page_id)
        #     if d < depth:
        #         for linked_page_id in pagelinks[page_id]:
        #             q.put_nowait((d + 1, linked_page_id, source_page_id))
        #
        # if mapping:
        #     return result
        #
        # page_ids = set()
        # for _, linked_page_ids in result.items():
        #     page_ids |= linked_page_ids
        # return page_ids

    raise NotImplementedError()


# page_ids = find_all_linked_pages(depth=1)


def generate_pagelinks_with_ids():
    pagelinks = load_wiki_pagelinks(PLWIKI_PAGELINKS_DICT_PATH)
    with open(PLWIKI_PAGELINKS_ID_PAIRS_CSV, "w") as pagelinks_id_pairs_file:
        csv_writer = csv.writer(pagelinks_id_pairs_file, delimiter=" ", quoting=csv.QUOTE_MINIMAL)
        for page_id, linked_page_ids in pagelinks.items():
            for linked_page_id in linked_page_ids:
                if linked_page_id != page_id:
                    csv_writer.writerow([page_id, linked_page_id])


def save_pagelinks_with_ids():
    with open(PLWIKI_PAGELINKS_IDS_SHUFFLED_CSV, "r") as pagelinks_ids_csv_file, open(
        PLWIKI_PAGELINKS_IDS_SHUFFLED_FILE, "wb"
    ) as pagelinks_ids_file:
        for line in pagelinks_ids_csv_file:
            from_page_id, to_page_id = line.split(" ")
            pickle.dump([from_page_id, to_page_id], pagelinks_ids_file)


LINKED_PAGE_IDS_WORD2VEC_MODEL = str(DATA_DIR / "wikipedia/raw/pl/linked_page_ids_word2vec.model")


def train_word2vec_for_pagelinks_ids():
    model = Word2Vec(
        corpus_file=str(PLWIKI_PAGELINKS_IDS_SHUFFLED_CSV),
        iter=10,  # number of epochs
        size=100,
        sg=1,
        hs=0,
        negative=5,
        window=1,
        min_count=1,
        workers=7,
    )
    model.save(LINKED_PAGE_IDS_WORD2VEC_MODEL)


MOST_SIMILAR_SQUAD_ARTICLES = DATA_DIR / "wikipedia/filtered/pl/most_similar_squad_articles"


def find_most_similar_squad_articles(topn=50):
    model = Word2Vec.load(LINKED_PAGE_IDS_WORD2VEC_MODEL)

    with open(SQUAD_EXTENDED_PATH, "r") as squad_extended_file:
        squad = json.load(squad_extended_file)["data"]

    most_similar = {}
    for article in squad:
        pl_page_id = article["plwiki_page_id"]
        if pl_page_id is not None:
            # a sequence of (ID, similarity) is assigned as a dict value
            most_similar[pl_page_id] = [
                (int(res[0]), res[1]) for res in model.wv.similar_by_vector(model.wv[str(pl_page_id)], topn)
            ]

    with open(MOST_SIMILAR_SQUAD_ARTICLES, "wb") as most_similar_squad_articles_file:
        pickle.dump(most_similar, most_similar_squad_articles_file)


# find_most_similar_squad_articles(topn=50)


def load_most_similar_squad_articles() -> Dict[int, List[Union[int, float]]]:
    with open(MOST_SIMILAR_SQUAD_ARTICLES, "rb") as most_similar_squad_articles_file:
        return pickle.load(most_similar_squad_articles_file)


def print_titles_of_most_similar_squad_articles():
    plwiki_pages = load_wiki_pages(PLWIKI_PAGE_CSV_PATH, title_to_page_id=False)
    most_similar = load_most_similar_squad_articles()

    for page_id, most_similar_page_ids in most_similar.items():
        print()
        print(f"Article: {plwiki_pages[page_id]}")
        print(f"Most similar: {', '.join([plwiki_pages[sim_page_id] for sim_page_id, sim in most_similar_page_ids])}")
        print()


# print_titles_of_most_similar_squad_articles()
