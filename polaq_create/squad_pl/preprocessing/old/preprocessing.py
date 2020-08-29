import re
import sys
import copy
import json
import ijson
import pickle
from urllib.parse import unquote
from collections import defaultdict
import xml.etree.ElementTree as etree


prog_squad = re.compile(r"[_,():&\']")
prog_wiki = re.compile(r"[_ ,():&\']")

# TODO Use BASE_DIR in all data paths (not only in this file)


def normalize_squad_title(title):
    return re.sub(prog_squad, "", unquote(title).lower())


def normalize_wiki_title(title):
    return re.sub(prog_wiki, "", unquote(title).lower())


def normalize_squad_titles(titles):
    return set(map(normalize_squad_title, titles))


def normalize_wiki_titles(titles):
    return set(map(normalize_wiki_title, titles))


def collect_squad_article_titles(squad_json_filename):
    with open(squad_json_filename, "r", encoding="utf8") as squad_json:
        objects = ijson.items(squad_json, "data.item")
        titles = [o["title"] for o in objects]
        return titles


def get_squad_titles(dev_squad_json_filename, train_squad_json_filename, squad_titles_data_filename):
    """
    Returns a set of all SQuAD titles (normalized)
    """
    print("Starting get_squad_titles", flush=True)
    try:
        with open(squad_titles_data_filename, "rb") as squad_titles_data_file:
            squad_titles_data = pickle.load(squad_titles_data_file)
            squad_titles_normalized = squad_titles_data["normalized"]
            squad_titles_original = squad_titles_data["original"]

    except FileNotFoundError:
        with open(squad_titles_data_filename, "wb") as squad_titles_data_file:
            dev_titles = collect_squad_article_titles(dev_squad_json_filename)
            train_titles = collect_squad_article_titles(train_squad_json_filename)
            squad_titles_original = set(dev_titles + train_titles)
            squad_titles_normalized = normalize_squad_titles(squad_titles_original)
            squad_titles_data = {}
            squad_titles_data["original"] = squad_titles_original
            squad_titles_data["normalized"] = squad_titles_normalized
            pickle.dump(squad_titles_data, squad_titles_data_file)

    # print('Number of original squad titles: {0}'.format(len(squad_titles_original)))
    # print('Number of normalized squad titles: {0}'.format(len(squad_titles_normalized)))

    return squad_titles_normalized


def filter_id_title_from_index(wiki_index_filename, filtered_wiki_ids_filename, squad_titles):
    """
        Returns a dictionary: id of a wiki article -> wiki article title (not normalized).
        """
    print("Starting filter_id_title_from_index", flush=True)
    try:
        with open(id_title_dict_with_redirects_filename, "rb") as id_title_dict_file:
            id_title_dict = pickle.load(id_title_dict_file)
            return id_title_dict

    except FileNotFoundError:
        with open(wiki_index_filename, "r", encoding="utf8") as index_file, open(
            id_title_dict_with_redirects_filename, "wb"
        ) as id_title_dict_file:
            id_title_dict = {}
            for line in index_file:
                # not all lines follow this format (but we hope it will work for SQuAD)
                _, id, wiki_title = line.split(":", 2)
                wiki_title = wiki_title.rstrip()  # remove newline
                wiki_title_normalized = normalize_wiki_title(wiki_title)
                if wiki_title_normalized in squad_titles:
                    id_title_dict[int(id)] = wiki_title
            pickle.dump(id_title_dict, id_title_dict_file)
            return id_title_dict


def filter_wiki_articles_with_redirects(wiki_xml_filename, filtered_wiki_articles_filename, ids):
    """
    Opens wikipedia xml file and leaves only needed articles - 
    those that have id in ids. We still have a lot of duplicated titles
    because of the redirects (#redirect in <text> tag).
    """
    print("Starting filter_wiki_articles_with_redirects", flush=True)
    with open(wiki_xml_filename, "r", encoding="utf8") as wiki_xml_file:
        try:
            with open(filtered_wiki_articles_filename, "r", encoding="utf8") as filtered_wiki_articles_file:
                pass

        except FileNotFoundError:
            with open(filtered_wiki_articles_filename, "w+", encoding="utf8") as filtered_wiki_articles_file:
                namespace = next(etree.iterparse(wiki_xml_filename, events=("start-ns", "end")))[1][1]
                etree.register_namespace("", namespace)
                wiki_xml_file.seek(0)  # return to the beginning
                # name of a tag is with a full namespace name
                siteinfo_tag = "{{{0}}}siteinfo".format(namespace)  # we need that info for wikiextractor
                page_tag = "{{{0}}}page".format(namespace)
                id_tag = "{{{0}}}id".format(namespace)
                pages = etree.Element("mediawiki")

                previous_elem = None
                for event, elem in etree.iterparse(wiki_xml_filename, events=("end",)):
                    if elem.tag == page_tag and int(elem.find(id_tag).text) in ids:
                        page = copy.deepcopy(elem)
                        pages.append(page)

                    elif elem.tag == siteinfo_tag:  # this tag is on the very beginning
                        siteinfo = copy.deepcopy(elem)
                        pages.append(siteinfo)

                    if previous_elem:
                        previous_elem.clear()  # remove from memory
                    previous_elem = elem

                tree = etree.ElementTree(pages)
                tree.write(filtered_wiki_articles_filename)


def remove_redirects(filtered_wiki_articles_filename):
    """
    Removes wiki articles that are just redirect articles.
    """
    print("Starting remove_redirects", flush=True)
    doc = etree.parse(filtered_wiki_articles_filename)
    root = doc.getroot()
    namespace = next(etree.iterparse(filtered_wiki_articles_filename, events=("start-ns", "end")))[1][1]
    etree.register_namespace("", namespace)
    page_tag = "{{{0}}}page".format(namespace)
    revision_tag = "{{{0}}}revision".format(namespace)
    text_tag = "{{{0}}}text".format(namespace)
    redirects = 0
    for tag in root.findall(page_tag):
        if tag.find(revision_tag).find(text_tag).text.lstrip().lower().startswith("#redirect"):
            root.remove(tag)
            redirects += 1
    if redirects > 0:
        doc.write(filtered_wiki_articles_filename)


def remove_unwanted_articles(filtered_wiki_articles_filename, ids):
    """
    Removes all wiki articles with id in ids.
    """
    print("Starting remove_unwanted_articles", flush=True)
    ids = set(ids)
    doc = etree.parse(filtered_wiki_articles_filename)
    root = doc.getroot()
    namespace = next(etree.iterparse(filtered_wiki_articles_filename, events=("start-ns", "end")))[1][1]
    etree.register_namespace("", namespace)
    page_tag = "{{{0}}}page".format(namespace)
    id_tag = "{{{0}}}id".format(namespace)
    removed = 0
    for tag in root.findall(page_tag):
        if int(tag.find(id_tag).text) in ids:
            root.remove(tag)
            removed += 1
    if removed > 0:
        doc.write(filtered_wiki_articles_filename)


def get_id_title_dict(filtered_wiki_articles_filename, id_title_dict_filename):
    """
    Gets or create (id -> title) dict from fully filtered xml wiki file.
    """
    print("Starting get_id_title_dict", flush=True)
    try:
        with open(id_title_dict_filename, "rb") as id_title_dict_file:
            id_title_dict = pickle.load(id_title_dict_file)
            return id_title_dict

    except FileNotFoundError:
        with open(id_title_dict_filename, "wb") as id_title_dict_file:
            doc = etree.parse(filtered_wiki_articles_filename)
            root = doc.getroot()
            namespace = next(etree.iterparse(filtered_wiki_articles_filename, events=("start-ns", "end")))[1][1]
            page_tag = "{{{0}}}page".format(namespace)
            id_tag = "{{{0}}}id".format(namespace)
            title_tag = "{{{0}}}title".format(namespace)
            id_title_dict = {}
            for tag in root.findall(page_tag):
                id = int(tag.find(id_tag).text)
                title = tag.find(title_tag).text
                id_title_dict[id] = title
            pickle.dump(id_title_dict, id_title_dict_file)
            return id_title_dict


def find_wiki_title_normalization_collisions(id_title_dict):
    """
    Checks whether every squad title (normalized) has at most one
    corresponding (normalized) wiki title. If not, it prints all collisions.
    It used to find ids of articles that we want to remove with remove_unwanted_articles.
    """
    norm_title_titles_dict = defaultdict(list)
    collisions_dict = {}
    for id, wiki_title in id_title_dict.items():
        norm_title_titles_dict[normalize_wiki_title(wiki_title)].append((wiki_title, id))
    for norm_title, titles in norm_title_titles_dict.items():
        if len(titles) > 1:
            collisions_dict[norm_title] = titles

    if collisions_dict:
        print("-----")
        print(
            (
                "There were found wikipedia titles that have the same normalized form.\n"
                "It is incorrect because every normalized SQuAD title should have"
                "exactly one (or zero) normalized wiki title.\nPlease change filtered wiki"
                "articles xml file and leave only those articles that are indeedin SQuAD. "
                "Found collisions:\n"
            )
        )

        for norm_title, titles in collisions_dict.items():
            print("{0}:".format(norm_title))
            for title, id in titles:
                print("    - {0}, id: {1}".format(title, id), flush=True)

        print("-----")
        sys.exit(1)


def combine_wiki_and_squad_data(
    dev_squad_json_filename,
    train_squad_json_filename,
    filtered_wiki_articles_filename,
    squad_data_json_filename,
    squad_data_filename,
    id_title_dict,
):
    """
    Find matching wiki articles for SQuAD data

    Takes (id -> wiki title) dict and assigns each article id to some SQuAD title.
    Every SQuAD title should have only one (or zero) corresponding wiki title.
    Those SQuAD elements that don't match any article are removed.
    """
    print("Starting combine_wiki_and_squad_data", flush=True)
    try:
        with open(squad_data_filename, "rb") as squad_data_file:
            squad_data = pickle.load(squad_data_file)
            return squad_data

    except FileNotFoundError:
        with open(squad_data_filename, "wb") as squad_data_file, open(
            squad_data_json_filename, "w", encoding="utf8"
        ) as squad_data_json_file, open(dev_squad_json_filename, "r", encoding="utf8") as dev_squad_json_file, open(
            train_squad_json_filename, "r", encoding="utf8"
        ) as train_squad_json_file, open(
            filtered_wiki_articles_filename, "r", encoding="utf8"
        ) as filtered_wiki_articles_file:

            # SQuAD data is a list of dicts, each with questions/answers
            # corresponding to max one wikipedia article (but with many contexts/fragments)
            dev_squad_data = json.load(dev_squad_json_file)["data"]
            train_squad_data = json.load(train_squad_json_file)["data"]
            squad_data = dev_squad_data + train_squad_data

            normalized_wiki_titles = normalize_wiki_titles(id_title_dict.values())

            # assertion
            # number of normalized wiki titles must be equal number of original wiki titles
            # thanks to that for every SQuAD title there is at most one corresponding wiki title
            assert len(normalized_wiki_titles) == len(id_title_dict.values())

            norm_title_id_dict = {}
            for id, wiki_title in id_title_dict.items():
                norm_title_id_dict[normalize_wiki_title(wiki_title)] = id

            already_used_titles = set()
            titles_to_remove = []
            for article_data in squad_data:
                squad_title = article_data["title"]
                normalized_squad_title = normalize_squad_title(squad_title)
                if normalized_squad_title in normalized_wiki_titles:
                    if normalized_squad_title not in already_used_titles:
                        # add new field 'en_wiki_page_id' to squad data
                        article_data["en_wiki_page_id"] = norm_title_id_dict[normalized_squad_title]
                        already_used_titles.add(normalized_squad_title)
                    else:
                        raise ValueError("Title was already used.")
                else:
                    # Remove it, it doesn't have matching article.
                    titles_to_remove.append(squad_title)

            squad_data = list(filter(lambda item: item["title"] not in titles_to_remove, squad_data))

            json.dump(squad_data, squad_data_json_file, indent=4)
            pickle.dump(squad_data, squad_data_file)
            return squad_data


if __name__ == "__main__":
    dev_squad_json_filename = "../data/dev-v1.1.json"
    train_squad_json_filename = "../data/train-v1.1.json"

    squad_titles_filename = "../data/preprocessing/squad_titles_data"
    wiki_index_filename = "../data/wikipedia/enwiki-20180120-pages-articles-multistream-index.txt"
    wiki_xml_filename = "../data/wikipedia/enwiki-20180120-pages-articles-multistream.xml"
    id_title_dict_with_redirects_filename = "../data/preprocessing/id_title_dict_with_redirects"
    filtered_wiki_articles_filename = "../data/preprocessing/filtered_wiki_articles.xml"
    id_title_dict_filename = "../data/preprocessing/id_title_dict"
    squad_data_json_filename = "../data/preprocessing/squad_data.json"
    squad_data_filename = "../data/preprocessing/squad_data"

    squad_titles = get_squad_titles(
        dev_squad_json_filename, train_squad_json_filename, squad_titles_filename
    )  # output file

    id_title_dict_with_redirects = filter_id_title_from_index(
        wiki_index_filename, id_title_dict_with_redirects_filename, squad_titles  # output file
    )
    ids_with_redirects = id_title_dict_with_redirects.keys()

    filter_wiki_articles_with_redirects(
        wiki_xml_filename, filtered_wiki_articles_filename, ids_with_redirects  # output file
    )

    remove_redirects(filtered_wiki_articles_filename)  # output file

    unwanted_articles_ids = [11033337, 7664201, 50955731, 58171, 6170143, 8533371, 13994263, 20900491]
    remove_unwanted_articles(filtered_wiki_articles_filename, unwanted_articles_ids)  # output file

    id_title_dict = get_id_title_dict(filtered_wiki_articles_filename, id_title_dict_filename)  # output file

    # squad_data is a result of preprocessing
    squad_data = combine_wiki_and_squad_data(
        dev_squad_json_filename,
        train_squad_json_filename,
        filtered_wiki_articles_filename,
        squad_data_json_filename,  # output file
        squad_data_filename,  # output file
        id_title_dict,
    )
