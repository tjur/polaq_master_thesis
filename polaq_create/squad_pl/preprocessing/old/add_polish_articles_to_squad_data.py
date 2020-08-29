import json
import copy
import pickle
import psycopg2
import xml.etree.ElementTree as etree


def add_polish_articles_to_squad_data(
    preprocess_squad_data_filename, pl_wiki_index_filename, squad_data_json_filename, squad_data_filename
):
    """
        For every article in preprocessing squad data tries (using langlinks database)
        to find corresponding polish article id. Using langlinks we get polish title
        and then we use index file for polish wikipedia to find article's ID.
        Articles without a match are removed.
        """
    try:
        with open(squad_data_filename, "rb") as squad_data_file:
            squad_data = pickle.load(squad_data_file)
            return squad_data

    except FileNotFoundError:
        with psycopg2.connect(
            dbname="langlinks", host="localhost,", user="postgres", password="postgres"
        ) as conn, open(squad_data_json_filename, "w", encoding="utf8") as squad_data_json_file, open(
            squad_data_filename, "wb"
        ) as squad_data_file, open(
            preprocess_squad_data_filename, "rb"
        ) as preprocess_squad_data_file, open(
            pl_wiki_index_filename, "r", encoding="utf8"
        ) as index_file:

            cursor = conn.cursor()
            squad_data = pickle.load(preprocess_squad_data_file)
            ids_to_remove = set()

            # collect all polish titles and ids from polish wiki index
            pl_wiki_title_id_dict = {}
            for line in index_file:
                _, id, pl_wiki_title = line.split(":", 2)
                pl_wiki_title = pl_wiki_title.rstrip()  # remove newline

                pl_wiki_title_id_dict[pl_wiki_title] = id

            for article_data in squad_data:
                en_wiki_page_id = article_data["en_wiki_page_id"]
                cursor.execute(
                    """SELECT ll_title FROM langlinks
                            WHERE ll_from=%(page_id)s AND ll_lang=%(lang)s""",
                    {"page_id": en_wiki_page_id, "lang": "pl"},
                )
                result = cursor.fetchone()
                if result:
                    # find pl_wiki_page_id using polish title and add it to article data
                    polish_title = result[0]
                    if polish_title in pl_wiki_title_id_dict:
                        article_data["pl_wiki_page_id"] = int(pl_wiki_title_id_dict[polish_title])
                    else:
                        # this is the case when we have title#section form, then we just need title
                        # in fact, there is only one such a case for SQuAD - Judaizm#Judaizm ortodoksyjny
                        if "#" in polish_title:
                            polish_title = polish_title.split("#", 1)[0]
                            if polish_title in pl_wiki_title_id_dict:
                                article_data["pl_wiki_page_id"] = int(pl_wiki_title_id_dict[polish_title])
                            else:
                                ids_to_remove.add(en_wiki_page_id)
                        else:
                            ids_to_remove.add(en_wiki_page_id)

                else:
                    # There is no corresponding polish article
                    # Remove article from squad_data
                    ids_to_remove.add(en_wiki_page_id)

            squad_data = list(filter(lambda item: item["en_wiki_page_id"] not in ids_to_remove, squad_data))

            json.dump(squad_data, squad_data_json_file, indent=4)
            pickle.dump(squad_data, squad_data_file)
            return squad_data


def filter_wiki_articles(
    squad_data,
    preprocess_en_wiki_xml_filename,
    pl_wiki_xml_filename,
    filtered_en_wiki_filename,
    filtered_pl_wiki_filename,
):
    """
    Removes not needed (those that are not in squad_data) articles from polish wiki xml
    file and from (initially filtered during preprocessing) english wiki xml file
    """
    try:
        with open(filtered_en_wiki_filename, "r", encoding="utf8") as filtered_en_wiki_file, open(
            filtered_pl_wiki_filename, "r", encoding="utf8"
        ) as filtered_pl_wiki_file:
            pass

    except FileNotFoundError:
        with open(preprocess_en_wiki_xml_filename, "r", encoding="utf8") as preprocess_en_wiki_xml_file, open(
            pl_wiki_xml_filename, "r", encoding="utf8"
        ) as pl_wiki_xml_file:

            en_wiki_ids = set(map(lambda item: item["en_wiki_page_id"], squad_data))
            pl_wiki_ids = set(map(lambda item: item["pl_wiki_page_id"], squad_data))

            # Filter english wiki xml file
            namespace = next(etree.iterparse(preprocess_en_wiki_xml_filename, events=("start-ns", "end")))[1][1]
            etree.register_namespace("", namespace)
            preprocess_en_wiki_xml_file.seek(0)  # return to the beginning
            # name of a tag is with a full namespace name
            page_tag = "{{{0}}}page".format(namespace)
            id_tag = "{{{0}}}id".format(namespace)
            pages = etree.Element("pages")
            previous_elem = None
            for event, elem in etree.iterparse(preprocess_en_wiki_xml_filename, events=("end",)):
                if elem.tag == page_tag and int(elem.find(id_tag).text) in en_wiki_ids:
                    page = copy.deepcopy(elem)
                    pages.append(page)
                if previous_elem:
                    previous_elem.clear()  # remove from memory
                previous_elem = elem
            tree = etree.ElementTree(pages)
            tree.write(filtered_en_wiki_filename)

            # Filter polish wiki xml file
            namespace = next(etree.iterparse(pl_wiki_xml_filename, events=("start-ns", "end")))[1][1]
            etree.register_namespace("", namespace)
            pl_wiki_xml_file.seek(0)  # return to the beginning
            # name of a tag is with a full namespace name
            page_tag = "{{{0}}}page".format(namespace)
            id_tag = "{{{0}}}id".format(namespace)
            pages = etree.Element("pages")
            previous_elem = None
            for event, elem in etree.iterparse(pl_wiki_xml_filename, events=("end",)):
                if elem.tag == page_tag and int(elem.find(id_tag).text) in pl_wiki_ids:
                    page = copy.deepcopy(elem)
                    pages.append(page)
                if previous_elem:
                    previous_elem.clear()  # remove from memory
                previous_elem = elem
            tree = etree.ElementTree(pages)
            tree.write(filtered_pl_wiki_filename)


if __name__ == "__main__":
    preprocessing_squad_data_filename = "../data/preprocessing/squad_data"
    en_wiki_langlinks_db = "../data/wikipedia/enwiki-20180320-langlinks.sql"
    pl_wiki_index_filename = "../data/wikipedia/plwiki-20180320-pages-articles-multistream-index.txt"
    squad_data_json_filename = "../data/squad_data.json"
    squad_data_filename = "../data/squad_data"
    preprocessing_en_wiki_xml_filename = "../data/preprocessing/filtered_wiki_articles.xml"
    pl_wiki_xml_filename = "../data/wikipedia/plwiki-20180320-pages-articles-multistream.xml"
    filtered_en_wiki_filename = "../data/wikipedia/filtered_en_wiki.xml"
    filtered_pl_wiki_filename = "../data/wikipedia/filtered_pl_wiki.xml"

    squad_data = add_polish_articles_to_squad_data(
        preprocessing_squad_data_filename,
        pl_wiki_index_filename,
        squad_data_json_filename,  # output file
        squad_data_filename,
    )  # output file

    filter_wiki_articles(
        squad_data,
        preprocessing_en_wiki_xml_filename,
        pl_wiki_xml_filename,
        filtered_en_wiki_filename,  # input/output file
        filtered_pl_wiki_filename,
    )  # input/output file

    """
    Result files are:
    squad_data, squad_data.json, filtered_en_wiki.xml, filtered_pl_wiki.xml
    """
