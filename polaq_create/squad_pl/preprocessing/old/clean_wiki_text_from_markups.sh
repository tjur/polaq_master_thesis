#!/bin/bash

# Using wikiextractor (https://github.com/MouhamadAboShokor/wikiextractor) module clean wikipedia text
# (both polish and english) from wikipedia markup language
# Wikipedia uses html encoding for some characters so we also decode text

PL_WIKI_RESULT="../data/wikipedia/filtered_cleaned_pl_wiki.xml"
EN_WIKI_RESULT="../data/wikipedia/filtered_cleaned_en_wiki.xml"

python -m wikiextractor.WikiExtractor --output - ../data/wikipedia/filtered_pl_wiki.xml | \
python -c "import html, sys; print(html.unescape(sys.stdin.read()), end='')" > $PL_WIKI_RESULT

python -m wikiextractor.WikiExtractor --output - ../data/wikipedia/filtered_en_wiki.xml | \
python -c "import html, sys; print(html.unescape(sys.stdin.read()), end='')" > $EN_WIKI_RESULT
