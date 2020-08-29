import logging
from pathlib import Path


# Change defined paths if needed

# Path to the whole project
from googletrans import Translator

BASE_DIR = Path("/home/tomek/Projects/magisterka/squad_pl/")

# Path to CoreNLP
CORENLP_HOME = Path("/home/tomek/Projects/magisterka/stanford-corenlp-full-2018-10-05/")

# Path to data folder
DATA_DIR = Path("/home/tomek/Projects/magisterka/squad_pl/data")

# Path to polish wordnet (słowosieć) xml file
PLWORDNET_PATH = Path("/home/tomek/Projects/magisterka/plwordnet_4_0/plwordnet-4.0.xml")

# Path to spaCy polish model with morfeusz2 lemmatization
# https://github.com/ipipan/spacy-pl
PL_SPACY_MODEL_MORFEUSZ = Path("/home/tomek/Projects/magisterka/pl_spacy_model_morfeusz_big-0.1.0/")

SQUAD_PATH = DATA_DIR / f"squad/raw/squad_full_dev_and_train.json"
SQUAD_PATH_1_1 = DATA_DIR / f"squad/raw/squad_full_dev_and_train-v1.1.json"
SQUAD_PATH_2_0 = DATA_DIR / f"squad/raw/squad_full_dev_and_train-v2.0.json"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

logger = logging.getLogger(__name__)

tranlator = Translator()
