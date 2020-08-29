.PHONY: clean poetry-install morfeusz-install blas-install install

clean:
	rm -rf squad_pl.egg-info

poetry-install:
	poetry install --no-dev

morfeusz-install:
	wget -O - http://download.sgjp.pl/apt/sgjp.gpg.key|sudo apt-key add -
	sudo apt-add-repository http://download.sgjp.pl/apt/ubuntu
	sudo apt update
	sudo apt install -y python3-morfeusz2
	sudo apt install -y morfeusz2
	wget http://download.sgjp.pl/morfeusz/20200209/Linux/18.04/64/morfeusz2-0.4.0-py3.6-Linux-amd64.egg
	poetry run easy_install morfeusz2-0.4.0-py3.6-Linux-amd64.egg
	rm morfeusz2-0.4.0-py3.6-Linux-amd64.egg

pl-spacy-model-morfeusz-install:
	# https://github.com/ipipan/spacy-pl
	python -m pip install ../pl_spacy_model_morfeusz_big-0.1.0.tar.gz

blas-install:
	sudo apt install -y libblas-dev liblapack-dev libatlas-base-dev

install: blas-install poetry-install morfeusz-install pl-spacy-model-morfeusz-install clean
