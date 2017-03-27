.DEFAULT_GOAL := all

help:
	@echo 'make         	  '
	@echo '     all     	  '
	@echo '     run     	  '
	@echo '     tidy    	  '
	@echo '     help    	  '
	@echo '     install 	  '
	@echo '     visualization '

all: install
	@echo '[*] Done.'

install:
	@echo '[+] Requirements'
	@pip install -r requirements.txt

run:
	@echo '[*] Running...'
	@./start.py

visualization:
	@echo '[*] Visualizing...'
	@./visualize_model.py

tidy:
	@echo '[-] Tidying...'
	# find . -name '__pycache__' -exec 'rm -R' 
	rm -f models/*
	rm -f models/visualization/*
	rm -f data/wav/*


.PHONY: tidy help run visualization 
