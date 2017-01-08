init:
	pip install -r requirements.txt

run:
	/usr/bin/python3 start.py

clean-cache:
	find . -name '__pycache__' -exec rm -R 

