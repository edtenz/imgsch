
.PHONY help:
help: ### Show this help.
	@echo "Makefile Usage:"
	@echo "  make init"
	@echo "  make install"
	@echo "  make run"

.PHONY init:
init: ### Initialize the project.
	@echo "Initializing..."
	#python3 -m venv venv
	# deactivate
	# source venv/bin/activate
	conda create -n towhee python=3.9
	conda activate towhee

.PHONY install:
install: ### Install the project.
	@echo "Installing..."
	pip3 install -U pip
	pip install -r requirements.txt



.PHONY run:
run:
	@echo "Running..."
	python src/main.py

