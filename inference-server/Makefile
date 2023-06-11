
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


.PHONY proto:
proto:
	#python -m grpc_tools.protoc -I ./api --python_out=./src --grpc_python_out=./src ./api/api.proto
	python -m grpc_tools.protoc --proto_path=./api ./api/api.proto --python_out=./src --grpc_python_out=./src

