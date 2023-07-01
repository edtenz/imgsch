BINARY=minio-proxy

all: build

.PHONY: clean
clean:
	rm -f bin/*

.PHONY: build
build: clean
	go build -o bin/$(BINARY) main.go
	GOOS=linux go build -o bin/$(BINARY)-linux main.go

.PHONY: run
run: clean build
	./bin/${BINARY}


.PHONY: deploy
deploy: clean build
	cp bin/minio-proxy ~/go/bin
	supervisorctl restart minio-proxy