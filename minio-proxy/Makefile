BINARY=minio-proxy

all: build

clean:
	rm -f bin/*

build: clean
	go build -o bin/$(BINARY) main.go
	GOOS=linux go build -o bin/$(BINARY)-linux main.go

run: clean build
	./bin/${BINARY}
