.PHONY run-docker:
run-docker: export API_HOST := 127.0.0.1
run-docker: export API_PORT := 8989
run-docker:
	docker run -d -p 8001:80 -e "API_URL=http://${API_HOST}:${API_PORT}" milvusbootcamp/img-search-client:1.0