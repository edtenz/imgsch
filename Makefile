

.PHONY start-docker:
start-docker:
	docker-compose up -d
	# docker-compose up 


.PHONY stop-docker:
stop-docker:
	docker-compose down


