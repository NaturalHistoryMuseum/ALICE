include .env
export

SHELL := /bin/bash

SYNC_DIR := ~/Google\ Drive/My\ Drive/ALICE
CURRENT_DATE := $(shell date +%Y-%m-%d)

sync: 
	rsync -auv . $(SYNC_DIR)  --exclude='.git/' --exclude='.venv/' --exclude='archive/' --exclude='*.egg-info/' --exclude='.DS_Store' --exclude='mask_rcnn_coco.h5' --exclude='*.pyc'

install: 
pip install wheel pytorch & pip install -r requirements.txt

bash:
	@docker exec -i -t alice.luigi bash

build:
	@docker build -t naturalhistorymuseum/alice-luigi -t naturalhistorymuseum/alice-luigi:$(CURRENT_DATE) docker

up:
	@echo "Starting containers"
	@docker compose up

down:
	@echo "Stopping & removing containers"
	@docker compose down	

process:
	@docker exec -i -t alice.luigi process

push:
	@echo "Stopping & removing containers"
	docker login -u bensc
	docker tag alice.luigi naturalhistorymuseum/alice_luigi:latest
	docker push naturalhistorymuseum/alice_luigi:latest