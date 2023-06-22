SHELL := /bin/bash

sync_dir := ~/Google\ Drive/My\ Drive/ALICE

sync: 
	rsync -auv . $(sync_dir)  --exclude='.git/' --exclude='.venv/' --exclude='archive/' --exclude='*.egg-info/' --exclude='.DS_Store' --exclude='mask_rcnn_coco.h5' --exclude='*.pyc'

install: 
	pip install . && pip install -r requirements.txt

