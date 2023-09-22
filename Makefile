SHELL := /bin/bash

sync_dir := ~/Google\ Drive/My\ Drive/ALICE

sync: 
	rsync -auv . $(sync_dir)  --exclude='.git/' --exclude='.venv/' --exclude='archive/' --exclude='*.egg-info/' --exclude='.DS_Store' --exclude='mask_rcnn_coco.h5' --exclude='*.pyc'

install: 
	pip install -r requirements.txt && pip install git+https://github.com/facebookresearch/detectron2.git@ff53992b1985b63bd3262b5a36167098e3dada02

