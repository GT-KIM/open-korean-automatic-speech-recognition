@echo off
set CONTAINER_NAME=open-korean-asr-container
set IMAGE_NAME=open-korean-asr
set DATASET_PATH=F://data
set PROJECT_PATH=%~dp0..

echo Attaching to running container...
docker restart %CONTAINER_NAME%
docker attach %CONTAINER_NAME%