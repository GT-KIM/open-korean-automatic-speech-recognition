#!/bin/bash
CONTAINER_NAME="open-korean-asr-container"
IMAGE_NAME="open-korean-asr"
DATASET_PATH="F://data"
PROJECT_PATH=$(dirname "$0")/..

if [ "$(docker ps -a -q -f name=${CONTAINER_NAME})" ]; then
    if [ "$(docker ps -q -f name=${CONTAINER_NAME})" ]; then
        echo "Attaching to running container..."
        docker attach ${CONTAINER_NAME}
    else
        echo "Starting existing container..."
        docker start -i ${CONTAINER_NAME}
    fi
else
    echo "Creating and starting new container..."
    docker run -it --gpus all --name ${CONTAINER_NAME} -v "${PROJECT_PATH}":/app/open-korean-asr -v "${DATASET_PATH}":/app/data ${IMAGE_NAME} /bin/bash
fi
