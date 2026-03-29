@echo off
set CONTAINER_NAME=open-korean-asr-container
set IMAGE_NAME=open-korean-asr
set DATASET_PATH=F://data
set PROJECT_PATH=%~dp0..

docker ps -a --format "{{.Names}}" | findstr /R "^%CONTAINER_NAME%$" > nul
if %errorlevel% == 0 (
    docker ps --format "{{.Names}}" | findstr /R "^%CONTAINER_NAME%$" > nul
    if %errorlevel% == 0 (
        echo Attaching to running container...
        docker attach %CONTAINER_NAME%
    ) else (
        echo Starting existing container...
        docker start -i %CONTAINER_NAME%
    )
) else (
    echo Creating and starting new container...
    docker run -it --gpus all --name %CONTAINER_NAME% -v "%PROJECT_PATH%:/app/open-korean-asr" -v "%DATASET_PATH%:/app/data" %IMAGE_NAME%
)
