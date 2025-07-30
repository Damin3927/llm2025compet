where nvidia-smi >nul 2>nul
IF %ERRORLEVEL% EQU 0 (
    echo build type: gpu
    set BASE_IMAGE_TYPE=gpu
) ELSE (
    echo build type: cpu
    set BASE_IMAGE_TYPE=cpu
)
echo start build
docker build --build-arg BASE_IMAGE_TYPE=%BASE_IMAGE_TYPE% -t llm2025compet .