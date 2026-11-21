@echo off
setlocal

set "PROJECT_ROOT=%~dp0"
pushd "%PROJECT_ROOT%"

set "PYTHON_EXE=%PROJECT_ROOT%..\ComfyUI_windows_portable2\python_embeded\python.exe"
if not exist "%PYTHON_EXE%" (
    echo [ERROR] Embedded Python was not found:
    echo %PYTHON_EXE%
    echo Check the ComfyUI_windows_portable2 folder path.
    pause
    exit /b 1
)

if not exist "%PROJECT_ROOT%comfy_profile_automation\__init__.py" (
    echo [ERROR] The comfy_profile_automation package is missing.
    echo The local project folder does not match the GitHub repository.
    echo Update or re-download the repository, then run this file again.
    pause
    exit /b 1
)

set "PYTHONPATH=%PROJECT_ROOT%;%PYTHONPATH%"

if not exist "%PROJECT_ROOT%config.yml" (
    copy "%PROJECT_ROOT%config.example.yml" "%PROJECT_ROOT%config.yml" >nul
)

"%PYTHON_EXE%" "%PROJECT_ROOT%main.py" --config "%PROJECT_ROOT%config.yml" %*
set "EXIT_CODE=%ERRORLEVEL%"

popd
pause
exit /b %EXIT_CODE%
