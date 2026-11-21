@echo off
cd /d "%~dp0"
if not exist config.yml copy config.example.yml config.yml
..\ComfyUI_windows_portable2\python_embeded\python.exe main.py --config config.yml
pause
