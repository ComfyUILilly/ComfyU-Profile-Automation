@echo off
cd /d "%~dp0"
if not exist config.yml copy config.example.yml config.yml
python main.py --config config.yml
pause
