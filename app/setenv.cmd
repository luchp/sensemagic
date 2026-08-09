@echo off
setlocal enabledelayedexpansion

REM --- Check if uv is available ---
where uv >nul 2>&1
if %errorlevel%==0 goto uv_installed

echo uv not found, installing...

REM --- Run the PowerShell installer ---
powershell -ExecutionPolicy Bypass -NoLogo -NoProfile -Command ^
    "irm https://astral.sh/uv/install.ps1 | iex"
echo You must restart this batch file to complete the installation!
pause
exit /b 1

:uv_installed

rem Change to directory this batchfile is in.
rem Needed if you double-click this file in your favorite GUI
cd /d %~dp0
rem Let uv generate .venv and sync the toml file
rem after that it runs setenv.py, which will patch Intel numpy
uv run setenv.py
if errorlevel 1 (
    echo Environment setup failed!
    pause
    exit /b 1
)
rem Get a nice title for batch file
for /f "delims=" %%v in ('uv run setenv.py -v') do set TITLE=%%v

rem Keep a console open with a fresh active enviroment
echo Activating environment ...
echo To make an installer, type py2exe
cmd /k ".venv\scripts\activate && title %TITLE%"

