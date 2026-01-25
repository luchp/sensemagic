echo off

:: change to directory this batchfile is in.
cd %~p0%
SET PYTHON_PATH=..
::Get parent foldername from the current batfile path.
SET "CDIR=%~dp0"
:: for loop requires removing trailing backslash from %~dp0 output
SET "CDIR=%CDIR:~0,-1%"
FOR %%i IN ("%CDIR%") DO SET "VENV_DIR=%%~nxi"
:: now that we know the parent folder name in VENV_DIR
SET VENV_DIR=%DATA_DIR%\venv\%VENV_DIR%
echo %VENV_DIR%

:: Activate the virtual environment
CALL "%VENV_DIR%\Scripts\activate.bat"

python -m uvicorn main:app --reload --port 8000

