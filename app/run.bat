echo off

:: change to directory this batchfile is in.
cd %~p0%
SET PYTHON_PATH=..

uv run python -m uvicorn main:app --reload --port 8000

