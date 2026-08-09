#!/bin/bash

# Move into the app directory
cd /home/projects/sensemagic/app

# Start FastAPI with uvicorn
uv run python -m uvicorn main:app --host 0.0.0.0 --port 9000

