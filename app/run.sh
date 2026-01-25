#!/bin/bash
# Activate the virtual environment
source /home/venv/fapi/bin/activate

# Move into the app directory
cd /home/projects/sensemagic/app

# Start FastAPI with uvicorn
exec python -m uvicorn main:app --host 0.0.0.0 --port 9000

