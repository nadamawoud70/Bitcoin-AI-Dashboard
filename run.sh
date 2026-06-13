#!/bin/bash
source .venv/bin/activate
python3 pipeline.py
streamlit run app.py