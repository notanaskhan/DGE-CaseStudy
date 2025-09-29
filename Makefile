
.PHONY: run-api run-ui fmt lint

PY?=.venv/bin/python
PIP?=.venv/bin/pip
UVICORN?=.venv/bin/uvicorn
STREAMLIT?=.venv/bin/streamlit

run-api:
	$(PY) -m uvicorn backend.fastapi_app.main:app --reload --port 8000

run-ui:
	$(STREAMLIT) run frontend/streamlit_app/Home.py --server.port $${UI_PORT:-8501}

fmt:
	ruff check --fix || true

lint:
	ruff check || true
