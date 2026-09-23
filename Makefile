.PHONY: install train eval demo clean

install:
	pip install -r backend/requirements.txt
	pip install streamlit matplotlib networkx scipy

train:
	python backend/scripts/train.py

eval:
	python backend/scripts/evaluate.py

api:
	uvicorn backend.api:app --reload

web:
	cd frontend_viral && npm run dev

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
