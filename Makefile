.PHONY: install install-edge predict demo api test figures notebooks clean

install:            ## full environment (training, evaluation, figures)
	pip install -r backend/requirements.txt

install-edge:       ## inference only (ONNX Runtime, no PyTorch)
	pip install -r backend/requirements-edge.txt

predict:            ## run the v4 ONNX model on one sample tile
	python scripts/predict_onnx.py data/samples/117991_sat.jpg --output prediction_117991.png

demo:               ## Streamlit demonstration
	python -m streamlit run demo_app.py

api:                ## FastAPI endpoint
	uvicorn backend.api:app --reload

test:
	python -m pytest backend/tests -q

figures:            ## regenerate the paper figures that run locally
	python scripts/paper_figures/architecture_diagrams.py
	python scripts/paper_figures/prediction_grid.py
	python scripts/paper_figures/pipeline_stages.py
	python scripts/paper_figures/graph_analysis.py
	python scripts/paper_figures/training_curves.py

notebooks:          ## rebuild the Kaggle notebooks from their source scripts
	python scripts/build_train_v4_notebook.py
	python scripts/paper_figures/build_eval_notebook.py

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
