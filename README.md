# Iris Insights

Iris Insights is a compact yet polished data science style project designed to
highlight an end-to-end workflow while remaining fully runnable in restricted
environments. The project synthesises an Iris-like dataset, performs lightweight
feature engineering, trains a simple classifier, and exports evaluation
artefacts — all with nothing more than the Python standard library.

## Project Structure

```
.
├── notebooks/
│   └── iris_analysis.ipynb      # Narrative analysis and portfolio talking points
├── reports/
│   └── figures/                 # Automatically generated text-based summaries
├── scripts/
│   ├── api_server.py            # Lightweight HTTP API exposing model predictions
│   └── run_pipeline.py          # End-to-end training, logging, and reporting script
├── src/iris_insights/           # Reusable Python package
│   ├── data.py
│   ├── evaluation.py
│   ├── features.py
│   └── model.py
└── tests/
    └── test_pipeline.py
```

## Getting Started

1. **(Optional) create a virtual environment**

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

2. **Run the automated pipeline**

   No extra packages are required. Execute the workflow directly:

   ```bash
   python scripts/run_pipeline.py
   ```

   The script now benchmarks the nearest centroid baseline alongside
   handcrafted random forest and gradient boosting ensembles. Every run logs
   parameters, metrics, and the classification report to
   `reports/experiments/<experiment>/<run>/` in an MLflow-inspired format. A
   high-level summary plus correlation and pairwise statistics remain
   available under `reports/`.

3. **Serve real-time predictions** *(optional)*

   Start a lightweight HTTP API that trains the random forest ensemble at
   launch and exposes `/predict` and `/health` endpoints:

   ```bash
   python scripts/api_server.py
   ```

   Send feature dictionaries (matching the Iris measurements) to
   `http://localhost:8000/predict` and receive numeric and human-readable
   species predictions in the response payload.

4. **Explore the notebook** *(optional)*

   Open `notebooks/iris_analysis.ipynb` for a high-level walkthrough of the
   modelling story. The notebook references the generated artefacts and suggests
   extensions that can be discussed in a portfolio or interview setting.

## Testing

Run the unit tests with:

```bash
pytest
```

## Why This Project Stands Out

- **Offline friendly**: Synthetic data generation, ensemble modelling, logging,
  and API deployment rely solely on the Python standard library, so the project
  runs without internet access or PyPI.
- **Well-documented**: Modularised code and narrative notebook explain each
  decision, making it easy for reviewers to follow.
- **Actionable artefacts**: Automated text summaries of correlations, class
  performance, and pipeline configuration demonstrate analytical depth even
  without graphical backends.

Feel free to extend the project with richer models, experiment tracking, or web
visualisations to further showcase your skills.
