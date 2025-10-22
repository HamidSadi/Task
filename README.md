# Iris Insights

Iris Insights is a compact yet polished data science project designed to highlight
end-to-end workflow skills for a portfolio-ready GitHub repository. The project
uses the classic Iris dataset to demonstrate exploratory data analysis (EDA),
feature engineering, model training, and automated reporting.

## Project Structure

```
.
├── notebooks/
│   └── iris_analysis.ipynb      # Narrative analysis and visualisations
├── reports/
│   └── figures/                 # Automatically generated plots
├── scripts/
│   └── run_pipeline.py          # End-to-end training and reporting script
├── src/iris_insights/           # Reusable Python package
│   ├── data.py
│   ├── evaluation.py
│   ├── features.py
│   └── model.py
└── tests/
    └── test_pipeline.py
```

## Getting Started

1. **Install dependencies**

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -e .
   ```

2. **Run the automated pipeline**

   ```bash
   python scripts/run_pipeline.py
   ```

   The script saves a text report summarising the model performance in
   `reports/model_report.txt` and exports figures to `reports/figures/`.

3. **Explore the notebook**

   Open `notebooks/iris_analysis.ipynb` to follow the exploratory narrative. The
   notebook walks through the project motivation, visualisations, and modelling
   decisions.

## Testing

Run the unit tests with:

```bash
pytest
```

## Why This Project Stands Out

- **Reproducible**: Deterministic data loading via `scikit-learn` with clear
  preprocessing steps and configuration.
- **Well-documented**: Modularised code and narrative notebook explain each
  decision, making it easy for reviewers to follow.
- **Visual and analytical**: Automated figure generation and detailed
  performance reporting demonstrate both storytelling and technical depth.

Feel free to extend the project with additional models, experiment tracking, or
deployment artifacts to further showcase your skills.
