# Rige-regression-playground-app-
A Streamlit app for expressing, hypertuning, cross-validating parameters for a CSV dataset and providing enriched datasets with imported information for the model. 
An interactive ML tool for students, builders, and researchers to:
	•	Upload CSV datasets
	•	Train Ridge Regression models (default or tuned)
	•	Run cross-validation
	•	Visualize model performance
	•	Download enriched datasets with predictions

 Features
	•	CSV upload with validation (numeric-only, no missing values, min. 5 rows).
	•	Train baseline Ridge Regression.
	•	Hyperparameter tuning (alpha grid).
	•	Cross-validation with adaptive folds.
	•	Interactive plots: predictions vs actual, residuals.
	•	Download enriched CSV (raw data + model predictions).

 Project Structure
 ridge_app/
│
├── app.py                # Main Streamlit app
├── requirements.txt      # Dependencies
├── utils/                # Helper modules
│   ├── data_utils.py     # Data validation & summaries
│   ├── model_utils.py    # Training, tuning, CV, predictions
│   └── viz_utils.py      # Visualization helpers
└── results/              # (Optional) Save enriched outputs here

Setup & Installation
# Clone project (or create folder)
mkdir ridge_app && cd ridge_app
# Create virtual environment
python3 -m venv venv
source venv/bin/activate   # Mac/Linux
# .\venv\Scripts\activate  # Windows
# Install dependencies
pip install -r requirements.txt

Running the App
streamlit run app.py
Open browser → http://localhost:8501

Usage Flow
	1.	Upload CSV
	•	Must have at least 1 target column (last column).
	•	Must be numeric-only (IDs/strings auto-dropped).
	2.	Prompts
	•	Load Dataset & Summarize → stats + correlations
	•	Train Default Ridge → baseline metrics + download enriched CSV
	•	Hypertune Ridge → best params + metrics + enriched CSV
	•	Cross-Validation → mean/std MSE across folds
	•	Visualize → predictions vs actual, residual plots
	3.	Download Enriched CSV
	•	Includes all features + target + new prediction column.

 Notes
	•	Discrepancy between metrics & enriched CSV is expected:
	•	App metrics = test set only (20% holdout)
	•	Enriched CSV metrics = entire dataset (train + test)
	•	For small datasets (<30 rows), metrics may be unstable.
	•	Best results come from datasets with 100+ rows.



