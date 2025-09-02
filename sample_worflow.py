#This worflow is a sample only; it might run as is or require modifications based on environment, extensions, platforms and OS. 

# 1)  Create project folder and enter it
mkdir ridge_app && cd ridge_app

# 2) Create utils folder for helper modules
mkdir utils

# 3) Create main app file
touch app.py

# 4) Create requirements file
touch requirements.txt

# 5) Create utility Python files
touch utils/data_utils.py
touch utils/model_utils.py
touch utils/viz_utils.py

# 6) (Optional) Create results folder for saved outputs
mkdir results

# 7) (Optional) Create virtual environment
python3 -m venv venv

# 8) Activate environment
source venv/bin/activate   # Mac/Linux
# .\venv\Scripts\activate  # Windows PowerShell

# 9) nstall dependencies
pip install streamlit scikit-learn pandas numpy matplotlib plotly

# 10) Save dependencies to requirements.txt
pip freeze > requirements.txt

# 11) Run the Streamlit app
streamlit run app.py

# 12) Deactivate environment when finished
deactivate
