import streamlit as st
import pandas as pd
from utils import data_utils, model_utils, viz_utils

st.set_page_config(page_title="Ridge Regression Playground", layout="wide")
st.title("📊 Ridge Regression Playground")
st.write("An interactive ML tool for students, builders, and researchers.")

# 1️⃣ Load dataset
st.header("1️⃣ Load your dataset")
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # validate CSV
    is_valid, msg = data_utils.validate_csv(df)
    st.info(msg)
    if not is_valid:
        st.stop()

    st.session_state["data"] = df
    st.subheader("Data Preview")
    st.dataframe(df.head())
    st.subheader("Summary Statistics")
    st.write(data_utils.summarize_data(df))

# 2️⃣ Train Ridge Regression (Default)
if "data" in st.session_state:
    st.header("2️⃣ Train Ridge Regression (Default)")
    if st.button("Run Default Model"):
        metrics, model = model_utils.train_default(st.session_state["data"])
        st.json(metrics)
        st.session_state["model"] = model

        # enriched CSV with predictions
        enriched_df = model_utils.add_predictions(model, st.session_state["data"])
        st.download_button(
            label="📥 Download Enriched CSV with Predictions",
            data=enriched_df.to_csv(index=False),
            file_name="enriched_dataset.csv",
            mime="text/csv"
        )

# 3️⃣ Hypertune Ridge Regression
if "data" in st.session_state:
    st.header("3️⃣ Hypertune Ridge Regression")
    if st.button("Run Hyperparameter Tuning"):
        best_params, best_score, tuned_model = model_utils.tune_ridge(st.session_state["data"])
        st.write(f"Best Params: {best_params}, MSE: {best_score:.4f}")
        st.session_state["model"] = tuned_model

        # enriched CSV with predictions
        enriched_df = model_utils.add_predictions(tuned_model, st.session_state["data"])
        st.download_button(
            label="📥 Download Enriched CSV with Predictions",
            data=enriched_df.to_csv(index=False),
            file_name="enriched_dataset.csv",
            mime="text/csv"
        )

# 4️⃣ Cross Validation
if "data" in st.session_state:
    st.header("4️⃣ Cross-Validation")
    if st.button("Run CV on Current Model"):
        cv_results = model_utils.cross_validate(st.session_state["data"])
        st.write(cv_results)

# 5️⃣ Visualize Performance
if "data" in st.session_state and "model" in st.session_state:
    st.header("5️⃣ Visualize Model Performance")
    if st.button("Show Plots"):
        fig1 = viz_utils.plot_predictions(st.session_state["model"], st.session_state["data"])
        st.pyplot(fig1)
        fig2 = viz_utils.plot_residuals(st.session_state["model"], st.session_state["data"])
        st.pyplot(fig2)
