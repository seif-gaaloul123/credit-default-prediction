import streamlit as st
import joblib
import numpy as np
import re

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Credit Default Prediction",
    layout="wide"
)

# ============================================================
# LOAD MODELS
# ============================================================
@st.cache_resource
def load_models():
    try :
        model = joblib.load("artifacts/xgb_model_tfidf.pkl")
        scaler = joblib.load("artifacts/scaler.pkl")
    tfidf = joblib.load("artifacts/tfidf.pkl")
    numerical_cols = joblib.load("artifacts/numerical_cols.pkl")
    return model, scaler, tfidf, numerical_cols
    except Exception as e:
        st.error(f"Error loading models: {e}")
        st.stop()

model, scaler, tfidf, numerical_cols = load_models()

# ============================================================
# TEXT CLEANING — must match training exactly
# ============================================================
def clean_text(text):
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# ============================================================
# UI
# ============================================================
st.title("Credit Default Prediction")
st.markdown("Predict the probability of a loan defaulting using XGBoost + TF-IDF")

st.markdown("---")

col1, col2 = st.columns(2)

# ============================================================
# TEXT INPUTS
# ============================================================
with col1:
    st.subheader("Loan Text Information")
    emp_title = st.text_input("Employment Title", placeholder="e.g. Software Engineer")
    title = st.text_input("Loan Title", placeholder="e.g. Debt consolidation")
    desc = st.text_area("Loan Description", placeholder="e.g. I need this loan to pay off credit cards")
    purpose = st.selectbox("Purpose", [
        "debt_consolidation", "credit_card", "home_improvement",
        "other", "major_purchase", "medical", "small_business",
        "car", "vacation", "moving", "house", "wedding",
        "renewable_energy", "educational"
    ])

# ============================================================
# NUMERICAL INPUTS
# ============================================================
with col2:
    st.subheader("Loan & Borrower Details")
    loan_amnt = st.number_input("Loan Amount ($)", min_value=500, max_value=40000, value=10000)
    int_rate = st.slider("Interest Rate (%)", min_value=5.0, max_value=30.0, value=12.5)
    annual_inc = st.number_input("Annual Income ($)", min_value=1000, max_value=500000, value=60000)
    dti = st.slider("Debt-to-Income Ratio", min_value=0.0, max_value=50.0, value=15.0)
    fico_range_low = st.slider("FICO Score (Low)", min_value=580, max_value=850, value=680)
    revol_util = st.slider("Revolving Utilization (%)", min_value=0.0, max_value=100.0, value=30.0)
    revol_bal = st.number_input("Revolving Balance ($)", min_value=0, max_value=500000, value=5000)
    open_acc = st.slider("Open Accounts", min_value=0, max_value=50, value=10)
    total_acc = st.slider("Total Accounts", min_value=0, max_value=100, value=20)
    mort_acc = st.slider("Mortgage Accounts", min_value=0, max_value=20, value=1)
    pub_rec = st.slider("Public Records", min_value=0, max_value=10, value=0)
    pub_rec_bankruptcies = st.slider("Bankruptcies", min_value=0, max_value=5, value=0)
    delinq_2yrs = st.slider("Delinquencies (2 years)", min_value=0, max_value=20, value=0)
    inq_last_6mths = st.slider("Inquiries Last 6 Months", min_value=0, max_value=10, value=1)

st.markdown("---")

# ============================================================
# PREDICT
# ============================================================
if st.button(" Default Risk", use_container_width=True):
    # Process text
    combined_text = " ".join([
        clean_text(emp_title),
        clean_text(title),
        clean_text(desc),
        clean_text(purpose)
    ]).strip()

    text_features = tfidf.transform([combined_text])

    # Numerical features — order must match numerical_cols
    numerical_input = np.array([[
        loan_amnt, int_rate, annual_inc, dti,
        delinq_2yrs, fico_range_low, inq_last_6mths,
        open_acc, pub_rec, revol_bal, revol_util,
        total_acc, mort_acc, pub_rec_bankruptcies
    ]])

    numerical_scaled = scaler.transform(numerical_input)
    X = np.concatenate([numerical_scaled, text_features.toarray()], axis=1)
    prob = model.predict_proba(X)[:, 1][0]

    # ============================================================
    # RESULTS
    # ============================================================
    st.markdown("---")
    st.subheader(" Prediction Results")

    col3, col4, col5 = st.columns(3)

    with col3:
        st.metric("Default Probability", f"{prob*100:.1f}%")

    with col4:
        if prob < 0.3:
            risk = " LOW RISK"
            color = "green"
        elif prob < 0.6:
            risk = " MEDIUM RISK"
            color = "orange"
        else:
            risk = " HIGH RISK"
            color = "red"
        st.metric("Risk Level", risk)

    with col5:
        recommendation = "APPROVE" if prob < 0.5 else " REJECT"
        st.metric("Recommendation", recommendation)

    # Progress bar as gauge
    st.markdown("**Default Probability Gauge:**")
    st.progress(float(prob))

    # SHAP explanationp
    st.markdown("---")
    st.subheader(" SHAP Explanation")
    try:
        import shap
        import matplotlib.pyplot as plt

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)

        all_feature_names = numerical_cols + list(tfidf.get_feature_names_out())

        fig, ax = plt.subplots(figsize=(10, 6))
        shap.summary_plot(
            shap_values,
            X,
            feature_names=all_feature_names,
            show=False,
            max_display=15
        )
        st.pyplot(fig)
        plt.close()
    except Exception as e:
        st.warning(f"SHAP explanation unavailable: {e}")

# ============================================================
# MODEL INFO
# ============================================================
with st.expander(" Model Information"):
    st.markdown("""
    **Model:** XGBoost Classifier  
    **Text Representation:** TF-IDF (500 features)  
    **Hyperparameter Tuning:** Optuna (50 trials, 3-fold CV)  
    **Validation:** Temporal split (train: 2007-2016, test: 2017+)  
    **Explainability:** SHAP TreeExplainer  
    
    **Test Set Performance:**
    - ROC-AUC: 0.75
    - Recall: 0.86
    - Average Precision: 0.39
    """)