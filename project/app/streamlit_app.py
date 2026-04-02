import os
import re
import joblib
import numpy as np
import streamlit as st

st.set_page_config(
    page_title="Credit Default Prediction",
    layout="wide"
)

@st.cache_resource
def load_models():
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model = joblib.load(os.path.join(base_path, "artifacts/xgb_model_tfidf.pkl"))
    scaler = joblib.load(os.path.join(base_path, "artifacts/scaler.pkl"))
    tfidf = joblib.load(os.path.join(base_path, "artifacts/tfidf.pkl"))
    numerical_cols = joblib.load(os.path.join(base_path, "artifacts/numerical_cols.pkl"))
    return model, scaler, tfidf, numerical_cols

model, scaler, tfidf, numerical_cols = load_models()

DEFAULTS = {
    'loan_amnt': 12000,
    'funded_amnt': 12000,
    'funded_amnt_inv': 12000,
    'int_rate': 13.5,
    'installment': 370,
    'annual_inc': 65000,
    'dti': 17.0,
    'delinq_2yrs': 0,
    'fico_range_low': 690,
    'fico_range_high': 694,
    'inq_last_6mths': 1,
    'open_acc': 11,
    'pub_rec': 0,
    'revol_bal': 8000,
    'revol_util': 45.0,
    'total_acc': 22,
    'collections_12_mths_ex_med': 0,
    'acc_now_delinq': 0,
    'tot_coll_amt': 0,
    'tot_cur_bal': 50000,
    'total_rev_hi_lim': 20000,
    'acc_open_past_24mths': 4,
    'avg_cur_bal': 5000,
    'bc_open_to_buy': 5000,
    'bc_util': 55.0,
    'chargeoff_within_12_mths': 0,
    'delinq_amnt': 0,
    'mo_sin_old_il_acct': 100,
    'mo_sin_old_rev_tl_op': 120,
    'mo_sin_rcnt_rev_tl_op': 6,
    'mo_sin_rcnt_tl': 5,
    'mort_acc': 1,
    'mths_since_recent_bc': 12,
    'mths_since_recent_inq': 6,
    'num_accts_ever_120_pd': 0,
    'num_actv_bc_tl': 3,
    'num_actv_rev_tl': 5,
    'num_bc_sats': 4,
    'num_bc_tl': 6,
    'num_il_tl': 8,
    'num_op_rev_tl': 6,
    'num_rev_accts': 12,
    'num_rev_tl_bal_gt_0': 5,
    'num_sats': 11,
    'num_tl_120dpd_2m': 0,
    'num_tl_30dpd': 0,
    'num_tl_90g_dpd_24m': 0,
    'num_tl_op_past_12m': 2,
    'pct_tl_nvr_dlq': 92.0,
    'percent_bc_gt_75': 30.0,
    'pub_rec_bankruptcies': 0,
    'tax_liens': 0,
    'tot_hi_cred_lim': 60000,
    'total_bal_ex_mort': 20000,
    'total_bc_limit': 12000,
    'total_il_high_credit_limit': 30000
}

def clean_text(text):
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

st.title("Credit Default Prediction")
st.markdown("Predict the probability of a loan defaulting using XGBoost + TF-IDF")
st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Loan Information")
    emp_title = st.text_input("Employment Title", placeholder="e.g. Software Engineer")
    title = st.text_input("Loan Title", placeholder="e.g. Debt consolidation")
    desc = st.text_area("Loan Description", placeholder="e.g. I need this loan to pay off credit cards")
    purpose = st.selectbox("Purpose", [
        "debt_consolidation", "credit_card", "home_improvement",
        "other", "major_purchase", "medical", "small_business",
        "car", "vacation", "moving", "house", "wedding",
        "renewable_energy", "educational"
    ])

with col2:
    st.subheader("Financial Details")
    st.caption("Remaining credit bureau fields use training median values automatically")
    loan_amnt = st.number_input("Loan Amount ($)", min_value=500, max_value=40000, value=12000)
    int_rate = st.slider("Interest Rate (%)", min_value=5.0, max_value=30.0, value=13.5)
    annual_inc = st.number_input("Annual Income ($)", min_value=1000, max_value=500000, value=65000)
    dti = st.slider("Debt-to-Income Ratio (%)", min_value=0.0, max_value=50.0, value=17.0)
    fico_range_low = st.slider("FICO Credit Score", min_value=580, max_value=850, value=690)
    revol_util = st.slider("Credit Card Utilization (%)", min_value=0.0, max_value=100.0, value=45.0)
    open_acc = st.slider("Number of Open Accounts", min_value=0, max_value=50, value=11)
    pub_rec = st.slider("Public Derogatory Records", min_value=0, max_value=10, value=0)

st.markdown("---")

if st.button("Predict Default Risk", use_container_width=True):

    combined_text = " ".join([
        clean_text(emp_title),
        clean_text(title),
        clean_text(desc),
        clean_text(purpose)
    ]).strip()

    text_features = tfidf.transform([combined_text])

    feature_values = DEFAULTS.copy()
    feature_values['loan_amnt'] = loan_amnt
    feature_values['funded_amnt'] = loan_amnt
    feature_values['funded_amnt_inv'] = loan_amnt
    feature_values['int_rate'] = int_rate
    feature_values['annual_inc'] = annual_inc
    feature_values['dti'] = dti
    feature_values['fico_range_low'] = fico_range_low
    feature_values['fico_range_high'] = fico_range_low + 4
    feature_values['revol_util'] = revol_util
    feature_values['open_acc'] = open_acc
    feature_values['pub_rec'] = pub_rec

    numerical_input = np.array([[feature_values[col] for col in numerical_cols]])
    numerical_scaled = scaler.transform(numerical_input)
    X = np.concatenate([numerical_scaled, text_features.toarray()], axis=1)
    X = X.astype(float)
    prob = model.predict_proba(X)[:, 1][0]

    st.markdown("---")
    st.subheader("Prediction Results")

    col3, col4, col5 = st.columns(3)

    with col3:
        st.metric("Default Probability", f"{prob*100:.1f}%")

    with col4:
        if prob < 0.3:
            risk = "LOW RISK"
        elif prob < 0.6:
            risk = "MEDIUM RISK"
        else:
            risk = "HIGH RISK"
        st.metric("Risk Level", risk)

    with col5:
        recommendation = "APPROVE" if prob < 0.5 else "REJECT"
        st.metric("Recommendation", recommendation)

    st.markdown("**Default Probability Gauge:**")
    st.progress(float(prob))

    st.markdown("---")
    st.subheader("SHAP Explanation")
    try:
        import shap
        import matplotlib.pyplot as plt
        X_shap = np.array(X, dtype=np.float64)

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        all_feature_names = list(numerical_cols) + list(tfidf.get_feature_names_out())

        fig, ax = plt.subplots(figsize=(10, 6))
        shap.summary_plot(
            shap_values, X,
            feature_names=all_feature_names,
            show=False,
            max_display=15
        )
        st.pyplot(fig)
        plt.close()
    except Exception as e:
        st.warning(f"SHAP explanation unavailable: {e}")

with st.expander("Model Information"):
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

    **Note:** App uses 8 key inputs. Remaining credit bureau 
    features use training data median values automatically.
    """)