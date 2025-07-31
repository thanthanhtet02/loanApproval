
import joblib
import pandas as pd
import numpy as np
try:
    from streamlit_extras.colored_header import colored_header
    from streamlit_extras.let_it_rain import rain
except ImportError:
    def colored_header(*args, **kwargs):
        pass
    def rain(*args, **kwargs):
        pass

# === Load Trained Model, Scaler, and Features ===
model = joblib.load("gb_best_model.pkl")
scaler = joblib.load("scaler.pkl")
features = joblib.load("features.pkl")

import streamlit as st
# === Page Layout & Sidebar ===
st.set_page_config(page_title="Loan Approval Predictor", layout="wide", page_icon="💰")

# === Modern Colorful Background ===
st.markdown(
    """
    <style>
    body {
        background: linear-gradient(120deg, #e0eafc 0%, #cfdef3 100%) !important;
    }
    .modern-card {
        background: linear-gradient(120deg, #f8fffa 60%, #e0f7fa 100%);
        border-radius: 18px;
        box-shadow: 0 4px 24px 0 rgba(60,72,88,.10);
        padding: 2.2rem 2.2rem 1.5rem 2.2rem;
        margin-bottom: 1.5rem;
        border: 1.5px solid #b2ebf2;
    }
    .modern-header {
        font-size: 1.35em;
        font-weight: 700;
        color: #0097a7;
        margin-bottom: 0.7em;
        letter-spacing: -0.5px;
        display: flex;
        align-items: center;
        gap: 0.5em;
    }
    </style>
    """,
    unsafe_allow_html=True
)


colored_header("Loan Approval Prediction App", description="Professional, fast, and secure loan eligibility check.", color_name="blue-green-70")

# === Input Fields ===
st.markdown("---")
st.markdown("### 👤 Applicant Information")

# === Modern Card-Style Input Sections ===
st.markdown("""
<div class='modern-card'>
  <div class='modern-header'>👤 Applicant Information</div>
</div>
""", unsafe_allow_html=True)
col1, col2, col3 = st.columns([1,1,1])
user_input = {}

def cap(val, minval, maxval):
    try:
        v = float(val)
    except Exception:
        v = minval
    return min(max(v, minval), maxval)

with col1:
    st.markdown("""
    <div class='modern-card'>
      <div class='modern-header'>📝 Personal Details</div>
    """, unsafe_allow_html=True)
    age = st.slider("Age", 18, 75, 30, help="Applicant's age in years.")
    emp_exp = st.slider("Employment Experience (Years)", 0, 50, 5, help="Total years of employment.")
    gender = st.selectbox("Gender", ["female", "male"], help="Applicant's gender.")
    education = st.selectbox("Education Level", ["Associate", "Bachelor", "Doctorate", "High School", "Master"], help="Highest education attained.")
    # Use capped values for model
    user_input['person_age'] = cap(age, 18, 75)
    user_input['person_emp_exp'] = cap(emp_exp, 0, 50)

with col2:
    st.markdown("""
    <div class='modern-card'>
      <div class='modern-header'>💵 Financial Details</div>
    """, unsafe_allow_html=True)
    income_raw = st.text_input("Annual Income (SGD)", value="50000", help="Gross annual income in SGD.")
    loan_amnt_raw = st.text_input("Loan Amount Requested (SGD)", value="10000", help="Requested loan amount.")
    int_rate_raw = st.text_input("Loan Interest Rate (%)", value="10.0", help="Interest rate for the loan.")
    cred_hist_raw = st.text_input("Credit History Length (Years)", value="6", help="Years of credit history.")
    credit_score_raw = st.text_input("Credit Score", value="700", help="Applicant's credit score.")

    # Use capped values for model
    income = cap(income_raw, 0, 300000)
    loan_amnt = cap(loan_amnt_raw, 0, 35000)
    int_rate = cap(int_rate_raw, 0, 20.0)
    cred_hist = cap(cred_hist_raw, 0, 20)
    credit_score = cap(credit_score_raw, 300, 850)
    user_input['person_income'] = income
    user_input['loan_amnt'] = loan_amnt
    user_input['loan_int_rate'] = int_rate
    user_input['cb_person_cred_hist_length'] = cred_hist
    user_input['credit_score'] = credit_score

with col3:
    st.markdown("""
    <div class='modern-card'>
      <div class='modern-header'>🏠 Other Information</div>
    """, unsafe_allow_html=True)
    home = st.selectbox("Home Ownership", ["MORTGAGE", "OTHER", "OWN", "RENT"], help="Current home ownership status.")
    intent = st.selectbox("Loan Purpose", ["EDUCATION", "HOMEIMPROVEMENT", "MEDICAL", "PERSONAL", "VENTURE", "DEBTCONSOLIDATION"], help="Purpose of the loan.")
    default = st.selectbox("Previous Loan Default", ["No", "Yes"], help="Any previous loan defaults?")

st.markdown("""
</div></div></div>
<hr style='border: none; border-top: 2px solid #b2ebf2; margin: 2em 0 1em 0;'>
""", unsafe_allow_html=True)

# === Validate income
if income <= 0:
    st.error("⚠️ Annual Income must be greater than 0 for prediction to work.")
    st.stop()

# === Feature Engineering ===
loan_percent_income = min(user_input['loan_amnt'] / income, 0.6)
user_input['loan_percent_income'] = loan_percent_income
user_input['aggressive_loan_flag'] = int((loan_percent_income > 0.3) and (user_input['loan_int_rate'] > 15))
user_input['risk_inversion_score'] = (user_input['loan_int_rate'] * loan_percent_income) / (user_input['credit_score'] + 1)

# === One-Hot Encoding based on feature list ===
for col in features:
    if col.startswith("person_gender_"):
        user_input[col] = int(f"person_gender_{gender}" == col)
    elif col.startswith("person_education_"):
        user_input[col] = int(f"person_education_{education}" == col)
    elif col.startswith("person_home_ownership_"):
        user_input[col] = int(f"person_home_ownership_{home}" == col)
    elif col.startswith("loan_intent_"):
        user_input[col] = int(f"loan_intent_{intent}" == col)

user_input['previous_loan_defaults_on_file_Yes'] = int(default == "Yes")

# === Income Bucket OHE ===
bucket = pd.cut([income], bins=[0, 25000, 50000, 100000, np.inf], labels=['Low', 'Mid', 'High', 'Very High'])[0]
user_input['income_bucket_Mid'] = int(bucket == "Mid")
user_input['income_bucket_High'] = int(bucket == "High")
user_input['income_bucket_Very High'] = int(bucket == "Very High")

# === Prepare Final Input DataFrame ===
input_df = pd.DataFrame([user_input])
input_df = input_df.reindex(columns=features, fill_value=0).astype(float)

import streamlit as st
import joblib
import pandas as pd
import numpy as np


# === Predict Button ===
st.markdown("---")
st.markdown("### 📊 Prediction Result")
if st.button("🔮 Predict Loan Approval", help="Click to get your loan approval result!"):
    try:
        #  Apply Scaler Before Prediction
        scaled_input = scaler.transform(input_df)

        prediction = model.predict(scaled_input)[0]
        prob = model.predict_proba(scaled_input)[0][1]

        # --- Recommendation & Risk Factors ---
        risk_factors = []
        recommendations = []
        # Simple logic for demo; you can expand this
        if user_input['credit_score'] < 650:
            risk_factors.append("Low credit score")
            recommendations.append("Work on improving your credit score before applying")
        if user_input['loan_percent_income'] > 0.3:
            risk_factors.append("High loan amount relative to income")
            recommendations.append("Consider reducing the requested loan amount")
        if user_input['previous_loan_defaults_on_file_Yes'] == 1:
            risk_factors.append("Previous loan defaults on file")
            recommendations.append("Build a positive repayment history")
        if user_input['person_income'] < 25000:
            risk_factors.append("Low annual income")
            recommendations.append("Increase your income to improve approval chances")
        if user_input['person_emp_exp'] < 2:
            risk_factors.append("Short employment history")
            recommendations.append("Gain more work experience")

        # --- Card Style Result ---
        card_style = """
        border-radius: 28px;
        border-top: 7px solid {border_color};
        background: rgba(255,255,255,0.75);
        backdrop-filter: blur(8px) saturate(1.2);
        box-shadow: 0 8px 40px 0 rgba(60,72,88,.18), 0 1.5px 8px 0 rgba(33,150,83,0.07);
        padding: 2.8rem 3.2rem 2.5rem 3.2rem;
        margin-top: 2.5rem;
        margin-bottom: 2.5rem;
        transition: box-shadow 0.3s, border-color 0.3s;
        font-family: 'Segoe UI', 'Roboto', 'Arial', sans-serif;
        position: relative;
        """
        emoji_approve = "<div style='font-size:3.5em;line-height:1;margin-bottom:0.2em;text-align:center;'>💸</div>"
        emoji_reject = "<div style='font-size:3.5em;line-height:1;margin-bottom:0.2em;text-align:center;'>🚫</div>"
        icon_approve = "<span style='font-size:2.5em;color:#219653;vertical-align:middle;'>&#10003;</span>"
        icon_reject = "<span style='font-size:2.5em;color:#d32f2f;vertical-align:middle;'>&#10007;</span>"

        if prediction == 1:
            st.markdown(f"""
                <div style='{card_style.format(border_color="#219653")}'>
                    {emoji_approve}
                    <div style='display:flex;align-items:center;gap:1.1em;margin-bottom:0.7em;justify-content:center;'>
                        {icon_approve}
                        <span style='font-size:2.2em;font-weight:800;color:#219653;letter-spacing:-1px;'>Loan Likely to be Approved!</span>
                    </div>
                    <div style='margin-top:0.2em;font-size:1.18em;color:#444;text-align:center;'><b>Confidence:</b> <span style='color:#219653'>{prob:.1%}</span></div>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div style='{card_style.format(border_color="#d32f2f")}'>
                    {emoji_reject}
                    <div style='display:flex;align-items:center;gap:1.1em;margin-bottom:0.7em;justify-content:center;'>
                        {icon_reject}
                        <span style='font-size:2.2em;font-weight:800;color:#d32f2f;letter-spacing:-1px;'>Loan Likely to be Rejected</span>
                    </div>
                    <div style='margin-top:0.2em;font-size:1.18em;color:#444;text-align:center;'><b>Confidence:</b> <span style='color:#d32f2f'>{1-prob:.1%}</span></div>
                    <div style='margin-top:1.3em;text-align:center;'><b style='color:#d32f2f;font-size:1.1em;'>Risk Factors:</b></div>
                    <ul style='margin:0.7em auto 0 auto;padding:0 0 0 1.2em;color:#d32f2f;font-size:1.13em;max-width:420px;'>
                        {''.join([f'<li>{risk}</li>' for risk in (risk_factors or ['Profile does not meet approval criteria'])])}
                    </ul>
                    <div style='margin-top:1.3em;text-align:center;'><b style='color:#1a237e;font-size:1.1em;'>Recommendations:</b></div>
                    <ul style='margin:0.7em auto 0 auto;padding:0 0 0 1.2em;color:#444;font-size:1.13em;max-width:420px;'>
                        {''.join([f'<li>{rec}</li>' for rec in (recommendations or ['Review your application details and try again'])])}
                    </ul>
                </div>
            """, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Prediction error: {e}")


st.markdown("---")
st.caption("Built with Streamlit • Model: Gradient Boosting • Scaled + One-Hot Encoded Features •")
