import streamlit as st
import joblib
import numpy as np

# ==================== PAGE CONFIG ====================
st.set_page_config(page_title="Customer Churn Predictor", page_icon="🏦", layout="wide")

# ==================== LOAD MODEL ====================
@st.cache_resource
def load_model():
    scaler = joblib.load("scaler.joblib")
    scaler.feature_names_in_ = None
    model = joblib.load("Churn_Prediction.joblib")
    return scaler, model

scaler, model = load_model()

# ==================== PROFESSIONAL NAVY & ORANGE THEME (ALL TEXT WHITE) ====================
st.markdown("""
<style>
    .stApp {
        background: #001f3d;
        color: #f0f0f0;
    }
    .big-title {
        font-size: 3.8rem !important;
        font-weight: 800;
        text-align: center;
        background: linear-gradient(90deg, #ff8c42, #ff6b35);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        white-space: nowrap;
        margin: 30px 0;
    }
    .subtitle {
        text-align: center;
        font-size: 1.4rem;
        color: #ffd7ba;
        margin-bottom: 50px;
    }
    .stButton>button {
        background: linear-gradient(135deg, #ff8c42, #ff6b35);
        color: white !important;
        border-radius: 50px;
        padding: 14px 40px;
        font-weight: bold;
        border: none;
        box-shadow: 0 8px 25px rgba(255, 140, 66, 0.4);
        transition: all 0.3s;
    }
    .stButton>button:hover {
        transform: translateY(-4px);
        box-shadow: 0 15px 35px rgba(255, 140, 66, 0.6);
    }
    .result-box {
        padding: 40px;
        border-radius: 25px;
        text-align: center;
        font-size: 32px;
        font-weight: bold;
        margin: 40px 0;
        box-shadow: 0 15px 40px rgba(0,0,0,0.3);
        color: white;
    }
    .help-box {
        background: rgba(255, 215, 186, 0.15);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 140, 66, 0.3);
        color: #ffd7ba;
        padding: 20px;
        border-radius: 18px;
        margin: 15px 0;
        box-shadow: 0 8px 25px rgba(0,0,0,0.2);
        font-size: 17px;
        transition: all 0.3s;
    }
    .help-box:hover {
        transform: translateY(-8px);
        box-shadow: 0 20px 40px rgba(255, 140, 66, 0.3);
        background: rgba(255, 215, 186, 0.25);
    }

    /* ALL LABELS & INPUT TEXT WHITE */
    label, .stMarkdown, .stSelectbox > div > div > div > div > div:first-child,
    .stNumberInput > div > div > label,
    .stSlider > div > div > div > label,
    .stRadio > div > label,
    .stExpander > div > label,
    .streamlit-expanderHeader {
        color: white !important;
        font-weight: 600 !important;
    }

    /* NUMBER INPUT TEXT */
    .stNumberInput > div > div > input {
        color: white !important;
        background-color: rgba(255, 255, 255, 0.1) !important;
    }

    /* RADIO BUTTON OPTION TEXT (Yes/No, Male/Female) - THIS IS THE FIX */
    .stRadio > div[role="radiogroup"] > label > div[data-testid="stMarkdownContainer"] > p {
        color: white !important;
        font-weight: 500 !important;
    }

    /* SLIDER VALUE */
    .stSlider > div > div > div > div > div > div {
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)

# ==================== TITLE ====================
st.markdown("<h1 class='big-title'>Customer Churn Predictor</h1>", unsafe_allow_html=True)
st.markdown('<p class="subtitle">Predict if a customer will stay or leave the bank</p>', unsafe_allow_html=True)

# ==================== QUICK PROFILES ====================
st.markdown("### 🚀 Quick Test Profiles")
col1, col2 = st.columns(2)
with col1:
    if st.button("✅ Safe Customer (Will Stay)", use_container_width=True):
        st.session_state.profile = "safe"
with col2:
    if st.button("⚠️ Risky Customer (Will Leave)", use_container_width=True):
        st.session_state.profile = "risky"

# Defaults
if st.session_state.get("profile") == "safe":
    defaults = {"credit":720, "age":42, "tenure":8, "balance":50000.0, "products":2,
                "country":"France", "card":"Yes", "active":"Yes", "salary":80000.0, "gender":"Male"}
elif st.session_state.get("profile") == "risky":
    defaults = {"credit":376, "age":29, "tenure":4, "balance":115046.0, "products":4,
                "country":"Germany", "card":"Yes", "active":"No", "salary":119346.0, "gender":"Female"}
else:
    defaults = {"credit":650, "age":38, "tenure":5, "balance":0.0, "products":1,
                "country":"France", "card":"Yes", "active":"Yes", "salary":50000.0, "gender":"Male"}

# ==================== INPUT FORM ====================
with st.form("churn_form"):
    c1, c2 = st.columns(2)
    with c1:
        credit_score = st.slider("💳 Credit Score", 350, 900, defaults["credit"])
        age = st.slider("🎂 Age", 18, 95, defaults["age"])
        tenure = st.slider("📅 Tenure (years)", 0, 10, defaults["tenure"])
        balance = st.number_input("💰 Balance ($)", min_value=0.0, max_value=250000.0,
                                  value=defaults["balance"], step=1000.0)
        num_products = st.selectbox("🏦 Number of Products", [1, 2, 3, 4], index=defaults["products"]-1)

    with c2:
        country = st.selectbox("🌍 Country", ["France", "Spain", "Germany"],
                              index=["France", "Spain", "Germany"].index(defaults["country"]))
        gender = st.radio("👤 Gender", ["Male", "Female"], index=0 if defaults["gender"] == "Male" else 1)
        has_cr_card = st.radio("💳 Has Credit Card?", ["Yes", "No"], index=0 if defaults["card"] == "Yes" else 1)
        is_active = st.radio("✅ Is Active Member?", ["Yes", "No"], index=0 if defaults["active"] == "Yes" else 1)
        salary = st.number_input("💵 Estimated Salary ($)", min_value=0.0, max_value=200000.0,
                                 value=defaults["salary"], step=1000.0)

    submitted = st.form_submit_button("🔮 Predict Churn Risk", use_container_width=True)

# ==================== PREDICTION ====================
if submitted:
    with st.spinner("Analyzing customer behavior..."):
        has_card = 1 if has_cr_card == "Yes" else 0
        active = 1 if is_active == "Yes" else 0
        gender_male = 1 if gender == "Male" else 0
        germany = 1 if country == "Germany" else 0
        spain = 1 if country == "Spain" else 0

        features = np.array([[credit_score, age, tenure, balance, num_products,
                              has_card, active, salary, germany, spain, gender_male]])
        scaled = scaler.transform(features)
        prediction = model.predict(scaled)[0]
        prob = model.predict_proba(scaled)[0]

        if prediction == 1:
            st.markdown(f"<div class='result-box' style='background:linear-gradient(135deg, #dc2626, #ef4444);'>HIGH RISK: Customer is likely to LEAVE!<br>Churn Probability: <b>{prob[1]:.1%}</b></div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='result-box' style='background:linear-gradient(135deg, #10b981, #34d399);'>LOW RISK: Customer is likely to STAY!<br>Stay Probability: <b>{prob[0]:.1%}</b></div>", unsafe_allow_html=True)

        st.progress(prob[1])
        st.caption(f"Churn Risk: {prob[1]:.1%}  |  Stay Chance: {prob[0]:.1%}")
        st.balloons()

# ==================== HELP SECTION ====================
with st.expander("❓ Need Help Understanding the Fields?", expanded=False):
    st.markdown("""
    <div class="help-box">
        <b>💳 Credit Score</b>: How trustworthy the customer is (350 = bad, 900 = excellent)
    </div>
    <div class="help-box">
        <b>🎂 Age</b>: Customer age (18–95)
    </div>
    <div class="help-box">
        <b>📅 Tenure</b>: Years with the bank
    </div>
    <div class="help-box">
        <b>💰 Balance</b>: Account balance — high + inactive = danger
    </div>
    <div class="help-box">
        <b>🏦 Number of Products</b>: 4 products + Germany = very high risk!
    </div>
    <div class="help-box">
        <b>💳 Has Credit Card?</b>: Has bank's credit card
    </div>
    <div class="help-box">
        <b>✅ Is Active Member?</b>: Uses account regularly? (No = red flag)
    </div>
    <div class="help-box">
        <b>💵 Estimated Salary</b>: Yearly income
    </div>
    <div class="help-box">
        <b>🌍 Country</b>: Germany has highest churn
    </div>
    <div class="help-box">
        <b>👤 Gender</b>: Female slightly higher churn rate
    </div>
    <div style="background:rgba(255, 140, 66, 0.2); padding:20px; border-radius:15px; margin-top:20px; text-align:center; font-weight:bold; color:#ff8c42;">
        🔥 Highest Risk: Germany + 4 Products + Female + Inactive + High Balance
    </div>
    """, unsafe_allow_html=True)

# ==================== FOOTER WITH ONLY CLICKABLE ICONS ====================
st.markdown("---")
st.markdown("<h3 style='text-align:center; color:#ffd7ba; margin-bottom:30px;'>Built with ❤️ by Arun</h3>", unsafe_allow_html=True)

st.markdown("""
<div style="display:flex; justify-content:center; gap:60px; margin:40px 0;">
    <a href="https://www.linkedin.com/in/vadlamudi-arun-kumar/" target="_blank" title="LinkedIn">
        <img src="https://cdn-icons-png.flaticon.com/512/174/174857.png" width="60" height="60" style="transition:0.3s;">
    </a>
    <a href="https://github.com/VAKGK" target="_blank" title="GitHub">
        <img src="https://cdn-icons-png.flaticon.com/512/25/25231.png" width="60" height="60" style="transition:0.3s;">
    </a>
    <a href="mailto:vadlamudiarunkumar3@gmail.com" title="Email">
        <img src="https://cdn-icons-png.flaticon.com/512/732/732200.png" width="60" height="60" style="transition:0.3s;">
    </a>
    <a href="https://codebasics.io/portfolio/VADLAMUDI-ARUN-KUMAR" target="_blank" title="Portfolio">
        <img src="https://cdn-icons-png.flaticon.com/512/2913/2913623.png" width="60" height="60" style="transition:0.3s;">
    </a>
</div>

<style>
    div a img:hover {
        transform: translateY(-12px) scale(1.2);
        filter: brightness(1.3);
    }
</style>
""", unsafe_allow_html=True)

st.caption("Built with Streamlit • Random Forest Model • Real Banking Insights")
