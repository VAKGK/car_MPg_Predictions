import streamlit as st
import joblib
import numpy as np
import os

# ==================== PAGE CONFIG - FULL WIDTH ====================
st.set_page_config(page_title="Car MPG Predictor", page_icon="🚗", layout="wide")

# ==================== CLEAN & ANIMATED UI ====================
st.markdown("""
<style>
    #MainMenu, footer, header, .stDeployButton {visibility: hidden !important;}
    .block-container {padding-top: 1rem !important;}
    hr {display: none !important;}

    .animated-title {
        font-size: 300px;
        font-weight: 900;
        text-align: center;
        background: linear-gradient(90deg, #7c3aed, #4f46e5);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        position: relative;
        overflow: hidden;
    }
    .animated-title::after {
        content: "";
        position: absolute;
        top: 0; left: -100%; width: 100%; height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.4), transparent);
        animation: shine 3s infinite;
    }
    @keyframes shine { 0% {left: -100%;} 50%,100% {left: 100%;} }

    .stButton>button {
        background: linear-gradient(135deg, #4f46e5, #9333ea);
        color: white; border-radius: 12px; padding: 12px 25px;
        font-size: 18px; border: none; box-shadow: 0 6px 20px rgba(90,50,200,0.25);
        animation: floatBtn 3s ease-in-out infinite;
    }
    @keyframes floatBtn { 0%,100% {transform: translateY(0);} 50% {transform: translateY(-5px);} }
    .stButton>button:hover {
        transform: translateY(-8px) scale(1.03);
        box-shadow: 0 15px 35px rgba(90,50,200,0.4);
    }

    .result-card {
        background: rgba(255,255,255,0.15); backdrop-filter: blur(15px);
        border-radius: 30px; padding: 50px 20px; text-align: center;
        border: 1px solid rgba(255,255,255,0.3);
        box-shadow: 0 15px 40px rgba(0,0,0,0.3);
        margin: 20px 0;
    }
</style>
""", unsafe_allow_html=True)

# ==================== LOAD MODEL ====================
@st.cache_resource
def load_model():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    scaler_path = os.path.join(BASE_DIR, "scaler.joblib")
    model_path = os.path.join(BASE_DIR, "car_mileage_model.joblib")

    if not os.path.exists(scaler_path) or not os.path.exists(model_path):
        st.error("Model files not found! Make sure scaler.joblib and car_mileage_model.joblib are in the same folder.")
        return None, None

    scaler = joblib.load(scaler_path)
    model = joblib.load(model_path)
    return scaler, model

scaler, model = load_model()
if scaler is None or model is None:
    st.stop()

# ==================== TITLE ====================
st.markdown("<h1 class='animated-title'>Car Fuel Efficiency Predictor</h1>", unsafe_allow_html=True)
st.markdown("<h5 style='text-align:center; color:#6b7280; margin-bottom:40px;'>1970–1982 Classic Cars</h5>", unsafe_allow_html=True)

# ==================== PRESET BUTTONS ====================
st.markdown("")
preset_col1, preset_col2, preset_col3, reset_col = st.columns([1,1,1,1])

with preset_col1:
    if st.button("🇯🇵 Economy Car", use_container_width=True):
        st.session_state.preset = "economy"

with preset_col2:
    if st.button("🇺🇸 Muscle Car", use_container_width=True):
        st.session_state.preset = "muscle"

with preset_col3:
    if st.button("👨‍👩‍👧‍👦 Family Sedan", use_container_width=True):
        st.session_state.preset = "family"

with reset_col:
    if st.button("🔄 Reset", use_container_width=True):
        st.session_state.clear()
        st.rerun()

# Preset defaults
if st.session_state.get("preset") == "economy":
    defaults = {"cyl":4, "disp":100, "hp":70, "weight":2200, "acc":18.0, "year":80, "origin":3}
elif st.session_state.get("preset") == "muscle":
    defaults = {"cyl":8, "disp":350, "hp":200, "weight":4000, "acc":12.0, "year":74, "origin":1}
elif st.session_state.get("preset") == "family":
    defaults = {"cyl":6, "disp":200, "hp":120, "weight":3200, "acc":15.0, "year":78, "origin":1}
else:
    defaults = {"cyl":4, "disp":200, "hp":120, "weight":3000, "acc":15.0, "year":78, "origin":3}

# ==================== INPUTS WITH EMOJIS ====================
col1, col2 = st.columns(2)

with col1:
    st.markdown("#### 🔥 Engine & Power")
    cylinders = st.number_input("🔩 Number of Cylinders", 3, 8, defaults["cyl"], step=1,
                                help="Most cars: 4 | Muscle/V8: 8")
    displacement = st.number_input("🛢️ Engine Displacement (cu in)", 68.0, 455.0, float(defaults["disp"]), step=10.0,
                                   help="Small engine: 70–150 | Big V8: 300–455")
    horsepower = st.number_input("🐎 Horsepower (HP)", 40, 250, defaults["hp"], step=5,
                                 help="Higher HP usually means lower fuel efficiency")

with col2:
    st.markdown("#### 🚀 Weight & Performance")
    weight = st.number_input("⚖️ Car Weight (lbs)", 1500, 6000, defaults["weight"], step=100,
                             help="Light: 2000–2800 | Heavy: 4000+")
    acceleration = st.number_input("🏁 0–60 mph (seconds)", 8.0, 30.0, defaults["acc"], step=0.5,
                                   help="Fast: 8–12 sec | Slow: 18+ sec")
    origin = st.selectbox("🌍 Country of Origin", [1, 2, 3],
                          format_func=lambda x: {1: "🇺🇸 USA", 2: "🇪🇺 Europe", 3: "🇯🇵 Japan"}[x],
                          index=defaults["origin"]-1,
                          help="Japanese & European cars were usually more efficient in this era")

model_year = st.slider(
    "📅 Model Year",
    min_value=70,
    max_value=82,
    value=defaults["year"],
    step=1,
    format="19%02d",
    help="Newer models (late 70s–80s) generally have better fuel efficiency due to regulations"
)

# ==================== PREDICT BUTTON ====================
if st.button("Predict Fuel Efficiency Now!", type="primary", use_container_width=True):
    with st.spinner("Calculating fuel efficiency..."):
        X = np.array([[cylinders, displacement, horsepower, weight, acceleration, model_year, origin]])
        prediction = model.predict(scaler.transform(X))[0]
        mpg = round(float(prediction), 1)
        km_per_liter = round(mpg * 0.425144, 1)

    st.markdown("<div style='margin:60px 0'></div>", unsafe_allow_html=True)

    col_mpg, col_kml = st.columns(2)
    with col_mpg:
        st.markdown(f"""
        <div class="result-card" style="background: linear-gradient(135deg, #5b21b6, #7c3aed);">
            <h1 style="font-size:70px; margin:0; color:white;">{mpg}</h1>
            <h3 style="margin:10px 0 0; color:white; opacity:0.9;">MPG</h3>
            <p style="margin:5px 0 0; font-size:15px; color:white;">Miles Per Gallon</p>
        </div>
        """, unsafe_allow_html=True)

    with col_kml:
        st.markdown(f"""
        <div class="result-card" style="background: linear-gradient(135deg, #dc2626, #f97316);">
            <h1 style="font-size:70px; margin:0; color:white;">{km_per_liter}</h1>
            <h3 style="margin:10px 0 0; color:white; opacity:0.9;">km/L</h3>
            <p style="margin:5px 0 0; font-size:15px; color:white;">Kilometers Per Liter</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='margin:30px 0'></div>", unsafe_allow_html=True)
    st.caption("🔄 1 MPG ≈ 0.425 km/L | Based on UCI Auto MPG Dataset")

    if mpg >= 35:
        st.success("🚀 Outstanding! Most likely a Japanese economy car — sips fuel!")
        st.balloons()
    elif mpg >= 28:
        st.success("✅ Excellent efficiency for the era!")
    elif mpg >= 20:
        st.info("👍 Solid balance of power and economy")
    else:
        st.warning("💪 Classic American muscle — drinks gas, but sounds amazing!")

# ==================== FOOTER WITH NETWORKING (USING STREAMLIT COLUMNS & IMAGES) ====================
# ==================== FOOTER WITH ONLY CLICKABLE LOGOS (NO LABELS) ====================
st.markdown("---")
st.markdown("<h3 style='text-align:center; margin-bottom:30px;'>Big blocks, small Civics, and everything that makes car lovers smile — this one’s for you! ❤️ Arun</h3>", unsafe_allow_html=True)

# Social Icons Row
st.markdown("""
<div style="display:flex; justify-content:center; gap:60px; margin:40px 0;">
    <a href="https://www.linkedin.com/in/vadlamudi-arun-kumar/" target="_blank" title="LinkedIn" style="text-decoration:none;">
        <img src="https://cdn-icons-png.flaticon.com/512/174/174857.png" width="60" height="60" style="transition:0.3s;">
    </a>
    <a href="https://github.com/VAKGK" target="_blank" title="GitHub" style="text-decoration:none;">
        <img src="https://cdn-icons-png.flaticon.com/512/25/25231.png" width="60" height="60" style="transition:0.3s;">
    </a>
    <a href="vadlamudiarunkumar3@gmail.com" title="Email" style="text-decoration:none;">
        <img src="https://cdn-icons-png.flaticon.com/512/732/732200.png" width="60" height="60" style="transition:0.3s;">
    </a>
    <a href="https://codebasics.io/portfolio/VADLAMUDI-ARUN-KUMAR target="_blank" title="Portfolio" style="text-decoration:none;">
        <img src="D:\DATA SCIENCE\ML\PROJECTS\MPG_REG\assets\pf.png" width="60" height="60" style="transition:0.3s;">
    </a>
</div>

<style>
    div a img:hover {
        transform: translateY(-10px) scale(1.2);
        box-shadow: 0 20px 40px rgba(0,0,0,0.4);
    }
</style>
""", unsafe_allow_html=True)

st.caption("Built with Streamlit • Linear Regression Model •MPG Dataset")
