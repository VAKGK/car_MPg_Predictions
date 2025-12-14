import streamlit as st
import joblib
import numpy as np
import os

# ==================== PAGE CONFIG ====================
st.set_page_config(page_title="Car MPG Predictor", page_icon="🚗", layout="wide")

# ==================== RICH, WARM & COLORFUL DESIGN ====================
st.markdown("""
<style>
    #MainMenu, footer, header, .stDeployButton {visibility: hidden !important;}
    .block-container {padding-top: 2rem !important; max-width: 90% !important;}
    hr {display: none !important;}

    /* Multi-layer warm pastel background */
    .main {
        background: linear-gradient(to bottom, 
            #FFF8E7 0%, #FFF8E7 35%, 
            #FFFBEF 35%, #FFFBEF 50%, 
            #FAD7C3 50%, #FAD7C3 65%,   /* Warm peach accent */
            #C7D9F0 65%, #C7D9F0 80%,   /* Soft lavender-blue */
            #A8DADC 80%, #A8DADC 100%); /* Original soft blue */
        min-height: 100vh;
    }

    /* Elegant warm title */
    h1 {
        font-size: 4.8rem !important;
        font-weight: 800 !important;
        text-align: center;
        background: linear-gradient(90deg, #FF7F50, #FF6B6B, #9F7AEA, #6EE7B7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem !important;
        font-family: 'Georgia', serif;
    }
    h5 {
        font-size: 1.9rem !important;
        text-align: center;
        color: #5D5D8D;
        margin-bottom: 50px !important;
        font-weight: 500;
    }

    /* Section headers with warm colors */
    .stMarkdown h4 {
        color: #E67E22 !important;  /* Warm orange */
        font-size: 1.9rem !important;
        margin-bottom: 25px;
        font-weight: 600;
    }

    /* Preset buttons - colorful and warm */
    .stButton > button {
        background: linear-gradient(135deg, #FF9A9E, #FAD0C4);
        color: #2C3E50;
        border-radius: 15px;
        border: none;
        font-size: 1.2rem;
        font-weight: 600;
        padding: 14px;
        box-shadow: 0 6px 20px rgba(255, 154, 158, 0.3);
        transition: all 0.3s;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #FAD0C4, #FF9A9E);
        transform: translateY(-5px);
        box-shadow: 0 12px 30px rgba(255, 154, 158, 0.4);
    }

    /* Main predict button - vibrant teal */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #2DD4BF, #14B8A6);
        color: white;
        font-size: 1.5rem;
        padding: 18px 50px;
        border-radius: 60px;
        box-shadow: 0 10px 35px rgba(45, 212, 191, 0.4);
        font-weight: bold;
    }
    .stButton > button[kind="primary"]:hover {
        transform: translateY(-8px);
        box-shadow: 0 20px 50px rgba(45, 212, 191, 0.5);
    }

    /* Result cards - golden & purple */
    .result-card {
        background: rgba(255,255,255,0.85);
        backdrop-filter: blur(12px);
        border-radius: 30px;
        padding: 50px 30px;
        text-align: center;
        border: 2px solid #FFE4B5;
        box-shadow: 0 20px 50px rgba(230, 126, 34, 0.15);
        margin: 40px 0;
    }
    .result-card h1 {
        font-size: 7rem !important;
        margin: 0;
        background: linear-gradient(135deg, #F59E0B, #EF4444);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 900;
    }
    .result-card h3 {
        margin: 15px 0 8px;
        color: #7C3AED;
        font-size: 2rem;
        font-weight: 600;
    }

    /* Social icons */
    div a img:hover {
        transform: translateY(-10px) scale(1.2);
        box-shadow: 0 20px 40px rgba(0,0,0,0.25);
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
        st.error("Model files not found! Please ensure scaler.joblib and car_mileage_model.joblib are in the same folder.")
        return None, None

    scaler = joblib.load(scaler_path)
    model = joblib.load(model_path)
    return scaler, model

scaler, model = load_model()
if scaler is None or model is None:
    st.stop()

# ==================== TITLE ====================
st.markdown("<h1>Car Fuel Efficiency Predictor</h1>", unsafe_allow_html=True)
st.markdown("<h5>1970–1982 Classic Cars • USA • Europe • Japan</h5>", unsafe_allow_html=True)

# ==================== PRESET BUTTONS ====================
st.markdown("### Choose a Car Style")
preset_col1, preset_col2, preset_col3, reset_col = st.columns(4)

with preset_col1:
    if st.button("🇯🇵 Japanese Economy", use_container_width=True):
        st.session_state.preset = "japan"

with preset_col2:
    if st.button("🇺🇸 American Muscle", use_container_width=True):
        st.session_state.preset = "muscle"

with preset_col3:
    if st.button("👔 Family Sedan", use_container_width=True):
        st.session_state.preset = "family"

with reset_col:
    if st.button("🔄 Reset All", use_container_width=True):
        st.session_state.clear()
        st.rerun()

# Apply presets
if st.session_state.get("preset") == "japan":
    defaults = {"cyl":4, "disp":100, "hp":70, "weight":2200, "acc":18.0, "year":80, "origin":3}
elif st.session_state.get("preset") == "muscle":
    defaults = {"cyl":8, "disp":350, "hp":200, "weight":4000, "acc":12.0, "year":74, "origin":1}
elif st.session_state.get("preset") == "family":
    defaults = {"cyl":6, "disp":200, "hp":120, "weight":3200, "acc":15.0, "year":78, "origin":1}
else:
    defaults = {"cyl":4, "disp":200, "hp":120, "weight":3000, "acc":15.0, "year":78, "origin":3}

# ==================== INPUTS ====================
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
                          help="Japanese & European cars were usually more efficient")

model_year = st.slider(
    "📅 Model Year (1970–1982)",
    min_value=70, max_value=82, value=defaults["year"],
    format="19%02d",
    help="Later years generally had better efficiency"
)

# ==================== PREDICT BUTTON ====================
if st.button("Predict Fuel Efficiency Now!", type="primary", use_container_width=True):
    with st.spinner("Analyzing your classic car..."):
        X = np.array([[cylinders, displacement, horsepower, weight, acceleration, model_year, origin]])
        prediction = model.predict(scaler.transform(X))[0]
        mpg = round(float(prediction), 1)
        km_per_liter = round(mpg * 0.425144, 1)

    st.markdown("<div style='margin:80px 0'></div>", unsafe_allow_html=True)

    col_mpg, col_kml = st.columns(2)
    with col_mpg:
        st.markdown(f"""
        <div class="result-card">
            <h1>{mpg}</h1>
            <h3>MPG</h3>
            <p style="font-size:1.2rem; color:#7C3AED;">Miles Per Gallon</p>
        </div>
        """, unsafe_allow_html=True)

    with col_kml:
        st.markdown(f"""
        <div class="result-card">
            <h1>{km_per_liter}</h1>
            <h3>km/L</h3>
            <p style="font-size:1.2rem; color:#7C3AED;">Kilometers Per Liter</p>
        </div>
        """, unsafe_allow_html=True)

    st.caption("1 MPG ≈ 0.425 km/L • Based on UCI Auto MPG Dataset")

    if mpg >= 35:
        st.success("🚀 Legendary efficiency — probably Japanese engineering at its finest!")
        st.balloons()
    elif mpg >= 28:
        st.success("✅ Excellent — great balance!")
    elif mpg >= 20:
        st.info("👍 Solid for the classic era")
    else:
        st.warning("💪 Raw American power — efficiency takes a back seat!")

# ==================== FOOTER ====================
st.markdown("---")
st.markdown("<h3 style='text-align:center; margin:50px 0; color:#E67E22;'>Big blocks, small Civics, and everything that makes car lovers smile — this one’s for you! ❤️ Arun</h3>", unsafe_allow_html=True)

st.markdown("""
<div style="display:flex; justify-content:center; gap:70px; margin:60px 0;">
    <a href="https://www.linkedin.com/in/vadlamudi-arun-kumar/" target="_blank">
        <img src="https://cdn-icons-png.flaticon.com/512/174/174857.png" width="65" height="65">
    </a>
    <a href="https://github.com/VAKGK" target="_blank">
        <img src="https://cdn-icons-png.flaticon.com/512/25/25231.png" width="65" height="65">
    </a>
    <a href="mailto:vadlamudiarunkumar3@gmail.com">
        <img src="https://cdn-icons-png.flaticon.com/512/732/732200.png" width="65" height="65">
    </a>
    <a href="https://codebasics.io/portfolio/VADLAMUDI-ARUN-KUMAR" target="_blank" title="Portfolio">
        <img src="https://cdn-icons-png.flaticon.com/128/10856/10856864.png" width="60" height="60" style="transition:0.3s;">
    </a>
    
</div>
""", unsafe_allow_html=True)

st.caption("Built with passion using Streamlit • Machine Learning • Classic Auto MPG Dataset")


