import streamlit as st
import joblib
import numpy as np
import os

# ==================== CLEAN & ANIMATED UI ====================
st.markdown("""
<style>
    #MainMenu, footer, header, .stDeployButton {visibility: hidden !important;}
    .block-container {padding-top: 2rem !important;}
    hr {display: none !important;}

    body {
        background: linear-gradient(135deg, #eef2ff, #e0e7ff);
    }

    .animated-title {
        font-size: 48px;
        font-weight: 800;
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
        font-size: 20px; border: none; box-shadow: 0 6px 20px rgba(90,50,200,0.25);
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

# ==================== LOAD MODEL – FIXED FOR STREAMLIT CLOUD ====================
@st.cache_resource
def load_model():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    scaler_path = os.path.join(BASE_DIR, "scaler.joblib")
    model_path = os.path.join(BASE_DIR, "car_mileage_model.joblib")

    if not os.path.exists(scaler_path):
        st.error(f"Scaler file not found! Expected at: {scaler_path}")
        st.info("Make sure both `scaler.joblib` and `car_mileage_model.joblib` are in the same folder as app.py")
        return None, None
    if not os.path.exists(model_path):
        st.error(f"Model file not found! Expected at: {model_path}")
        st.info("Make sure both `scaler.joblib` and `car_mileage_model.joblib` are in the same folder as app.py")
        return None, None

    scaler = joblib.load(scaler_path)
    model = joblib.load(model_path)
    return scaler, model

scaler, model = load_model()
if scaler is None or model is None:
    st.stop()

# ==================== PAGE CONFIG ====================
st.set_page_config(page_title="Car MPG Predictor", page_icon="🚗", layout="centered")

# ==================== TITLE ====================
st.markdown("<h1 class='animated-title'>Car Fuel Efficiency Predictor</h1>", unsafe_allow_html=True)
st.markdown("<h5 style='text-align:center; color:#6b7280; margin-bottom:50px;'>1970–1982 Classic Cars</h5>", unsafe_allow_html=True)

# ==================== INPUTS ====================
col1, col2 = st.columns(2)

with col1:
    st.markdown("#### Engine & Power")
    cylinders = st.number_input("Number of Cylinders", 3, 8, 4, step=1,
                                help="Most cars have 4. Muscle cars & trucks have 6 or 8.")
    displacement = st.number_input("Engine Displacement (cu in)", 68.0, 455.0, 200.0, step=10.0,
                                   help="Small car: 70–150 | Medium: 150–300 | Large: 300–455")
    horsepower = st.number_input("Horsepower (HP)", 40, 250, 120, step=5,
                                 help="Normal: 100–160 HP | Sporty/Muscle: 180–250+ HP")

with col2:
    st.markdown("#### Weight & Performance")
    weight = st.number_input("Car Weight (lbs)", 1500, 6000, 3000, step=100,
                             help="Light: 2000–2800 | Average: 3000–4000 | Heavy: 4500+")
    acceleration = st.number_input("0–60 mph (seconds)", 8.0, 30.0, 15.0, step=0.5,
                                   help="Fast: 8–12 sec | Normal: 13–18 sec | Slow: 20+ sec")
    origin = st.selectbox("Country of Origin", [1, 2, 3],
                          format_func=lambda x: {1: "USA", 2: "Europe", 3: "Japan"}[x],
                          index=2,
                          help="Japanese cars were usually the most fuel-efficient in the 70s–80s!")

model_year = st.slider("Model Year", 70, 82, 78, help="70 = 1970, 82 = 1982. Newer = slightly better MPG")

# ==================== EXAMPLES ====================
st.info("""
**Real-Life Examples**  
1976 Ford Mustang → USA • 8 cyl • 300 HP • 3800 lbs • year 76 → ~14 MPG (~6.0 km/L)  
1980 Honda Civic → Japan • 4 cyl • 67 HP • 2000 lbs • year 80 → ~36 MPG (~15.3 km/L)  
1978 VW Golf/Rabbit → Europe • 4 cyl • 78 HP • 2200 lbs • year 78 → ~31 MPG (~13.2 km/L)
""")

# ==================== PREDICT BUTTON ====================
if st.button("Predict Fuel Efficiency Now!", type="primary", use_container_width=True):
    with st.spinner("Revving up the prediction engine..."):
        X = np.array([[cylinders, displacement, horsepower, weight, acceleration, model_year, origin]])
        prediction = model.predict(scaler.transform(X))[0]
        mpg = round(float(prediction), 1)
        km_per_liter = round(mpg * 0.425144, 1)  # Accurate conversion: 1 MPG ≈ 0.425144 km/L

    st.markdown("<div style='margin:60px 0'></div>", unsafe_allow_html=True)

    # === SIDE-BY-SIDE MPG & km/L CARDS ===
    col_mpg, col_kml = st.columns(2)

    with col_mpg:
        st.markdown(f"""
        <div class="result-card" style="background: linear-gradient(135deg, #5b21b6, #7c3aed);">
            <h1 style="font-size:70px; margin:0; color:white;">{mpg}</h1>
            <h3 style="margin:10px 0 0; color:white; opacity:0.9;">MPG</h3>
            <p style="margin:5px 0 0; font-size:15px; color:white;">Miles Per Gallon (US Standard)</p>
        </div>
        """, unsafe_allow_html=True)

    with col_kml:
        st.markdown(f"""
        <div class="result-card" style="background: linear-gradient(135deg, #dc2626, #f97316);">
            <h1 style="font-size:70px; margin:0; color:white;">{km_per_liter}</h1>
            <h3 style="margin:10px 0 0; color:white; opacity:0.9;">km/L</h3>
            <p style="margin:5px 0 0; font-size:15px; color:white;">Kilometers Per Liter (Global)</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='margin:40px 0'></div>", unsafe_allow_html=True)
    st.caption("🔄 Conversion: 1 MPG ≈ 0.425 km/L | Perfect for comparing US classics with the world!")

    # === Feedback based on MPG ===
    if mpg >= 35:
        st.success("Outstanding efficiency! Probably a legend like the Civic!")
        st.balloons()
    elif mpg >= 28:
        st.success("Excellent fuel economy — sips fuel like a pro!")
    elif mpg >= 20:
        st.info("Solid for the era — balanced power and efficiency")
    else:
        st.warning("Classic American muscle — drinks fuel, but sounds epic!")

# ==================== FOOTER ====================
st.markdown(
    "<p style='text-align:center; color:#94a3b8; margin-top:100px; font-size:16px; font-weight:600;'>"
    "Big blocks, small Civics, and everything that makes car lovers smile — this one’s for you! ❤️ from Arun</p>",
    unsafe_allow_html=True
)
