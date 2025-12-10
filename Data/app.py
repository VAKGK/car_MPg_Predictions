import streamlit as st
import joblib
import numpy as np

# ==================== CLEAN UI ====================
st.markdown("""
<style>
    #MainMenu, footer, header, .stDeployButton {visibility: hidden !important;}
    .block-container {padding-top: 2rem !important;}
    hr {display: none !important;}
</style>
""", unsafe_allow_html=True)

# ==================== CUSTOM ANIMATED UI ====================
st.markdown("""
<style>
/* Hide Streamlit default elements */
#MainMenu, footer, header, .stDeployButton {visibility: hidden !important;}

/* Soft fade + slide-down animation for page */
.block-container {
    padding-top: 2rem !important;
    animation: slideFade 1.2s ease-out;
}
@keyframes slideFade {
    0% {opacity: 0; transform: translateY(-20px);}
    100% {opacity: 1; transform: translateY(0);}
}

/* Modern pastel background */
body {
    background: linear-gradient(135deg, #eef2ff, #e0e7ff);
}

/* NEW Animated Title — smooth shine effect */
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
    top: 0;
    left: -100%;
    width: 100%;
    height: 100%;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.4), transparent);
    animation: shine 3s infinite;
}
@keyframes shine {
    0% {left: -100%;}
    50% {left: 100%;}
    100% {left: 100%;}
}

/* Predict Button – floating effect */
.stButton>button {
    background: linear-gradient(135deg, #4f46e5, #9333ea);
    color: white;
    border-radius: 12px;
    padding: 12px 25px;
    font-size: 20px;
    border: none;
    transition: all .3s ease-in-out;
    animation: floatBtn 3s ease-in-out infinite;
    box-shadow: 0 6px 20px rgba(90, 50, 200, 0.25);
}
@keyframes floatBtn {
    0% {transform: translateY(0);}
    50% {transform: translateY(-5px);}
    100% {transform: translateY(0);}
}
.stButton>button:hover {
    transform: translateY(-8px) scale(1.03);
    box-shadow: 0 15px 35px rgba(90, 50, 200, 0.4);
}

/* MPG Result Box – Glassmorphism + soft glow */
.mpg-box {
    background: rgba(255, 255, 255, 0.15);
    backdrop-filter: blur(15px);
    border-radius: 40px;
    padding: 70px 40px;
    border: 1px solid rgba(255,255,255,0.3);
    animation: glowBox 2.5s infinite alternate ease-in-out;
}
@keyframes glowBox {
    0% {box-shadow: 0 0 15px rgba(124, 58, 237, 0.3);}
    100% {box-shadow: 0 0 35px rgba(124, 58, 237, 0.6);}
}
</style>
""", unsafe_allow_html=True)


# ==================== LOAD MODEL (7 features with Origin) ====================
@st.cache_resource
def load_model():
    # OPTION 1: Use the folder of your script (recommended)
    BASE_DIR = os.path.dirname("D:\DATA SCIENCE\ML\PROJECTS\MPG_REG\app.py")
    scaler_path = os.path.join(BASE_DIR, 'scaler.joblib')
    model_path = os.path.join(BASE_DIR, 'car_mileage_model.joblib')

    # OPTION 2: Or specify the full local path manually
    # scaler_path = "/full/path/to/scaler.joblib"
    # model_path = "/full/path/to/car_mileage_model.joblib"

    # Check if files exist
    if not os.path.exists(scaler_path):
        st.error(f"Cannot find file: {scaler_path}")
        return None, None
    if not os.path.exists(model_path):
        st.error(f"Cannot find file: {model_path}")
        return None, None

    # Load the scaler and model
    scaler = joblib.load(scaler_path)
    model = joblib.load(model_path)

    return scaler, model

scaler, model = load_model()

# ==================== PAGE SETUP ====================
st.set_page_config(page_title="Car MPG Predictor", page_icon="favicon.ico", layout="centered")

# ==================== TITLE ====================
st.markdown("<h1 style='text-align:center; color:#1e40af; margin-bottom:8px;'>Car Fuel Efficiency Predictor</h1>",
            unsafe_allow_html=True)
st.markdown(
    "<h5 style='text-align:center; color:#6b7280; margin-bottom:50px;'>1970–1982 Classic Cars</h5>",
    unsafe_allow_html=True)

# ==================== INPUTS – 2 COLUMNS WITH HELP TEXT ON EVERY FIELD ====================
col1, col2 = st.columns(2)

with col1:
    st.markdown("#### Engine & Power")

    cylinders = st.number_input(
        "Number of Cylinders",
        min_value=3, max_value=8, value=4, step=1,
        help="Most cars have 4. Muscle cars & trucks have 6 or 8."
    )

    displacement = st.number_input(
        "Engine Displacement (cubic inches)",
        min_value=68.0, max_value=455.0, value=200.0, step=10.0,
        help="Small car: 70–150 ci | Medium: 150–300 ci | Large: 300–455 ci (bigger = more power, less MPG)"
    )

    horsepower = st.number_input(
        "Horsepower (HP)",
        min_value=40, max_value=250, value=120, step=5,
        help="Normal car: 100–160 HP | Sporty: 180–250 HP | Muscle cars go higher!"
    )

with col2:
    st.markdown("#### Weight & Performance")

    weight = st.number_input(
        "Car Weight (pounds)",
        min_value=1500, max_value=6000, value=3000, step=100,
        help="Light car: 2000–2800 lbs | Average: 3000–4000 lbs | Heavy: 4500+ lbs"
    )

    acceleration = st.number_input(
        "0–60 mph (seconds)",
        min_value=8.0, max_value=30.0, value=15.0, step=0.5,
        help="Fast car: 8–12 sec | Normal: 13–18 sec | Slow: 20+ sec"
    )

    origin = st.selectbox(
        "Country of Origin",
        options=[1, 2, 3],
        format_func=lambda x: {1: "USA", 2: "Europe", 3: "Japan"}[x],
        index=2,
        help="Japanese cars were usually the most fuel-efficient in the 70s–80s!"
    )

# ==================== MODEL YEAR SLIDER ====================
model_year = st.slider(
    "Model Year",
    min_value=70, max_value=82, value=78,
    help="70 = 1970, 82 = 1982. Newer years = slightly better MPG"
)

# ==================== EXAMPLES ====================
st.info("""
**Real-Life Examples**  
1976 Ford Mustang → USA • 8 cyl • 300 HP • 3800 lbs • year 76 → ~14 MPG  
1980 Honda Civic → Japan • 4 cyl • 67 HP • 2000 lbs • year 80 → ~36 MPG  
1978 VW Golf/Rabbit → Europe • 4 cyl • 78 HP • 2200 lbs • year 78 → ~31 MPG
""")

# ==================== PREDICT BUTTON ====================
predict = st.button("Predict MPG Now!", type="primary", use_container_width=True)

# ==================== RESULT ====================
if predict:
    X = np.array([[cylinders, displacement, horsepower, weight, acceleration, model_year, origin]])
    mpg = round(float(model.predict(scaler.transform(X))[0]), 1)

    st.markdown("<div style='margin:60px 0'></div>", unsafe_allow_html=True)

    st.markdown(f"""
    <div style="text-align:center; padding:80px 30px; background:linear-gradient(135deg, #5b21b6, #7c3aed);
                color:white; border-radius:40px; box-shadow:0 20px 50px rgba(0,0,0,0.35);">
        <h1 style="font-size:80px; margin:0; color:white; line-height:1;">{mpg}</h1>
        <h2 style="margin:15px 0 0; opacity:0.95;">Miles Per Gallon (MPG)</h2>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='margin:50px 0'></div>", unsafe_allow_html=True)

    if mpg >= 35:
        st.success("Outstanding! Most likely a Japanese car!");
        st.balloons()
    elif mpg >= 28:
        st.success("Excellent fuel economy!")
    elif mpg >= 20:
        st.info("Good for the 1970s")
    else:
        st.warning("Classic American V8 — drinks gas, but sounds amazing!")

# ==================== FOOTER ====================
st.markdown(
    "<p style='text-align:center; color:#94a3b8; margin-top:100px; font-size:15px;'> Big blocks, small Civics, and everything that makes car lovers smile — this one’s for you! ❤️ from Arun </p>",
    unsafe_allow_html=True)
