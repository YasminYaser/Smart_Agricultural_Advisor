import numpy as np
import pandas as pd
import streamlit as st

from model_utils import load_or_train

st.set_page_config(
    page_title="Smart Agricultural Advisor",
    page_icon="🌾",
    layout="centered",
)

# --------------------------------------------------------------------------
# Models (loaded once; retrained automatically if the saved files can't load)
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner="Warming up the models 🌱")
def get_models():
    return load_or_train()


rf_model, le, xgb_model, fert_mean, retrained = get_models()

CROP_EMOJI = {
    "apple": "🍎", "banana": "🍌", "blackgram": "🫘", "chickpea": "🫛",
    "coconut": "🥥", "coffee": "☕", "cotton": "☁️", "grapes": "🍇",
    "jute": "🌿", "kidneybeans": "🫘", "lentil": "🫘", "maize": "🌽",
    "mango": "🥭", "mothbeans": "🫘", "mungbean": "🌱", "muskmelon": "🍈",
    "orange": "🍊", "papaya": "🍈", "pigeonpeas": "🫛", "pomegranate": "🍎",
    "rice": "🌾", "watermelon": "🍉",
}

# Training ranges of the yield model (used for the reliability warning)
YIELD_RANGES = {
    "Temperature": (23.8, 40.3),
    "Nitrogen": (58.8, 80.2),
    "Phosphorus": (17.7, 25.2),
    "Potassium": (14.7, 22.1),
}
YIELD_MIN, YIELD_MAX = 5.15, 12.34

# --------------------------------------------------------------------------
# Style + animations
# --------------------------------------------------------------------------
CSS = """
<style>
#MainMenu, footer {visibility: hidden;}
header[data-testid="stHeader"] {background: transparent;}

.stApp {
  background:
    radial-gradient(circle at 12% 0%, rgba(52,211,153,.16), transparent 42%),
    radial-gradient(circle at 92% 8%, rgba(251,191,36,.08), transparent 38%),
    #0e1913;
  color: #e8f3ec;
}
/* slow glowing orb behind the page */
.stApp::before {
  content: ""; position: fixed; width: 520px; height: 520px; top: -160px; left: 50%;
  margin-left: -260px; border-radius: 50%; pointer-events: none; z-index: 0;
  background: radial-gradient(circle, rgba(52,211,153,.20), transparent 65%);
  filter: blur(30px); animation: drift 14s ease-in-out infinite alternate;
}
@keyframes drift {from {transform: translateX(-140px) scale(1);} to {transform: translateX(140px) scale(1.15);}}

.block-container {position: relative; z-index: 1; padding-top: 2.2rem; max-width: 840px;}

@keyframes fadeup {from {opacity: 0; transform: translateY(16px);} to {opacity: 1; transform: translateY(0);}}

/* hero */
.hero {text-align: center; padding: 8px 0 22px; animation: fadeup .8s ease both;}
.tag {display: inline-block; font-size: .72rem; letter-spacing: .18em; font-weight: 700; color: #34d399;
  border: 1px solid rgba(52,211,153,.35); background: rgba(52,211,153,.08); padding: 5px 14px; border-radius: 999px;}
.hero h1 {font-size: 2.7rem; font-weight: 800; margin: 14px 0 8px; line-height: 1.15; color: #f4faf6; letter-spacing: -.5px;}
.hero h1 span {background: linear-gradient(90deg, #34d399, #a3e635, #fbbf24); -webkit-background-clip: text;
  background-clip: text; -webkit-text-fill-color: transparent; color: transparent;}
.hero p {color: #9db8a8; font-size: 1.05rem; margin: 0;}

/* input cards */
[data-testid="stVerticalBlockBorderWrapper"] {
  background: rgba(255,255,255,.035);
  border: 1px solid rgba(255,255,255,.09) !important;
  border-radius: 18px !important;
  animation: fadeup .8s ease both;
  transition: transform .25s ease, border-color .25s ease, box-shadow .25s ease;
}
[data-testid="stVerticalBlockBorderWrapper"]:hover {
  transform: translateY(-3px);
  border-color: rgba(52,211,153,.45) !important;
  box-shadow: 0 10px 30px rgba(52,211,153,.10);
}
.card-title {font-weight: 700; color: #34d399; font-size: .8rem; letter-spacing: .14em; text-transform: uppercase; margin-bottom: 4px;}

.stApp label, .stApp [data-testid="stWidgetLabel"] p {color: #d5e6db !important; font-weight: 500;}
.stApp [data-testid="stCaptionContainer"] {color: #8aa595;}

/* button */
.stButton > button {
  width: 100%; border: none; border-radius: 14px; padding: .85rem 1rem; font-weight: 700; font-size: 1.05rem;
  background: #34d399; color: #06281a;
  transition: transform .2s ease, box-shadow .2s ease, background .2s ease;
}
.stButton > button:hover {background: #4ee0ad; color: #06281a; transform: translateY(-2px); box-shadow: 0 10px 26px rgba(52,211,153,.35);}
.stButton > button:active {transform: scale(.98);}

/* results */
.results {display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-top: 10px;}
@media (max-width: 640px) {.results {grid-template-columns: 1fr;}}
.rcard {background: rgba(255,255,255,.045); border: 1px solid rgba(255,255,255,.10); border-radius: 20px;
  padding: 24px 20px; text-align: center; animation: pop .6s cubic-bezier(.2,1.2,.4,1) both;}
.rcard.second {animation-delay: .15s;}
@keyframes pop {from {opacity: 0; transform: translateY(24px) scale(.96);} to {opacity: 1; transform: translateY(0) scale(1);}}
.rlabel {color: #8aa595; font-size: .75rem; letter-spacing: .16em; text-transform: uppercase; font-weight: 700;}
.remoji {font-size: 56px; line-height: 1.2; margin: 8px 0 2px;}
.rvalue {font-size: 2.1rem; font-weight: 800; color: #34d399; line-height: 1.15;}
.rvalue.white {color: #f4faf6;}
.rsub {color: #9db8a8; font-size: .9rem; margin-top: 4px;}

.bars {margin-top: 16px; text-align: left;}
.brow {display: flex; align-items: center; gap: 8px; margin: 8px 0; font-size: .88rem; color: #cfe3d6; font-weight: 500;}
.brow .nm {width: 110px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;}
.bar {flex: 1; height: 8px; background: rgba(255,255,255,.08); border-radius: 99px; overflow: hidden;}
.fill {height: 100%; width: var(--w); border-radius: 99px; background: linear-gradient(90deg, #34d399, #a3e635);
  animation: grow 1.2s cubic-bezier(.2,.9,.3,1) both;}
@keyframes grow {from {width: 0;} to {width: var(--w);}}
.brow .pc {width: 42px; text-align: right; color: #9db8a8;}

.gauge {margin: 22px 6px 0;}
.gauge .bar {height: 10px; background: linear-gradient(90deg, #fbbf24, #a3e635, #34d399); overflow: visible; position: relative;}
.marker {position: absolute; top: -6px; left: var(--pos); width: 5px; height: 22px; border-radius: 4px; background: #fff;
  box-shadow: 0 0 10px rgba(255,255,255,.7); transform: translateX(-50%); animation: slide 1.3s cubic-bezier(.2,.9,.3,1) both;}
@keyframes slide {from {left: 0;} to {left: var(--pos);}}
.gscale {display: flex; justify-content: space-between; font-size: .75rem; color: #8aa595; margin-top: 10px;}

/* expander for the notes */
[data-testid="stExpander"] {background: rgba(255,255,255,.03); border: 1px solid rgba(255,255,255,.08) !important; border-radius: 14px;}
[data-testid="stExpander"] summary p {color: #cfe3d6 !important;}

.footer {text-align: center; color: #6f8a7a; font-size: .8rem; margin: 28px 0 6px;}
</style>
"""

HERO = """
<div class="hero">
<span class="tag">AI CROP ADVISOR</span>
<h1>Smart <span>Agricultural</span> Advisor</h1>
<p>Enter your soil and weather readings and get the best crop to grow, plus an estimated yield.</p>
</div>
"""

st.markdown(CSS, unsafe_allow_html=True)
st.markdown(HERO, unsafe_allow_html=True)

# --------------------------------------------------------------------------
# Inputs
# --------------------------------------------------------------------------
col_soil, col_weather = st.columns(2)

with col_soil:
    with st.container(border=True):
        st.markdown('<div class="card-title">Soil</div>', unsafe_allow_html=True)
        N = st.slider("Nitrogen (N)", 0, 140, 90)
        P = st.slider("Phosphorus (P)", 5, 145, 42)
        K = st.slider("Potassium (K)", 5, 205, 43)
        ph = st.slider("Soil pH", 3.5, 9.9, 6.5, step=0.1)

with col_weather:
    with st.container(border=True):
        st.markdown('<div class="card-title">Weather</div>', unsafe_allow_html=True)
        temperature = st.slider("Temperature (°C)", 8.0, 44.0, 21.0, step=0.5)
        humidity = st.slider("Humidity (%)", 14.0, 100.0, 82.0, step=1.0)
        rainfall = st.slider("Rainfall (mm)", 20.0, 300.0, 200.0, step=1.0)

with st.container(border=True):
    use_fert = st.checkbox("I know the fertilizer amount (kg/ha)")
    if use_fert:
        fertilizer = st.slider("Fertilizer (kg/ha)", 0.0, 200.0, float(round(fert_mean, 1)), step=0.5)
    else:
        fertilizer = float(fert_mean)
        st.caption(f"Using the dataset average: {fert_mean:.1f} kg/ha")

go = st.button("Get my recommendation", type="primary")

# --------------------------------------------------------------------------
# Prediction
# --------------------------------------------------------------------------
if go:
    crop_in = pd.DataFrame(
        [[N, P, K, temperature, humidity, ph, rainfall]],
        columns=["N", "P", "K", "temperature", "humidity", "ph", "rainfall"],
    )
    proba = rf_model.predict_proba(crop_in)[0]
    order = np.argsort(proba)[::-1][:3]
    names = le.inverse_transform(rf_model.classes_)  # class names in model order
    top = [(str(names[i]), float(proba[i])) for i in order]
    crop, conf = top[0]

    yield_in = pd.DataFrame(
        [[fertilizer, temperature, N, P, K]],
        columns=["Fertilizer", "temp", "N", "P", "K"],
    )
    est_yield = float(xgb_model.predict(yield_in)[0])

    # out-of-range warning for the yield model
    checks = {**YIELD_RANGES}
    values = {"Temperature": temperature, "Nitrogen": N, "Phosphorus": P, "Potassium": K}
    outside = [f"{k} ({lo}–{hi})" for k, (lo, hi) in checks.items() if not lo <= values[k] <= hi]
    if use_fert and not 49.8 <= fertilizer <= 80.2:
        outside.append("Fertilizer (49.8–80.2)")

    bars = "".join(
        f'<div class="brow"><span class="nm">{CROP_EMOJI.get(n, "🌱")} {n.capitalize()}</span>'
        f'<div class="bar"><div class="fill" style="--w:{p*100:.0f}%"></div></div>'
        f'<span class="pc">{p*100:.0f}%</span></div>'
        for n, p in top
    )

    pos = float(np.clip((est_yield - YIELD_MIN) / (YIELD_MAX - YIELD_MIN), 0, 1)) * 100
    level = "Low" if pos < 33 else "Medium" if pos < 66 else "High"

    html = (
        '<div class="results">'
        '<div class="rcard">'
        '<div class="rlabel">Recommended crop</div>'
        f'<div class="remoji">{CROP_EMOJI.get(crop, "🌱")}</div>'
        f'<div class="rvalue">{crop.capitalize()}</div>'
        f'<div class="rsub">{conf*100:.0f}% match</div>'
        f'<div class="bars">{bars}</div>'
        '</div>'
        '<div class="rcard second">'
        '<div class="rlabel">Estimated yield</div>'
        '<div class="remoji">📈</div>'
        f'<div class="rvalue white">{est_yield:.2f} t/ha</div>'
        f'<div class="rsub">{level} compared with the training data</div>'
        '<div class="gauge"><div class="bar">'
        f'<div class="marker" style="--pos:{pos:.0f}%"></div></div>'
        f'<div class="gscale"><span>{YIELD_MIN:.1f}</span><span>{YIELD_MAX:.1f}</span></div></div>'
        '</div>'
        '</div>'
    )
    st.markdown(html, unsafe_allow_html=True)

    with st.expander("How reliable is the yield estimate?"):
        if outside:
            st.write(
                "Some of your inputs are outside the ranges the yield model was trained on "
                "(" + ", ".join(outside) + "). For these values the yield number is only a rough guide."
            )
        else:
            st.write("Your inputs are inside the ranges the yield model was trained on.")
        st.write(
            "The yield model was trained on a separate dataset and is not specific to the recommended "
            "crop, so treat it as a general estimate. The crop recommendation does not depend on it."
        )

st.markdown('<div class="footer">Smart Agricultural Advisor</div>', unsafe_allow_html=True)
