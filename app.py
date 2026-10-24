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

/* animated gradient background */
.stApp {
  background: linear-gradient(125deg, #e6f7ea, #f7fbe8, #dff3ef, #eaf8e2, #e6f7ea);
  background-size: 400% 400%;
  animation: bgmove 22s ease infinite;
}
@keyframes bgmove {0%{background-position:0% 50%} 50%{background-position:100% 50%} 100%{background-position:0% 50%}}

.block-container {position: relative; z-index: 1; padding-top: 2rem; max-width: 820px;}

/* floating emoji layer */
.floaters {position: fixed; inset: 0; overflow: hidden; pointer-events: none; z-index: 0;}
.floaters span {position: absolute; bottom: -60px; font-size: 28px; opacity: 0;
  animation: rise linear infinite;}
@keyframes rise {
  0%   {transform: translateY(0) rotate(0deg); opacity: 0;}
  10%  {opacity: .55;}
  90%  {opacity: .45;}
  100% {transform: translateY(-115vh) rotate(360deg); opacity: 0;}
}
.floaters span:nth-child(1){left:6%;  animation-duration:19s; animation-delay:0s;}
.floaters span:nth-child(2){left:18%; animation-duration:24s; animation-delay:4s;  font-size:22px;}
.floaters span:nth-child(3){left:32%; animation-duration:21s; animation-delay:9s;}
.floaters span:nth-child(4){left:47%; animation-duration:26s; animation-delay:2s;  font-size:34px;}
.floaters span:nth-child(5){left:61%; animation-duration:20s; animation-delay:7s;}
.floaters span:nth-child(6){left:74%; animation-duration:23s; animation-delay:12s; font-size:24px;}
.floaters span:nth-child(7){left:86%; animation-duration:25s; animation-delay:5s;}
.floaters span:nth-child(8){left:94%; animation-duration:18s; animation-delay:10s; font-size:20px;}

/* hero */
.hero {text-align: center; padding: 10px 0 18px; animation: fadeup .9s ease both;}
.hero .big {font-size: 64px; display: inline-block; animation: sway 3.5s ease-in-out infinite; transform-origin: 50% 90%;}
@keyframes sway {0%,100%{transform: rotate(-7deg)} 50%{transform: rotate(7deg)}}
.hero h1 {font-size: 2.6rem; font-weight: 800; margin: 4px 0 6px; line-height: 1.15;
  background: linear-gradient(90deg, #1f8a4c, #7bc043, #1f8a4c);
  background-size: 200% auto; -webkit-background-clip: text; background-clip: text;
  -webkit-text-fill-color: transparent; color: transparent;
  animation: shine 5s linear infinite;}
@keyframes shine {to {background-position: 200% center;}}
.hero p {color: #3f6b52; font-size: 1.05rem; margin: 0;}
.pills {margin-top: 14px; display: flex; gap: 10px; justify-content: center; flex-wrap: wrap;}
.pill {background: rgba(255,255,255,.75); border: 1px solid #cfe9d6; color: #25663f;
  padding: 5px 14px; border-radius: 999px; font-size: .85rem; font-weight: 600;
  backdrop-filter: blur(6px); animation: fadeup .9s ease both;}
.pill:nth-child(2){animation-delay:.15s} .pill:nth-child(3){animation-delay:.3s}

@keyframes fadeup {from {opacity: 0; transform: translateY(18px);} to {opacity: 1; transform: translateY(0);}}

/* input cards */
[data-testid="stVerticalBlockBorderWrapper"] {
  background: rgba(255,255,255,.72);
  border: 1px solid #d4ecda !important;
  border-radius: 20px !important;
  backdrop-filter: blur(8px);
  box-shadow: 0 8px 28px rgba(46,158,91,.10);
  animation: fadeup .8s ease both;
  transition: transform .25s ease, box-shadow .25s ease;
}
[data-testid="stVerticalBlockBorderWrapper"]:hover {
  transform: translateY(-3px);
  box-shadow: 0 14px 34px rgba(46,158,91,.18);
}
.card-title {font-weight: 700; color: #1b5e3a; font-size: 1.05rem; margin-bottom: 2px;}

/* button */
.stButton > button {
  width: 100%; border: none; border-radius: 14px; padding: .8rem 1rem; font-weight: 700; font-size: 1.05rem;
  background: linear-gradient(90deg, #2e9e5b, #7bc043); color: #fff;
  box-shadow: 0 6px 18px rgba(46,158,91,.35);
  transition: transform .2s ease, box-shadow .2s ease;
  animation: pulse 2.6s ease-in-out infinite;
}
.stButton > button:hover {transform: translateY(-2px) scale(1.01); box-shadow: 0 10px 26px rgba(46,158,91,.45); color:#fff;}
.stButton > button:active {transform: scale(.98);}
@keyframes pulse {0%,100%{box-shadow: 0 6px 18px rgba(46,158,91,.35)} 50%{box-shadow: 0 6px 30px rgba(123,192,67,.65)}}

/* results */
.results {display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-top: 8px;}
@media (max-width: 640px) {.results {grid-template-columns: 1fr;}}
.rcard {background: rgba(255,255,255,.9); border: 1px solid #d4ecda; border-radius: 22px; padding: 22px 20px;
  text-align: center; box-shadow: 0 10px 30px rgba(46,158,91,.14);
  animation: pop .7s cubic-bezier(.2,1.4,.4,1) both;}
.rcard.second {animation-delay: .18s;}
@keyframes pop {from {opacity: 0; transform: scale(.7) translateY(20px);} to {opacity: 1; transform: scale(1) translateY(0);}}
.rlabel {color: #5b8a6e; font-size: .8rem; letter-spacing: .12em; text-transform: uppercase; font-weight: 700;}
.remoji {font-size: 68px; line-height: 1.1; margin: 6px 0; display: inline-block; animation: bounce 2s ease-in-out infinite;}
@keyframes bounce {0%,100%{transform: translateY(0)} 50%{transform: translateY(-10px)}}
.rvalue {font-size: 2.1rem; font-weight: 800; color: #1b5e3a; line-height: 1.15;}
.rsub {color: #5b8a6e; font-size: .9rem; margin-top: 4px;}

.bars {margin-top: 14px; text-align: left;}
.brow {display: flex; align-items: center; gap: 8px; margin: 7px 0; font-size: .88rem; color: #2f5d44; font-weight: 600;}
.brow .nm {width: 105px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;}
.bar {flex: 1; height: 10px; background: #e3f3e7; border-radius: 99px; overflow: hidden;}
.fill {height: 100%; width: var(--w); border-radius: 99px;
  background: linear-gradient(90deg, #2e9e5b, #9ad14b);
  animation: grow 1.3s cubic-bezier(.2,.9,.3,1) both;}
@keyframes grow {from {width: 0;} to {width: var(--w);}}
.brow .pc {width: 44px; text-align: right;}

.gauge {margin: 16px 6px 0;}
.gauge .bar {height: 14px; background: linear-gradient(90deg, #f6d365, #c8e36b, #4cbf74); opacity: .35; position: relative; overflow: visible;}
.marker {position: absolute; top: -5px; left: var(--pos); width: 4px; height: 24px; border-radius: 4px; background: #1b5e3a;
  transform: translateX(-50%); animation: slide 1.4s cubic-bezier(.2,.9,.3,1) both;}
@keyframes slide {from {left: 0;} to {left: var(--pos);}}
.gscale {display: flex; justify-content: space-between; font-size: .75rem; color: #5b8a6e; margin-top: 8px;}

.note {margin-top: 16px; background: rgba(255,255,255,.7); border-left: 4px solid #7bc043; border-radius: 10px;
  padding: 10px 14px; color: #3f6b52; font-size: .85rem; animation: fadeup 1s ease .4s both;}
.warnbox {margin-top: 14px; background: #fff7e0; border-left: 4px solid #f0b429; border-radius: 10px;
  padding: 10px 14px; color: #7a5a00; font-size: .88rem; animation: fadeup .6s ease both;}

/* labels / text colors so they stay readable */
.stApp label, .stApp [data-testid="stWidgetLabel"] p {color: #1b3a2a !important; font-weight: 600;}
.footer {text-align: center; color: #6b9a7e; font-size: .8rem; margin: 26px 0 6px;}
</style>
"""

FLOATERS = (
    '<div class="floaters">'
    + "".join(f"<span>{e}</span>" for e in ["🌾", "🌱", "🍃", "☀️", "💧", "🌽", "🍀", "🌻"])
    + "</div>"
)

HERO = """
<div class="hero">
<div class="big">🌾</div>
<h1>Smart Agricultural Advisor</h1>
<p>Tell us about your soil and weather. We'll tell you what to grow.</p>
<div class="pills">
<span class="pill">🌳 Random Forest · crop</span>
<span class="pill">⚡ XGBoost · yield</span>
<span class="pill">🧪 Soil + weather</span>
</div>
</div>
"""

st.markdown(CSS, unsafe_allow_html=True)
st.markdown(FLOATERS, unsafe_allow_html=True)
st.markdown(HERO, unsafe_allow_html=True)

if retrained:
    st.caption("ℹ️ Saved models couldn't be loaded on this server, so fresh ones were trained from the CSV files.")

# --------------------------------------------------------------------------
# Inputs
# --------------------------------------------------------------------------
col_soil, col_weather = st.columns(2)

with col_soil:
    with st.container(border=True):
        st.markdown('<div class="card-title">🧪 Soil</div>', unsafe_allow_html=True)
        N = st.slider("Nitrogen (N)", 0, 140, 90)
        P = st.slider("Phosphorus (P)", 5, 145, 42)
        K = st.slider("Potassium (K)", 5, 205, 43)
        ph = st.slider("Soil pH", 3.5, 9.9, 6.5, step=0.1)

with col_weather:
    with st.container(border=True):
        st.markdown('<div class="card-title">🌤️ Weather</div>', unsafe_allow_html=True)
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

go = st.button("🌱  Get my recommendation", type="primary")

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
        f'<div class="rvalue">{est_yield:.2f} t/ha</div>'
        f'<div class="rsub">{level} compared with the training data</div>'
        '<div class="gauge"><div class="bar">'
        f'<div class="marker" style="--pos:{pos:.0f}%"></div></div>'
        f'<div class="gscale"><span>{YIELD_MIN:.1f}</span><span>{YIELD_MAX:.1f}</span></div></div>'
        '</div>'
        '</div>'
    )
    if outside:
        html += (
            '<div class="warnbox">⚠️ Some inputs are outside the yield model\'s training range, '
            "so the yield estimate may be unreliable: " + ", ".join(outside) + "</div>"
        )
    html += (
        '<div class="note">ℹ️ The yield model was trained on a separate dataset and is not '
        "specific to the recommended crop. Treat it as a rough estimate.</div>"
    )
    st.markdown(html, unsafe_allow_html=True)
    st.toast(f"Best match: {crop.capitalize()} {CROP_EMOJI.get(crop, '🌱')}")

st.markdown('<div class="footer">Made with 🌱 · Smart Agricultural Advisor</div>', unsafe_allow_html=True)
