import numpy as np
import pandas as pd
import streamlit as st

from model_utils import load_or_train

st.set_page_config(
    page_title="Smart Agricultural Advisor",
    page_icon="🌾",
    layout="wide",
)


@st.cache_resource(show_spinner="Warming up the models...")
def get_models():
    return load_or_train()


rf_model, le, xgb_model, fert_mean, _retrained = get_models()

CROP_EMOJI = {
    "apple": "🍎", "banana": "🍌", "blackgram": "🫘", "chickpea": "🫛",
    "coconut": "🥥", "coffee": "☕", "cotton": "☁️", "grapes": "🍇",
    "jute": "🌿", "kidneybeans": "🫘", "lentil": "🫘", "maize": "🌽",
    "mango": "🥭", "mothbeans": "🫘", "mungbean": "🌱", "muskmelon": "🍈",
    "orange": "🍊", "papaya": "🍈", "pigeonpeas": "🫛", "pomegranate": "🍎",
    "rice": "🌾", "watermelon": "🍉",
}

# Ranges the yield model was trained on (used for the reliability note)
YIELD_RANGES = {
    "Temperature": (23.8, 40.3),
    "Nitrogen": (58.8, 80.2),
    "Phosphorus": (17.7, 25.2),
    "Potassium": (14.7, 22.1),
}
FERT_RANGE = (49.8, 80.2)
YIELD_MIN, YIELD_MAX = 5.15, 12.34

# --------------------------------------------------------------------------
# Style
# --------------------------------------------------------------------------
HILLS = (
    "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 800 160' "
    "preserveAspectRatio='none'%3E%3Cpath d='M0 90 C150 30 300 130 450 80 S700 40 800 90 V160 H0Z' "
    "fill='%23246b3f' fill-opacity='.55'/%3E%3Cpath d='M0 125 C200 85 350 150 520 112 S720 95 800 125 "
    "V160 H0Z' fill='%23164d2e' fill-opacity='.9'/%3E%3C/svg%3E"
)

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;500;600;700;800&display=swap');

.stApp, .stApp *:not([data-testid="stIconMaterial"]):not([class*="material"]) {font-family: 'Cairo', 'Segoe UI', sans-serif;}
.stApp {background: #f1f9f3;}
#MainMenu, footer {visibility: hidden;}
header[data-testid="stHeader"] {background: transparent;}
.block-container {max-width: 1240px; padding: 1.4rem 2.5rem 3rem;}
@media (max-width: 640px) {.block-container {padding: 1rem .8rem 2rem;}}

@keyframes fadeup {from {opacity: 0; transform: translateY(14px);} to {opacity: 1; transform: translateY(0);}}
@keyframes glow {0%,100% {opacity: .55; transform: scale(1);} 50% {opacity: .9; transform: scale(1.12);}}

/* top bar */
.topbar {display: flex; align-items: center; gap: 12px; margin-bottom: 14px; animation: fadeup .6s ease both;}
.logo {width: 46px; height: 46px; border-radius: 14px; background: linear-gradient(135deg, #1f7a45, #3dbb6e);
  display: flex; align-items: center; justify-content: center; font-size: 24px; box-shadow: 0 6px 16px rgba(31,122,69,.3);}
.topbar .ar {font-weight: 800; font-size: 1.15rem; color: #123d25; line-height: 1.2;}
.topbar .en {font-size: .8rem; color: #5a7d67; font-weight: 500;}

/* hero banner */
.hero {position: relative; overflow: hidden; border-radius: 26px; padding: 50px 28px 84px; text-align: center; margin-bottom: 18px;
  background:
    url("__HILLS__") bottom / 100% 110px no-repeat,
    radial-gradient(circle at 78% 18%, rgba(255,226,140,.35), transparent 38%),
    linear-gradient(180deg, #0d3a22 0%, #1b5e38 100%);
  box-shadow: 0 18px 40px rgba(15,61,36,.25); animation: fadeup .7s ease both;}
.hero .sun {position: absolute; width: 150px; height: 150px; right: 9%; top: 6%; border-radius: 50%;
  background: radial-gradient(circle, rgba(255,230,150,.55), transparent 68%); animation: glow 6s ease-in-out infinite;}
.hero h1 {position: relative; color: #fff; font-size: 2.5rem; font-weight: 800; line-height: 1.2; margin: 0 0 12px;
  padding: 0; letter-spacing: -.3px;}
.hero p {position: relative; color: #d3ecdb; font-size: 1.02rem; margin: 0 auto; max-width: 560px;}
@media (max-width: 640px) {.hero h1 {font-size: 1.8rem;} .hero {padding: 40px 18px 90px;}}

/* main panel */
.st-key-panel {background: #fff; border-radius: 24px; padding: 24px 22px 26px; box-shadow: 0 16px 36px rgba(15,61,36,.10);
  border: 1px solid #e3efe7;}
@media (max-width: 640px) {.st-key-panel {padding: 16px 12px 20px;}}
.rtitle {display: flex; align-items: center; gap: 10px; font-weight: 800; color: #123d25; font-size: 1.3rem; margin: 30px 0 14px;
  animation: fadeup .6s ease both;}
.rtitle .ic {width: 38px; height: 38px; border-radius: 11px; background: #dcf3e3; display: inline-flex; align-items: center; justify-content: center;}

/* section titles */
.sec {display: flex; align-items: center; gap: 10px; font-weight: 800; color: #123d25; font-size: 1.08rem; margin: 6px 0 12px;}
.sec .ic {width: 34px; height: 34px; border-radius: 10px; display: inline-flex; align-items: center; justify-content: center; font-size: 17px;}
.ic.soil {background: #dcf3e3;} .ic.wx {background: #dcecfa;} .ic.fe {background: #fdf0c6;}

/* input cards */
[class*="st-key-soil_"], [class*="st-key-wx_"], [class*="st-key-fert"] {
  border-radius: 16px; padding: 12px 14px 8px; transition: transform .2s ease, box-shadow .2s ease;
  animation: fadeup .6s ease both;}
[class*="st-key-soil_"] {background: #f0f9f2; border: 1px solid #dcefe2;}
[class*="st-key-wx_"] {background: #eff6fd; border: 1px solid #d9e8f6;}
[class*="st-key-fert"] {background: #fff9e6; border: 1px solid #f6e8b4;}
[class*="st-key-soil_"]:hover, [class*="st-key-wx_"]:hover, [class*="st-key-fert"]:hover {
  transform: translateY(-3px); box-shadow: 0 10px 22px rgba(15,61,36,.10);}
[class*="st-key-soil_"] label p, [class*="st-key-wx_"] label p, [class*="st-key-fert"] label p {
  font-size: .8rem !important; color: #55705f !important; font-weight: 600;}
[class*="st-key-soil_"] input, [class*="st-key-wx_"] input, [class*="st-key-fert"] input {
  font-size: 1.45rem !important; font-weight: 800 !important; color: #14532d !important; background: transparent !important;}
[class*="st-key-soil_"] [data-baseweb="input"], [class*="st-key-wx_"] [data-baseweb="input"], [class*="st-key-fert"] [data-baseweb="input"],
[class*="st-key-soil_"] [data-baseweb="base-input"], [class*="st-key-wx_"] [data-baseweb="base-input"], [class*="st-key-fert"] [data-baseweb="base-input"] {
  background: transparent !important; border: none !important;}
.rng {font-size: .72rem; color: #8aa396; margin: -4px 0 4px 2px;}

/* button */
.stButton > button {width: 100%; max-width: 420px; display: block; margin: 0 auto; border: none; border-radius: 14px; padding: .8rem 1rem; font-weight: 800; font-size: 1.05rem;
  background: linear-gradient(90deg, #1f7a45, #2e9e5b); color: #fff; box-shadow: 0 8px 20px rgba(31,122,69,.30);
  transition: transform .2s ease, box-shadow .2s ease;}
.stButton > button:hover {color: #fff; transform: translateY(-2px); box-shadow: 0 12px 26px rgba(31,122,69,.42);}
.stButton > button:active {transform: scale(.98);}

/* results */
.results {display: grid; grid-template-columns: 1fr 1fr; gap: 18px; max-width: 980px; margin: 0 auto;}
@media (max-width: 640px) {.results {grid-template-columns: 1fr;}}
.rcard {background: #fff; border: 1px solid #dcefe2; border-radius: 20px; padding: 24px 20px; text-align: center;
  box-shadow: 0 10px 28px rgba(31,122,69,.10); animation: pop .6s cubic-bezier(.2,1.2,.4,1) both;}
.rcard.second {animation-delay: .15s;}
@keyframes pop {from {opacity: 0; transform: translateY(22px) scale(.96);} to {opacity: 1; transform: translateY(0) scale(1);}}
.rlabel {color: #6a8a76; font-size: .74rem; letter-spacing: .14em; text-transform: uppercase; font-weight: 700;}
.remoji {font-size: 58px; line-height: 1.2; margin: 8px 0 2px;}
.rvalue {font-size: 2.1rem; font-weight: 800; color: #1f7a45; line-height: 1.15;}
.rvalue.dark {color: #123d25;}
.rsub {color: #6a8a76; font-size: .9rem; margin-top: 4px;}

.bars {margin-top: 16px; text-align: left;}
.brow {display: flex; align-items: center; gap: 8px; margin: 9px 0; font-size: .88rem; color: #2f5d44; font-weight: 600;}
.brow .nm {width: 112px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;}
.bar {flex: 1; height: 9px; background: #e4f1e8; border-radius: 99px; overflow: hidden;}
.fill {height: 100%; width: var(--w); border-radius: 99px; background: linear-gradient(90deg, #2e9e5b, #9ad14b);
  animation: grow 1.2s cubic-bezier(.2,.9,.3,1) both;}
@keyframes grow {from {width: 0;} to {width: var(--w);}}
.brow .pc {width: 44px; text-align: right; color: #55705f;}

.gauge {margin: 24px 6px 0;}
.gauge .bar {height: 10px; background: linear-gradient(90deg, #f6c453, #b4dd5c, #2e9e5b); overflow: visible; position: relative;}
.marker {position: absolute; top: -6px; left: var(--pos); width: 5px; height: 22px; border-radius: 4px; background: #123d25;
  transform: translateX(-50%); animation: slide 1.3s cubic-bezier(.2,.9,.3,1) both;}
@keyframes slide {from {left: 0;} to {left: var(--pos);}}
.gscale {display: flex; justify-content: space-between; font-size: .75rem; color: #7f998a; margin-top: 10px;}

.chips {display: flex; flex-wrap: wrap; gap: 8px; justify-content: center; margin: 18px 0 6px; animation: fadeup .8s ease .3s both;}
.chip {background: #f0f9f2; border: 1px solid #dcefe2; color: #2f5d44; font-size: .8rem; font-weight: 600; padding: 5px 12px; border-radius: 999px;}
.chip b {color: #14532d;}

.empty {text-align: center; padding: 40px 10px; color: #6a8a76; animation: fadeup .6s ease both;}
.empty .eico {font-size: 46px; margin-bottom: 6px;}
.empty b {display: block; color: #123d25; font-size: 1.1rem; margin-bottom: 4px;}

[data-testid="stExpander"] {background: #f7fbf8; border: 1px solid #dcefe2 !important; border-radius: 14px; margin-top: 16px;}
[data-testid="stExpander"] summary p {color: #14532d !important; font-weight: 700;}
.footer {text-align: center; color: #7f998a; font-size: .8rem; margin: 26px 0 0;}
</style>
""".replace("__HILLS__", HILLS)

HEADER = """
<div class="topbar">
<div class="logo">🌾</div>
<div><div class="ar">المرشد الزراعي الذكي</div><div class="en">Smart Agricultural Advisor</div></div>
</div>
<div class="hero">
<div class="sun"></div>
<h1>Choose the right crop… with confidence</h1>
<p>Enter your soil and weather data. We'll recommend the best crop and estimate the expected yield.</p>
</div>
"""

st.markdown(CSS, unsafe_allow_html=True)
st.markdown(HEADER, unsafe_allow_html=True)



def section(icon_cls, icon, title):
    st.markdown(
        f'<div class="sec"><span class="ic {icon_cls}">{icon}</span>{title}</div>',
        unsafe_allow_html=True,
    )


def field(prefix, key, label, lo, hi, default, step, fmt=None, rng_text=None):
    with st.container(key=f"{prefix}_{key}"):
        v = st.number_input(label, min_value=lo, max_value=hi, value=default, step=step, format=fmt)
        st.markdown(f'<div class="rng">{rng_text or f"{lo} – {hi}"}</div>', unsafe_allow_html=True)
    return v


# --------------------------------------------------------------------------
# Inputs
# --------------------------------------------------------------------------
with st.container(key="panel"):
    section("soil", "🌱", "Soil data")
    c = st.columns(4)
    with c[0]:
        N = field("soil", "N", "N — Nitrogen (kg/ha)", 0, 140, 90, 1)
    with c[1]:
        P = field("soil", "P", "P — Phosphorus (kg/ha)", 5, 145, 42, 1)
    with c[2]:
        K = field("soil", "K", "K — Potassium (kg/ha)", 5, 205, 43, 1)
    with c[3]:
        ph = field("soil", "ph", "pH — Soil pH", 3.5, 9.9, 6.5, 0.1, "%.1f")

    section("wx", "🌤️", "Weather & fertilizer")
    c = st.columns(4)
    with c[0]:
        temperature = field("wx", "t", "Temperature (°C)", 8.0, 44.0, 21.0, 0.5, "%.1f")
    with c[1]:
        humidity = field("wx", "h", "Humidity (%)", 14.0, 100.0, 82.0, 1.0, "%.0f")
    with c[2]:
        rainfall = field("wx", "r", "Rainfall (mm)", 20.0, 300.0, 200.0, 1.0, "%.0f")
    with c[3]:
        with st.container(key="fert_box"):
            use_fert = st.checkbox("Fertilizer (optional)")
            fertilizer = st.number_input(
                "Fertilizer (kg/ha)", min_value=0.0, max_value=200.0,
                value=float(round(fert_mean, 1)), step=0.5, format="%.1f",
                disabled=not use_fert,
            )
            if not use_fert:
                fertilizer = float(fert_mean)
            st.markdown(
                '<div class="rng">0 – 200</div>' if use_fert
                else f'<div class="rng">Using average ({fert_mean:.1f})</div>',
                unsafe_allow_html=True,
            )
    st.write("")
    go = st.button("Get Recommendation", type="primary")

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
    names = le.inverse_transform(rf_model.classes_)
    top = [(str(names[i]), float(proba[i])) for i in order]

    yield_in = pd.DataFrame(
        [[fertilizer, temperature, N, P, K]],
        columns=["Fertilizer", "temp", "N", "P", "K"],
    )
    est_yield = float(xgb_model.predict(yield_in)[0])

    values = {"Temperature": temperature, "Nitrogen": N, "Phosphorus": P, "Potassium": K}
    outside = [f"{k} ({lo}–{hi})" for k, (lo, hi) in YIELD_RANGES.items() if not lo <= values[k] <= hi]
    if use_fert and not FERT_RANGE[0] <= fertilizer <= FERT_RANGE[1]:
        outside.append(f"Fertilizer ({FERT_RANGE[0]}–{FERT_RANGE[1]})")

    st.session_state["last"] = {
        "top": top,
        "yield": est_yield,
        "outside": outside,
        "inputs": [
            ("N", N), ("P", P), ("K", K), ("pH", ph),
            ("Temp", f"{temperature:g}°C"), ("Humidity", f"{humidity:g}%"), ("Rain", f"{rainfall:g} mm"),
        ],
    }

# --------------------------------------------------------------------------
# Results (shown right under the inputs)
# --------------------------------------------------------------------------
last = st.session_state.get("last")
if last:
    if True:
        st.markdown(
            '<div id="result-anchor"></div><div class="rtitle"><span class="ic">📊</span>Your result</div>',
            unsafe_allow_html=True,
        )
        top, est_yield, outside = last["top"], last["yield"], last["outside"]
        crop, conf = top[0]

        bars = "".join(
            f'<div class="brow"><span class="nm">{CROP_EMOJI.get(n, "🌱")} {n.capitalize()}</span>'
            f'<div class="bar"><div class="fill" style="--w:{p*100:.0f}%"></div></div>'
            f'<span class="pc">{p*100:.0f}%</span></div>'
            for n, p in top
        )
        pos = float(np.clip((est_yield - YIELD_MIN) / (YIELD_MAX - YIELD_MIN), 0, 1)) * 100
        level = "Low" if pos < 33 else "Medium" if pos < 66 else "High"
        chips = "".join(f'<span class="chip">{k} <b>{v}</b></span>' for k, v in last["inputs"])

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
            f'<div class="rvalue dark">{est_yield:.2f} t/ha</div>'
            f'<div class="rsub">{level} compared with the training data</div>'
            '<div class="gauge"><div class="bar">'
            f'<div class="marker" style="--pos:{pos:.0f}%"></div></div>'
            f'<div class="gscale"><span>{YIELD_MIN:.1f}</span><span>{YIELD_MAX:.1f}</span></div></div>'
            '</div>'
            '</div>'
            f'<div class="chips">{chips}</div>'
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

# Scroll down to the result right after a new prediction
if go:
    try:
        import streamlit.components.v1 as components

        components.html(
            "<script>try{setTimeout(function(){const e=window.parent.document.getElementById('result-anchor');"
            "if(e){e.scrollIntoView({behavior:'smooth',block:'start'});}},350);}catch(e){}</script>",
            height=0,
        )
    except Exception:
        pass

st.markdown('<div class="footer">Smart Agricultural Advisor</div>', unsafe_allow_html=True)
