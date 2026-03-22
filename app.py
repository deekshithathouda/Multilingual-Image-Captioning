# ============================================================
# app.py — Upload + Camera + Neon Button Effects
# ============================================================

import streamlit as st
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from deep_translator import GoogleTranslator
import torch

st.set_page_config(
    page_title="Multilingual Image Captioning",
    page_icon="✦",
    layout="centered",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,300&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
html, body, [data-testid="stAppViewContainer"] {
    background: #0a0a0f; color: #f0ede8; font-family: 'DM Sans', sans-serif;
}
[data-testid="stAppViewContainer"] {
    background:
        radial-gradient(ellipse 80% 60% at 20% 10%, rgba(255,140,60,0.08) 0%, transparent 60%),
        radial-gradient(ellipse 60% 50% at 80% 80%, rgba(100,200,255,0.06) 0%, transparent 60%),
        #0a0a0f;
}
[data-testid="stHeader"], [data-testid="stToolbar"],
footer, #MainMenu { display: none !important; visibility: hidden !important; }
.block-container { padding: 0 2rem 4rem !important; max-width: 860px !important; }

/* Splash */
#splash {
    position: fixed; inset: 0; background: #0a0a0f; z-index: 9999;
    display: flex; flex-direction: column;
    align-items: center; justify-content: center; gap: 1.2rem;
    animation: splashFade 0.6s ease 3.5s forwards;
}
@keyframes splashFade { to { opacity: 0; pointer-events: none; } }
.splash-title {
    font-family: 'Syne', sans-serif; font-size: clamp(2rem, 6vw, 3.8rem);
    font-weight: 800; text-align: center; line-height: 1.1;
    letter-spacing: -0.03em; color: #f0ede8;
    opacity: 0; animation: riseIn 0.8s ease 0.3s forwards;
}
.splash-title span {
    background: linear-gradient(135deg, #ff8c3c 0%, #ffb347 50%, #64c8ff 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
}
.splash-sub {
    font-size: 1rem; font-weight: 300; color: rgba(240,237,232,0.6);
    text-align: center; opacity: 0; animation: riseIn 0.7s ease 0.7s forwards;
}
.splash-bar-wrap {
    width: 180px; height: 2px; background: rgba(255,255,255,0.07);
    border-radius: 2px; margin-top: 0.8rem; overflow: hidden;
    opacity: 0; animation: riseIn 0.5s ease 1s forwards;
}
.splash-bar {
    height: 100%; width: 0%; background: linear-gradient(90deg, #ff8c3c, #64c8ff);
    border-radius: 2px; animation: loadBar 2.2s ease 1.1s forwards;
}
@keyframes loadBar { to { width: 100%; } }
@keyframes riseIn {
    from { opacity: 0; transform: translateY(16px); }
    to   { opacity: 1; transform: translateY(0); }
}
#main-app { opacity: 0; animation: appReveal 0.8s ease 4.1s forwards; }
@keyframes appReveal {
    from { opacity: 0; transform: translateY(12px); }
    to   { opacity: 1; transform: translateY(0); }
}

/* Title */
.app-title {
    font-family: 'Syne', sans-serif; font-size: clamp(2rem, 5vw, 3.2rem);
    font-weight: 800; letter-spacing: -0.03em; text-align: center;
    padding: 3.5rem 0 0.5rem; line-height: 1.1;
}
.app-title span {
    background: linear-gradient(135deg, #ff8c3c 0%, #ffb347 50%, #64c8ff 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
}
.app-subtitle {
    text-align: center; font-size: 0.95rem; font-weight: 400;
    color: rgba(240,237,232,0.85); letter-spacing: 0.04em; margin-bottom: 2.8rem;
}
.sec-label {
    font-size: 0.75rem; font-weight: 700; letter-spacing: 0.2em;
    text-transform: uppercase; color: rgba(240,237,232,0.9); margin-bottom: 0.7rem;
}

/* File uploader */
[data-testid="stFileUploader"] { background: transparent !important; }
[data-testid="stFileUploader"] > div {
    background: rgba(255,255,255,0.92) !important;
    border: 1px dashed rgba(255,140,60,0.5) !important; border-radius: 16px !important;
}
[data-testid="stFileUploader"] > div:hover {
    border-color: #ff8c3c !important; background: rgba(255,255,255,0.97) !important;
}
[data-testid="stFileUploader"] label, [data-testid="stFileUploader"] small,
[data-testid="stFileUploader"] span, [data-testid="stFileUploader"] p,
[data-testid="stFileUploader"] div,
[data-testid="stFileUploaderDropzoneInstructions"] span,
[data-testid="stFileUploaderDropzoneInstructions"] small {
    color: #111111 !important; font-family: 'DM Sans', sans-serif !important;
    font-size: 0.9rem !important; font-weight: 500 !important;
}
[data-testid="stFileUploader"] button {
    background: #ff8c3c !important; color: #fff !important;
    border: none !important; border-radius: 8px !important; font-weight: 600 !important;
}

/* Camera input */
[data-testid="stCameraInput"] > div {
    background: transparent !important;
    border: none !important; box-shadow: none !important; padding: 0 !important;
}
[data-testid="stCameraInput"] video {
    border-radius: 16px !important;
    border: 1px solid rgba(100,200,255,0.3) !important;
    width: 100% !important;
}
[data-testid="stCameraInput"] img {
    border-radius: 16px !important; width: 100% !important;
}
[data-testid="stCameraInput"] button {
    background: rgba(255,60,60,0.1) !important; color: #ff6464 !important;
    border: 1px solid rgba(255,60,60,0.3) !important; border-radius: 10px !important;
    font-family: 'DM Sans', sans-serif !important; font-weight: 600 !important;
    width: 100% !important; padding: 0.6rem !important; margin-top: 0.5rem !important;
}
[data-testid="stCameraInput"] button:hover {
    background: rgba(255,60,60,0.2) !important; border-color: #ff6464 !important;
}

/* ALL buttons base style */
.stButton > button {
    width: 100% !important;
    background: linear-gradient(135deg, #ff8c3c, #ffb347) !important;
    color: #0a0a0f !important; border: none !important; border-radius: 14px !important;
    font-family: 'Syne', sans-serif !important; font-weight: 700 !important;
    font-size: 0.95rem !important; letter-spacing: 0.03em !important;
    padding: 0.85rem !important; cursor: pointer !important;
    transition: all 0.25s ease !important;
    box-shadow: 0 4px 28px rgba(255,140,60,0.28) !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 36px rgba(255,140,60,0.42) !important;
}
.stButton > button:active { transform: translateY(0) !important; }

/* Image */
[data-testid="stImage"] img {
    border-radius: 18px !important;
    border: 1px solid rgba(255,255,255,0.1) !important; margin: 1rem 0 !important;
}

/* Captions */
.captions-header {
    font-family: 'Syne', sans-serif; font-size: 0.75rem; font-weight: 700;
    letter-spacing: 0.2em; text-transform: uppercase; color: rgba(240,237,232,0.9);
    margin: 1.8rem 0 1rem; padding-bottom: 0.5rem;
    border-bottom: 1px solid rgba(255,255,255,0.07);
}
.caption-item {
    background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.07);
    border-radius: 14px; padding: 1.2rem 1.4rem; margin-bottom: 0.8rem;
    position: relative; animation: cardIn 0.4s ease forwards; transition: border-color 0.2s;
}
.caption-item:hover { border-color: rgba(255,140,60,0.3); }
@keyframes cardIn {
    from { opacity: 0; transform: translateY(8px); }
    to   { opacity: 1; transform: translateY(0); }
}
.caption-num {
    display: inline-block; font-family: 'Syne', sans-serif;
    font-size: 0.65rem; font-weight: 700; letter-spacing: 0.1em;
    color: #0a0a0f; background: linear-gradient(135deg, #ff8c3c, #ffb347);
    border-radius: 100px; padding: 0.15rem 0.55rem; margin-bottom: 0.5rem;
}
.caption-en {
    font-family: 'Syne', sans-serif; font-size: 1.05rem;
    font-weight: 600; color: #f0ede8; line-height: 1.5;
}

/* Generate button neon pulse */
.gen-wrap .stButton > button {
    background: linear-gradient(135deg, #ff8c3c, #ffb347) !important;
    animation: neonPulse 2s ease-in-out infinite !important;
}
@keyframes neonPulse {
    0%   { box-shadow: 0 0 8px  rgba(255,140,60,0.4), 0 0 16px rgba(255,140,60,0.2); }
    50%  { box-shadow: 0 0 18px rgba(255,140,60,0.8), 0 0 35px rgba(255,140,60,0.5), 0 0 55px rgba(255,140,60,0.25); }
    100% { box-shadow: 0 0 8px  rgba(255,140,60,0.4), 0 0 16px rgba(255,140,60,0.2); }
}
.gen-wrap .stButton > button:hover {
    animation: none !important;
    box-shadow: 0 0 22px rgba(255,140,60,0.9), 0 0 45px rgba(255,140,60,0.6), 0 0 70px rgba(255,140,60,0.3) !important;
    transform: translateY(-2px) !important;
}

/* Footer */
.app-footer {
    text-align: center; margin-top: 3.5rem; padding-top: 1.5rem;
    border-top: 1px solid rgba(255,255,255,0.07); font-size: 0.72rem;
    letter-spacing: 0.12em; text-transform: uppercase; color: rgba(240,237,232,0.4);
}
.footer-star { color: #ff8c3c; }
</style>
""", unsafe_allow_html=True)

# ── Splash ────────────────────────────────────────────────────
st.markdown("""
<div id="splash">
    <div class="splash-title">
        Welcome to<br><span>Multilingual Image</span><br>Captioning System
    </div>
    <div class="splash-sub">English &nbsp;·&nbsp; हिन्दी &nbsp;·&nbsp; తెలుగు</div>
    <div class="splash-bar-wrap"><div class="splash-bar"></div></div>
</div>
""", unsafe_allow_html=True)

# ── Load model ────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    proc = BlipProcessor.from_pretrained(
               "Salesforce/blip-image-captioning-base")
    mdl  = BlipForConditionalGeneration.from_pretrained(
               "Salesforce/blip-image-captioning-base",
               torch_dtype      = torch.float16,
               low_cpu_mem_usage= True
           )
    mdl.eval()
    return proc, mdl, "cpu"

processor, model, device = load_model()

def generate_captions(img):
    inputs = processor(images=img, return_tensors="pt").to(device)
    captions = []
    with torch.no_grad():
        # FASTER configs — reduced beams and sampling
        configs = [
            dict(max_length=30, num_beams=2, early_stopping=True),
            dict(max_length=20, num_beams=2, early_stopping=True),
            dict(max_length=30, do_sample=True, temperature=0.7, top_k=30),
            dict(max_length=30, do_sample=True, temperature=0.9, top_k=50),
            dict(max_length=30, do_sample=True, temperature=1.1, top_k=80),
        ]
        for cfg in configs:
            out = model.generate(**inputs, **cfg)
            captions.append(
                processor.decode(out[0], skip_special_tokens=True).strip())
    seen, unique = set(), []
    for cap in captions:
        if cap.lower().strip() not in seen:
            seen.add(cap.lower().strip())
            unique.append(cap)
    return unique

def translate_text(text, code):
    if code == "en": return None
    try:
        return GoogleTranslator(source="en", target=code).translate(text)
    except:
        return "[Translation unavailable]"

# ── Session state ─────────────────────────────────────────────
if "language"     not in st.session_state: st.session_state.language     = "English"
if "show_camera"  not in st.session_state: st.session_state.show_camera  = False
if "camera_image" not in st.session_state: st.session_state.camera_image = None

lang_map = {"English":"en", "Hindi":"hi", "Telugu":"te"}

# ── App ───────────────────────────────────────────────────────
st.markdown('<div id="main-app">', unsafe_allow_html=True)

st.markdown("""
<div class="app-title">Multilingual<br><span>Image Captioning</span></div>
<div class="app-subtitle">
    Upload &nbsp;·&nbsp; Caption &nbsp;·&nbsp; Translate &nbsp; ✦ &nbsp;
    English &nbsp;·&nbsp; हिन्दी &nbsp;·&nbsp; తెలుగు
</div>
""", unsafe_allow_html=True)

# ── Upload + Camera icon ──────────────────────────────────────
st.markdown('<div class="sec-label">Upload Image</div>', unsafe_allow_html=True)

upload_col, cam_col = st.columns([11, 1])
with upload_col:
    uploaded = st.file_uploader("", type=["jpg","jpeg","png"],
                                label_visibility="collapsed")
with cam_col:
    if st.button("📷", use_container_width=True, help="Open Camera"):
        st.session_state.show_camera  = not st.session_state.show_camera
        st.session_state.camera_image = None
        st.rerun()

# ── Camera section ────────────────────────────────────────────
image = None

if st.session_state.show_camera and not uploaded:
    st.markdown("""
    <div style='font-size:0.7rem; font-weight:700; letter-spacing:0.2em;
         text-transform:uppercase; color:#64c8ff; margin-bottom:0.6rem;'>
        📷 &nbsp; Click On Take Photo
    </div>
    """, unsafe_allow_html=True)

    camera_photo = st.camera_input("", label_visibility="collapsed",
                                   key="camera_widget")
    if camera_photo:
        st.session_state.camera_image = camera_photo
        st.session_state.show_camera  = False
        st.rerun()

# ── Determine image source ────────────────────────────────────
if uploaded:
    image = Image.open(uploaded).convert("RGB")
    st.session_state.camera_image = None
elif st.session_state.camera_image:
    image = Image.open(st.session_state.camera_image).convert("RGB")

# ── Show image once ───────────────────────────────────────────
if image:
    st.image(image, use_container_width=True)
    if st.session_state.camera_image and not uploaded:
        if st.button("✕  Clear Photo", use_container_width=False):
            st.session_state.camera_image = None
            st.session_state.show_camera  = False
            st.rerun()

# ── Language buttons with NEON highlight ─────────────────────
st.markdown('<div class="sec-label" style="margin-top:1.6rem">Select Language</div>',
            unsafe_allow_html=True)

# Neon glow for selected button
neon = "background:rgba(255,140,60,0.2)!important;border:1.5px solid #ff8c3c!important;color:#ff8c3c!important;box-shadow:0 0 10px rgba(255,140,60,0.6),0 0 22px rgba(255,140,60,0.35),0 0 40px rgba(255,140,60,0.15)!important;text-shadow:0 0 8px rgba(255,140,60,0.8)!important;"
none = ""

en_s = neon if st.session_state.language == "English" else none
hi_s = neon if st.session_state.language == "Hindi"   else none
te_s = neon if st.session_state.language == "Telugu"  else none

st.markdown(f"""
<style>
/* Language neon highlight — targets the language button row */
div[data-testid="stHorizontalBlock"].lang-row
    div[data-testid="column"]:nth-child(1) .stButton > button {{ {en_s} }}
div[data-testid="stHorizontalBlock"].lang-row
    div[data-testid="column"]:nth-child(2) .stButton > button {{ {hi_s} }}
div[data-testid="stHorizontalBlock"].lang-row
    div[data-testid="column"]:nth-child(3) .stButton > button {{ {te_s} }}
</style>
""", unsafe_allow_html=True)

# Wrap in identifiable div
st.markdown('<div class="lang-row">', unsafe_allow_html=True)
c1, c2, c3 = st.columns(3)
with c1:
    if st.button("English", use_container_width=True, key="btn_en"):
        st.session_state.language = "English"; st.rerun()
with c2:
    if st.button("Hindi",   use_container_width=True, key="btn_hi"):
        st.session_state.language = "Hindi";   st.rerun()
with c3:
    if st.button("Telugu",  use_container_width=True, key="btn_te"):
        st.session_state.language = "Telugu";  st.rerun()
st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<div style='height:0.8rem'></div>", unsafe_allow_html=True)

# ── Generate button with neon pulse ──────────────────────────
st.markdown('<div class="gen-wrap">', unsafe_allow_html=True)
gen_btn = st.button("✦  Generate Captions", use_container_width=True, key="btn_gen")
st.markdown("</div>", unsafe_allow_html=True)

# ── Output ────────────────────────────────────────────────────
if image and gen_btn:
    selected_code = lang_map[st.session_state.language]
    captions      = generate_captions(image)

    st.markdown('<div class="captions-header">✦ &nbsp; Generated Captions</div>',
                unsafe_allow_html=True)

    for i, cap in enumerate(captions):
        if selected_code == "en":
            display_text = cap
        else:
            translated   = translate_text(cap, selected_code)
            display_text = translated if translated and \
                           translated != "[Translation unavailable]" else cap

        st.markdown(
            '<div class="caption-item">'
            '<div class="caption-num">Caption ' + str(i+1) + '</div>'
            '<div class="caption-en">' + display_text + '</div>'
            '</div>',
            unsafe_allow_html=True
        )

elif not image and gen_btn:
    st.warning("Please upload an image or use the camera 📷")

st.markdown("""
<div class="app-footer">
    B.Tech CSE-AIML Final Year Project
    <span class="footer-star"> ✦ </span>
    Image Captioning
</div>
""", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)
