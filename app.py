import streamlit as st
import os
import json
import time
import google.generativeai as genai
from elevenlabs.client import ElevenLabs
from moviepy.editor import VideoFileClip, AudioFileClip

# --- 1. إعدادات الصفحة ---
st.set_page_config(
    page_title="استوديو المحتوى الذكي",
    page_icon="🎬",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# دالة عرض الهوية
def render_header(image_name, alt_text):
    if os.path.exists(image_name):
        st.image(image_name, use_column_width=True)
    else:
        st.header(alt_text)

# --- 2. المفاتيح ---
def load_api_keys():
    try:
        GOOGLE_KEY = st.secrets["GOOGLE_API_KEY"]
        ELEVEN_KEY = st.secrets["ELEVENLABS_API_KEY"]
        return GOOGLE_KEY, ELEVEN_KEY
    except:
        st.error("⚠️ يرجى التأكد من ملف secrets.toml")
        st.stop()

GOOGLE_API_KEY, ELEVENLABS_API_KEY = load_api_keys()
genai.configure(api_key=GOOGLE_API_KEY)
eleven_client = ElevenLabs(api_key=ELEVENLABS_API_KEY)

# --- 3. اختيار الموديل ---
@st.cache_resource
def get_working_model_name():
    candidates = ["gemini-1.5-flash", "models/gemini-1.5-flash", "gemini-pro"]
    try:
        available = [m.name for m in genai.list_models()]
        for c in candidates:
            if c in available or f"models/{c}" in available:
                return c
    except: pass
    return "gemini-1.5-flash"

CURRENT_MODEL_NAME = get_working_model_name()

# --- 4. إدارة الحالة ---
if 'analysis_done' not in st.session_state: st.session_state['analysis_done'] = False
if 'clips_data' not in st.session_state: st.session_state['clips_data'] = []
if 'dubbed_video' not in st.session_state: st.session_state['dubbed_video'] = None
if 'generated_clips' not in st.session_state: st.session_state['generated_clips'] = []

# --- 5. دوال المعالجة (محسنة للسرعة القصوى) ---

def check_video_duration(video_path, max_minutes=5):
    try:
        clip = VideoFileClip(video_path)
        dur = clip.duration
        clip.close()
        if dur > max_minutes * 60: return False, dur
        return True, dur
    except: return True, 0

def extract_audio(video_path):
    # تحسين كبير: استخراج صوت مضغوط جداً لتسريع الرفع والتحليل
    video = VideoFileClip(video_path)
    audio_path = "temp_audio.mp3"
    # bitrate="32k": جودة منخفضة لكن كافية جداً للكلام (سريعة جداً)
    video.audio.write_audiofile(audio_path, bitrate="32k", fps=16000, logger=None)
    video.close()
    return audio_path

def detect_speaker_gender(audio_path):
    model = genai.GenerativeModel(CURRENT_MODEL_NAME)
    try:
        with open(audio_path, "rb") as f: audio_data = f.read()
        prompt = "Listen. Identify MAIN speaker gender. Return ONLY 'Male' or 'Female'."
        response = model.generate_content([prompt, {"mime_type": "audio/mp3", "data": audio_data}])
        if "female" in response.text.strip().lower(): return "female"
        return "male"
    except: return "male"

def transcribe_and_translate(audio_path, target_lang):
    model = genai.GenerativeModel(CURRENT_MODEL_NAME)
    try:
        with open(audio_path, "rb") as f: audio_data = f.read()
        # برومبت محسن ومختصر
        prompt = f"Transcribe speech and translate to {target_lang}. Return ONLY translated text."
        response = model.generate_content([prompt, {"mime_type": "audio/mp3", "data": audio_data}])
        return response.text
    except: return None

def generate_dubbed_audio(text, voice_id):
    try:
        audio_generator = eleven_client.text_to_speech.convert(
            text=text, voice_id=voice_id, model_id="eleven_multilingual_v2"
        )
        save_path = "dubbed_audio.mp3"
        with open(save_path, "wb") as f:
            for chunk in audio_generator: f.write(chunk)
        return save_path
    except: return None

def merge_audio_video(video_path, audio_path):
    video = VideoFileClip(video_path)
    new_audio = AudioFileClip(audio_path)
    final_video = video.set_audio(new_audio)
    if new_audio.duration < video.duration:
        final_video = final_video.subclip(0, new_audio.duration)
    output_path = "final_dubbed_video.mp4"
    # استخدام preset ultrafast
    final_video.write_videofile(output_path, codec="libx264", audio_codec="aac", preset="ultrafast", threads=4, logger=None)
    video.close()
    new_audio.close()
    return output_path

def analyze_video_for_clips(video_path):
    model = genai.GenerativeModel(CURRENT_MODEL_NAME)
    try:
        myfile = genai.upload_file(video_path)
        while myfile.state.name == "PROCESSING":
            time.sleep(1)
            myfile = genai.get_file(myfile.name)
        prompt = """
        Identify 1-3 MOST viral shorts (15-60s).
        Return valid JSON: [{"start": 10, "end": 40, "label": "Topic"}]
        """
        response = model.generate_content([prompt, myfile])
        text = response.text.replace("```json", "").replace("```", "").strip()
        return json.loads(text)
    except: return []

def cut_clips_processing(original_video_path, timestamps):
    video = VideoFileClip(original_video_path)
    generated_files = []
    for i, item in enumerate(timestamps):
        try:
            start, end = item.get('start'), item.get('end')
            label = item.get('label', f'Clip {i+1}')
            clip = video.subclip(start, end)
            name = f"clip_{i}_{label}.mp4"
            clip.write_videofile(name, codec="libx264", audio_codec="aac", preset="ultrafast", threads=4, logger=None)
            generated_files.append({"path": name, "label": label})
        except: continue
    video.close()
    return generated_files

# --- 6. الواجهة ---
render_header("banner.jpg", "استوديو المحتوى الذكي")
st.caption("أتمتة صناعة المحتوى الإعلامي")

st.markdown("### 1. رفع الفيديو")
upload_option = st.radio("المصدر:", ["رفع ملف", "فيديو تجريبي (Demo)"], horizontal=True)
video_path = None

if upload_option == "رفع ملف":
    uploaded_file = st.file_uploader("ملف MP4", type=["mp4"])
    if uploaded_file:
        with open("temp_video.mp4", "wb") as f: f.write(uploaded_file.getbuffer())
        video_path = "temp_video.mp4"
elif upload_option == "فيديو تجريبي (Demo)":
    if st.button("تحميل الفيديو التجريبي") and os.path.exists("sample.mp4"):
        video_path = "sample.mp4"
        st.success("تم التحميل!")

if video_path:
    valid, dur = check_video_duration(video_path, 5)
    if not valid:
        st.error(f"الفيديو طويل ({int(dur/60)} دقيقة).")
    else:
        st.video(video_path)
        st.divider()
        c1, c2 = st.columns(2)
        with c1:
            render_header("dubbing.png", "🎙️ الدبلجة")
            enable_dubbing = st.checkbox("تفعيل الدبلجة")
            # القائمة الشاملة
            all_langs = ["Arabic", "English", "French", "Spanish", "German", "Chinese", "Japanese", "Russian", "Turkish", "Italian"]
            target_lang = st.selectbox("اللغة المستهدفة", all_langs)
        with c2:
            render_header("clipping.png", "✂️ القص الذكي")
            enable_clipping = st.checkbox("استخراج المقاطع")

        if st.button("🔍 تحليل الفيديو (المعاينة)", use_container_width=True):
            st.session_state['analysis_done'] = False
            st.session_state['clips_data'] = []
            with st.spinner("جاري التحليل..."):
                if enable_clipping:
                    st.session_state['clips_data'] = analyze_video_for_clips(video_path)
                st.session_state['analysis_done'] = True

        if st.session_state['analysis_done']:
            st.divider()
            st.info("التقرير:")
            if enable_dubbing: st.write("✅ الدبلجة جاهزة.")
            if enable_clipping:
                count = len(st.session_state['clips_data'])
                if count > 0: st.success(f"وجدنا {count} مقاطع.")
                else: st.warning("لا يوجد مقاطع قوية.")
            
            if st.button("🚀 تنفيذ العمليات", type="primary", use_container_width=True):
                st.session_state['dubbed_video'] = None
                st.session_state['generated_clips'] = []
                with st.status("جاري التنفيذ...", expanded=True) as status:
                    if enable_dubbing:
                        status.write("🎙️ معالجة الصوت...")
                        aud = extract_audio(video_path) # النسخة السريعة
                        gend = detect_speaker_gender(aud)
                        status.write(f"الترجمة ({target_lang})...")
                        txt = transcribe_and_translate(aud, target_lang)
                        if txt:
                            status.write("توليد الصوت (ElevenLabs)...")
                            voice = "21m00Tcm4TlvDq8ikWAM" if gend == "female" else "pNInz6obpgDQGcFmaJgB"
                            dub = generate_dubbed_audio(txt, voice)
                            if dub:
                                status.write("دمج الفيديو...")
                                st.session_state['dubbed_video'] = merge_audio_video(video_path, dub)
                    
                    if enable_clipping and st.session_state['clips_data']:
                        status.write("قص المقاطع...")
                        st.session_state['generated_clips'] = cut_clips_processing(video_path, st.session_state['clips_data'])
                    status.update(label="تم!", state="complete")

if st.session_state['dubbed_video']:
    st.header("النتائج")
    st.video(st.session_state['dubbed_video'])

if st.session_state['generated_clips']:
    st.subheader("المقاطع")
    for clip in st.session_state['generated_clips']:
        st.write(f"**{clip['label']}**")
        st.video(clip['path'])