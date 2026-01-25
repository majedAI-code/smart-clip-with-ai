import streamlit as st
import os
import json
import time
import google.generativeai as genai
from elevenlabs.client import ElevenLabs
from moviepy.editor import VideoFileClip, AudioFileClip

# --- إعدادات الصفحة ---
st.set_page_config(
    page_title="استوديو المحتوى الذكي",
    page_icon="🎬",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# --- دوال بصرية ---
def render_header(image_name, alt_text):
    if os.path.exists(image_name):
        st.image(image_name, use_column_width=True)
    else:
        st.header(alt_text)

# --- تحميل المفاتيح ---
def load_api_keys():
    try:
        GOOGLE_KEY = st.secrets["GOOGLE_API_KEY"]
        ELEVEN_KEY = st.secrets["ELEVENLABS_API_KEY"]
        return GOOGLE_KEY, ELEVEN_KEY
    except:
        st.error("⚠️ يرجى وضع المفاتيح في Secrets")
        st.stop()

GOOGLE_API_KEY, ELEVENLABS_API_KEY = load_api_keys()

genai.configure(api_key=GOOGLE_API_KEY)
eleven_client = ElevenLabs(api_key=ELEVENLABS_API_KEY)

# --- (الحل الجذري) دالة اكتشاف الموديل الصحيح ---
@st.cache_resource
def get_working_model_name():
    # قائمة الموديلات المحتملة
    candidates = [
        "gemini-1.5-flash", 
        "models/gemini-1.5-flash", 
        "gemini-1.5-pro", 
        "models/gemini-1.5-pro", 
        "gemini-pro"
    ]
    try:
        # نسأل جوجل: ما هي الموديلات المتاحة لهذا المفتاح؟
        available_models = [m.name for m in genai.list_models()]
        for c in candidates:
            if c in available_models or f"models/{c}" in available_models:
                return c
    except:
        pass
    return "gemini-1.5-flash" # احتياطي

CURRENT_MODEL_NAME = get_working_model_name()

# --- إدارة الذاكرة (Session State) ---
if 'analysis_done' not in st.session_state: st.session_state['analysis_done'] = False
if 'clips_data' not in st.session_state: st.session_state['clips_data'] = []
if 'dubbed_video' not in st.session_state: st.session_state['dubbed_video'] = None
if 'generated_clips' not in st.session_state: st.session_state['generated_clips'] = []

# --- دوال المعالجة ---

def check_video_duration(video_path, max_minutes=5):
    try:
        clip = VideoFileClip(video_path)
        dur = clip.duration
        clip.close()
        if dur > max_minutes * 60: return False, dur
        return True, dur
    except: return True, 0

def extract_audio(video_path):
    video = VideoFileClip(video_path)
    audio_path = "temp_audio.mp3"
    video.audio.write_audiofile(audio_path, logger=None)
    video.close()
    return audio_path

def detect_speaker_gender(audio_path):
    model = genai.GenerativeModel(CURRENT_MODEL_NAME)
    try:
        with open(audio_path, "rb") as f: audio_data = f.read()
        prompt = "Identify the gender of the MAIN speaker. Return 'Male' or 'Female'."
        response = model.generate_content([prompt, {"mime_type": "audio/mp3", "data": audio_data}])
        if "female" in response.text.strip().lower(): return "female"
        return "male"
    except: return "male"

def transcribe_and_translate(audio_path, target_lang):
    model = genai.GenerativeModel(CURRENT_MODEL_NAME)
    try:
        with open(audio_path, "rb") as f: audio_data = f.read()
        prompt = f"Transcribe and translate to {target_lang}. Return ONLY the translation text."
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
    final_video.write_videofile(output_path, codec="libx264", audio_codec="aac", logger=None)
    video.close()
    new_audio.close()
    return output_path

def analyze_video_for_clips(video_path):
    """
    هذه الدالة فقط تحلل وتخبرنا بعدد المقاطع دون قص
    """
    model = genai.GenerativeModel(CURRENT_MODEL_NAME)
    try:
        myfile = genai.upload_file(video_path)
        while myfile.state.name == "PROCESSING":
            time.sleep(1)
            myfile = genai.get_file(myfile.name)
            
        prompt = """
        Analyze the video. Identify MOST viral segments (15-60s).
        Return valid JSON only: [{"start": 10, "end": 40, "label": "Topic"}]
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
            clip.write_videofile(name, codec="libx264", audio_codec="aac", logger=None)
            generated_files.append({"path": name, "label": label})
        except: continue
    video.close()
    return generated_files

# --- الواجهة (UI) ---

render_header("banner.jpg", "استوديو المحتوى الذكي")
st.caption("أتمتة صناعة المحتوى الإعلامي باستخدام الذكاء الاصطناعي التوليدي")

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
        st.success("تم تحميل الفيديو التجريبي!")

if video_path:
    # فحص المدة
    valid, dur = check_video_duration(video_path, 5)
    if not valid:
        st.error(f"الفيديو طويل ({int(dur/60)} دقيقة). الحد الأقصى 5 دقائق.")
    else:
        st.video(video_path)
        st.divider()
        
        # --- المرحلة 1: الخيارات والتحليل ---
        c1, c2 = st.columns(2)
        with c1:
            render_header("dubbing.png", "🎙️ الدبلجة")
            enable_dubbing = st.checkbox("تفعيل الدبلجة")
            target_lang = st.selectbox("اللغة", ["Arabic", "English", "French"])
        with c2:
            render_header("clipping.png", "✂️ القص الذكي")
            enable_clipping = st.checkbox("استخراج المقاطع")

        # زر التحليل الأولي
        if st.button("🔍 تحليل الفيديو (كم مقطع؟)", use_container_width=True):
            st.session_state['analysis_done'] = False # تصفير الحالة القديمة
            st.session_state['clips_data'] = []
            
            with st.spinner("جاري سؤال الذكاء الاصطناعي عن المقاطع المناسبة..."):
                # تحليل القص
                if enable_clipping:
                    clips_found = analyze_video_for_clips(video_path)
                    st.session_state['clips_data'] = clips_found
                
                # حفظ حالة أن التحليل تم
                st.session_state['analysis_done'] = True

        # --- المرحلة 2: عرض النتيجة والتنفيذ ---
        if st.session_state['analysis_done']:
            st.divider()
            st.info("📊 نتيجة التحليل:")
            
            # تقرير الدبلجة
            if enable_dubbing:
                st.write("✅ الدبلجة: جاهزة للتنفيذ.")
            
            # تقرير القص (هنا يظهر العدد قبل التنفيذ)
            if enable_clipping:
                count = len(st.session_state['clips_data'])
                if count > 0:
                    st.success(f"وجدنا {count} مقاطع مرشحة للانتشار.")
                    st.json(st.session_state['clips_data']) # عرض التفاصيل بشفافية
                else:
                    st.warning("لم يجد الذكاء الاصطناعي مقاطع قوية، لكن يمكنك المتابعة.")
            
            st.divider()
            
            # زر التنفيذ النهائي
            if st.button("🚀 تنفيذ القص والدبلجة الآن", type="primary", use_container_width=True):
                with st.status("جاري المعالجة النهائية...", expanded=True) as status:
                    
                    # 1. تنفيذ الدبلجة
                    if enable_dubbing:
                        status.write("🎙️ جاري الدبلجة...")
                        aud = extract_audio(video_path)
                        gend = detect_speaker_gender(aud)
                        txt = transcribe_and_translate(aud, target_lang)
                        if txt:
                            voice = "21m00Tcm4TlvDq8ikWAM" if gend == "female" else "pNInz6obpgDQGcFmaJgB"
                            dub = generate_dubbed_audio(txt, voice)
                            if dub: st.session_state['dubbed_video'] = merge_audio_video(video_path, dub)
                    
                    # 2. تنفيذ القص (بناء على التحليل السابق)
                    if enable_clipping and st.session_state['clips_data']:
                        status.write("✂️ جاري قص المقاطع...")
                        st.session_state['generated_clips'] = cut_clips_processing(video_path, st.session_state['clips_data'])
                    
                    status.update(label="✅ تمت العملية بنجاح!", state="complete")

# --- عرض النتائج النهائية ---
if st.session_state['dubbed_video']:
    st.header("🎥 الفيديو المدبلج")
    st.video(st.session_state['dubbed_video'])

if st.session_state['generated_clips']:
    st.header("🔥 المقاطع المستخرجة")
    for i, clip in enumerate(st.session_state['generated_clips']):
        st.write(f"**{clip['label']}**")
        st.video(clip['path'])