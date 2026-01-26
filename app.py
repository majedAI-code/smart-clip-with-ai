import streamlit as st
import os
import json
import time
import google.generativeai as genai
from elevenlabs.client import ElevenLabs
from moviepy.editor import VideoFileClip, AudioFileClip

# --- 1. إعدادات الصفحة والهوية ---
st.set_page_config(
    page_title="استوديو المحتوى الذكي",
    page_icon="🎬",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# دالة لعرض الصور (الهوية البصرية)
def render_header(image_name, alt_text):
    if os.path.exists(image_name):
        st.image(image_name, use_column_width=True)
    else:
        st.header(alt_text)

# --- 2. تحميل المفاتيح ---
def load_api_keys():
    try:
        GOOGLE_KEY = st.secrets["GOOGLE_API_KEY"]
        ELEVEN_KEY = st.secrets["ELEVENLABS_API_KEY"]
        return GOOGLE_KEY, ELEVEN_KEY
    except:
        st.error("⚠️ يرجى التأكد من وضع المفاتيح في ملف secrets.toml")
        st.stop()

GOOGLE_API_KEY, ELEVENLABS_API_KEY = load_api_keys()

genai.configure(api_key=GOOGLE_API_KEY)
eleven_client = ElevenLabs(api_key=ELEVENLABS_API_KEY)

# --- 3. دالة اختيار موديل Gemini المتاح (حل مشكلة NotFound) ---
@st.cache_resource
def get_working_model_name():
    candidates = [
        "gemini-1.5-flash", 
        "models/gemini-1.5-flash", 
        "gemini-1.5-pro", 
        "models/gemini-1.5-pro", 
        "gemini-pro"
    ]
    try:
        available_models = [m.name for m in genai.list_models()]
        for c in candidates:
            if c in available_models or f"models/{c}" in available_models:
                return c
    except:
        pass
    return "gemini-1.5-flash"

CURRENT_MODEL_NAME = get_working_model_name()

# --- 4. إدارة الذاكرة (Session State) ---
if 'analysis_done' not in st.session_state: st.session_state['analysis_done'] = False
if 'clips_data' not in st.session_state: st.session_state['clips_data'] = []
if 'dubbed_video' not in st.session_state: st.session_state['dubbed_video'] = None
if 'generated_clips' not in st.session_state: st.session_state['generated_clips'] = []

# --- 5. دوال المعالجة (محسنة للسرعة) ---

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
        prompt = "Listen to the voice. Identify the gender of the MAIN speaker. Return ONLY 'Male' or 'Female'."
        response = model.generate_content([prompt, {"mime_type": "audio/mp3", "data": audio_data}])
        if "female" in response.text.strip().lower(): return "female"
        return "male"
    except: return "male"

def transcribe_and_translate(audio_path, target_lang):
    model = genai.GenerativeModel(CURRENT_MODEL_NAME)
    try:
        with open(audio_path, "rb") as f: audio_data = f.read()
        prompt = f"Transcribe the speech and translate it to {target_lang}. Return ONLY the translated text."
        response = model.generate_content([prompt, {"mime_type": "audio/mp3", "data": audio_data}])
        return response.text
    except: return None

def generate_dubbed_audio(text, voice_id):
    try:
        audio_generator = eleven_client.text_to_speech.convert(
            text=text, 
            voice_id=voice_id, 
            model_id="eleven_multilingual_v2"
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
    
    # قص الفيديو إذا كان الصوت الجديد أقصر
    if new_audio.duration < video.duration:
        final_video = final_video.subclip(0, new_audio.duration)
    
    output_path = "final_dubbed_video.mp4"
    
    # --- تحسين السرعة هنا (Ultrafast) ---
    final_video.write_videofile(
        output_path, 
        codec="libx264", 
        audio_codec="aac", 
        preset="ultrafast",  # السر في السرعة
        threads=4,          # استخدام كل الأنوية
        logger=None
    )
    
    video.close()
    new_audio.close()
    return output_path

def analyze_video_for_clips(video_path):
    # مرحلة التحليل فقط
    model = genai.GenerativeModel(CURRENT_MODEL_NAME)
    try:
        myfile = genai.upload_file(video_path)
        while myfile.state.name == "PROCESSING":
            time.sleep(1)
            myfile = genai.get_file(myfile.name)
            
        prompt = """
        Analyze this video. Identify the MOST viral and engaging segments (Shorts/Reels).
        - Duration: 15 to 60 seconds.
        - Return a valid JSON list: [{"start": 10, "end": 40, "label": "Topic Name"}, ...]
        - If no good clips found, return empty list.
        """
        response = model.generate_content([prompt, myfile])
        text = response.text.replace("```json", "").replace("```", "").strip()
        return json.loads(text)
    except: return []

def cut_clips_processing(original_video_path, timestamps):
    # مرحلة التنفيذ (القص السريع)
    video = VideoFileClip(original_video_path)
    generated_files = []
    for i, item in enumerate(timestamps):
        try:
            start, end = item.get('start'), item.get('end')
            label = item.get('label', f'Clip {i+1}')
            clip = video.subclip(start, end)
            name = f"clip_{i}_{label}.mp4"
            
            # --- تحسين السرعة هنا أيضاً ---
            clip.write_videofile(
                name, 
                codec="libx264", 
                audio_codec="aac", 
                preset="ultrafast",
                threads=4,
                logger=None
            )
            generated_files.append({"path": name, "label": label})
        except: continue
    video.close()
    return generated_files

# --- 6. واجهة المستخدم (UI) ---

# البانر
render_header("banner.jpg", "استوديو المحتوى الذكي")
st.caption("أتمتة صناعة المحتوى الإعلامي باستخدام الذكاء الاصطناعي التوليدي")

# قسم الرفع
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
        st.error(f"⚠️ الفيديو طويل جداً ({int(dur/60)} دقيقة). يرجى استخدام فيديو أقصر للعرض.")
    else:
        st.video(video_path)
        st.divider()
        
        # --- المرحلة 1: الخيارات والتحليل ---
        c1, c2 = st.columns(2)
        
        with c1:
            render_header("dubbing.png", "🎙️ الدبلجة")
            enable_dubbing = st.checkbox("تفعيل الدبلجة")
            
            # القائمة الكاملة (29 لغة)
            all_languages = [
                "Arabic", "English", "French", "Spanish", "German", 
                "Chinese", "Japanese", "Hindi", "Italian", "Portuguese", 
                "Russian", "Turkish", "Korean", "Dutch", "Swedish", 
                "Indonesian", "Vietnamese", "Filipino", "Ukrainian", 
                "Greek", "Czech", "Finnish", "Romanian", "Danish", 
                "Bulgarian", "Malay", "Slovak", "Croatian", "Polish"
            ]
            target_lang = st.selectbox("اللغة المستهدفة", all_languages)
            
        with c2:
            render_header("clipping.png", "✂️ القص الذكي")
            enable_clipping = st.checkbox("استخراج المقاطع")

        # زر التحليل
        if st.button("🔍 تحليل الفيديو (المعاينة)", use_container_width=True):
            st.session_state['analysis_done'] = False
            st.session_state['clips_data'] = []
            
            with st.spinner("جاري تحليل المحتوى بواسطة Gemini AI..."):
                if enable_clipping:
                    clips_found = analyze_video_for_clips(video_path)
                    st.session_state['clips_data'] = clips_found
                st.session_state['analysis_done'] = True

        # --- المرحلة 2: النتائج والتنفيذ ---
        if st.session_state['analysis_done']:
            st.divider()
            st.info("📊 تقرير التحليل:")
            
            if enable_dubbing:
                st.write("✅ خدمة الدبلجة: جاهزة.")
            
            if enable_clipping:
                count = len(st.session_state['clips_data'])
                if count > 0:
                    st.success(f"وجدنا {count} مقاطع مرشحة للانتشار.")
                    st.json(st.session_state['clips_data'])
                else:
                    st.warning("لم يتم العثور على مقاطع قوية، لكن يمكنك المتابعة.")
            
            st.divider()
            
            # زر التنفيذ النهائي
            if st.button("🚀 تنفيذ العمليات (Start Processing)", type="primary", use_container_width=True):
                st.session_state['dubbed_video'] = None
                st.session_state['generated_clips'] = []
                
                with st.status("جاري المعالجة في الاستوديو...", expanded=True) as status:
                    
                    # 1. تنفيذ الدبلجة
                    if enable_dubbing:
                        status.write("🎙️ جاري استخراج الصوت وتحديد الهوية...")
                        aud = extract_audio(video_path)
                        gend = detect_speaker_gender(aud)
                        
                        status.write(f"📝 جاري الترجمة ({target_lang})...")
                        txt = transcribe_and_translate(aud, target_lang)
                        
                        if txt:
                            status.write("🗣️ جاري توليد الصوت الجديد (ElevenLabs)...")
                            voice = "21m00Tcm4TlvDq8ikWAM" if gend == "female" else "pNInz6obpgDQGcFmaJgB"
                            dub = generate_dubbed_audio(txt, voice)
                            
                            if dub:
                                status.write("🎬 جاري دمج الصوت مع الفيديو (قد يستغرق وقتاً حسب الحجم)...")
                                final_dub = merge_audio_video(video_path, dub)
                                st.session_state['dubbed_video'] = final_dub
                    
                    # 2. تنفيذ القص
                    if enable_clipping and st.session_state['clips_data']:
                        status.write("✂️ جاري قص المقاطع وتصديرها...")
                        clips = cut_clips_processing(video_path, st.session_state['clips_data'])
                        st.session_state['generated_clips'] = clips
                    
                    status.update(label="✅ تمت العملية بنجاح!", state="complete")

# --- العرض النهائي ---
if st.session_state['dubbed_video']:
    st.header("🎥 الفيديو المدبلج")
    st.video(st.session_state['dubbed_video'])
    with open(st.session_state['dubbed_video'], "rb") as f:
         st.download_button("تحميل الفيديو المدبلج", f, file_name="dubbed_video.mp4")

if st.session_state['generated_clips']:
    st.header("🔥 المقاطع المستخرجة")
    for clip in st.session_state['generated_clips']:
        st.write(f"**📌 {clip['label']}**")
        st.video(clip['path'])
        with open(clip['path'], "rb") as f:
            st.download_button(f"تحميل {clip['label']}", f, file_name=clip['path'])