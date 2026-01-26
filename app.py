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

# --- 3. اختيار الموديل (تثبيت على Flash لضمان السرعة) ---
CURRENT_MODEL_NAME = "gemini-1.5-flash"

# --- 4. إدارة الحالة ---
if 'dubbed_video' not in st.session_state: st.session_state['dubbed_video'] = None
if 'generated_clips' not in st.session_state: st.session_state['generated_clips'] = []

# --- 5. دوال المعالجة (المستقرة) ---

def check_video_duration(video_path, max_minutes=5):
    try:
        clip = VideoFileClip(video_path)
        dur = clip.duration
        clip.close()
        if dur > max_minutes * 60: return False, dur
        return True, dur
    except: return True, 0

def extract_audio(video_path):
    # عدلنا الإعدادات لتكون مقبولة لدى Gemini
    video = VideoFileClip(video_path)
    audio_path = "temp_audio.mp3"
    video.audio.write_audiofile(
        audio_path, 
        bitrate="64k",      # جودة قياسية (أضمن من 32k)
        fps=22050,          # تردد قياسي للصوت البشري
        codec="libmp3lame",
        logger=None
    )
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
        prompt = f"Transcribe the speech and translate it to {target_lang}. Return ONLY the translated text."
        # إيقاف فلاتر الأمان لتجنب حظر المحتوى العادي
        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        ]
        response = model.generate_content(
            [prompt, {"mime_type": "audio/mp3", "data": audio_data}],
            safety_settings=safety_settings
        )
        return response.text
    except Exception as e:
        # طباعة الخطأ لمعرفته
        st.error(f"خطأ في الترجمة: {e}")
        return None

def generate_dubbed_audio(text, voice_id):
    try:
        audio_generator = eleven_client.text_to_speech.convert(
            text=text, voice_id=voice_id, model_id="eleven_multilingual_v2"
        )
        save_path = "dubbed_audio.mp3"
        with open(save_path, "wb") as f:
            for chunk in audio_generator: f.write(chunk)
        return save_path
    except Exception as e:
        st.error(f"خطأ ElevenLabs: {e}")
        return None

def merge_audio_video(video_path, audio_path):
    video = VideoFileClip(video_path)
    new_audio = AudioFileClip(audio_path)
    final_video = video.set_audio(new_audio)
    if new_audio.duration < video.duration:
        final_video = final_video.subclip(0, new_audio.duration)
    output_path = "final_dubbed_video.mp4"
    final_video.write_videofile(output_path, codec="libx264", audio_codec="aac", preset="ultrafast", threads=4, logger=None)
    video.close()
    new_audio.close()
    return output_path

# --- دالة القص (كما طلبت: العدد والمدة) ---
def analyze_and_cut_specific(video_path, num_clips, clip_duration):
    model = genai.GenerativeModel(CURRENT_MODEL_NAME)
    video_clip = VideoFileClip(video_path)
    total_duration = video_clip.duration
    video_clip.close()
    
    try:
        myfile = genai.upload_file(video_path)
        while myfile.state.name == "PROCESSING":
            time.sleep(1)
            myfile = genai.get_file(myfile.name)
            
        prompt = f"""
        Analyze this video. Find exactly {num_clips} best segments.
        Each segment MUST be exactly {clip_duration} seconds long.
        Return valid JSON only: [{{ "start": 10, "end": {10+clip_duration}, "label": "Topic" }}]
        Make sure timestamps are within video duration ({total_duration}s).
        """
        response = model.generate_content([prompt, myfile])
        text = response.text.replace("```json", "").replace("```", "").strip()
        timestamps = json.loads(text)
    except: 
        timestamps = []
        for i in range(num_clips):
            start = i * clip_duration
            if start + clip_duration > total_duration: break
            timestamps.append({"start": start, "end": start + clip_duration, "label": f"Clip {i+1}"})

    video = VideoFileClip(video_path)
    generated_files = []
    
    for i, item in enumerate(timestamps):
        try:
            start = float(item.get('start'))
            end = start + float(clip_duration) 
            label = item.get('label', f'Clip {i+1}')
            
            clip = video.subclip(start, end)
            name = f"clip_{i}_{label}.mp4"
            clip.write_videofile(
                name, codec="libx264", audio_codec="aac", preset="ultrafast", threads=4, logger=None
            )
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
        col_dub, col_cut = st.columns(2)

        # === الدبلجة ===
        with col_dub:
            render_header("dubbing.png", "🎙️ الدبلجة")
            st.markdown("---")
            all_langs = ["Arabic", "English", "French", "Spanish", "German", "Chinese", "Japanese", "Russian"]
            target_lang = st.selectbox("اللغة المستهدفة", all_langs)
            
            if st.button("🚀 تنفيذ الدبلجة فقط", use_container_width=True):
                st.session_state['dubbed_video'] = None
                with st.status("جاري الدبلجة...", expanded=True) as status:
                    status.write("1. استخراج الصوت...")
                    aud = extract_audio(video_path)
                    
                    status.write("2. تحليل الهوية الصوتية...")
                    gend = detect_speaker_gender(aud)
                    
                    status.write(f"3. الترجمة إلى {target_lang}...")
                    txt = transcribe_and_translate(aud, target_lang)
                    
                    if txt:
                        status.write("4. توليد الصوت (ElevenLabs)...")
                        # اختيار صوت مناسب
                        voice = "21m00Tcm4TlvDq8ikWAM" if gend == "female" else "pNInz6obpgDQGcFmaJgB"
                        dub = generate_dubbed_audio(txt, voice)
                        
                        if dub:
                            status.write("5. دمج الفيديو النهائي...")
                            st.session_state['dubbed_video'] = merge_audio_video(video_path, dub)
                            status.update(label="✅ تمت الدبلجة!", state="complete")
                        else:
                            status.update(label="❌ فشل توليد الصوت", state="error")
                    else:
                        status.update(label="❌ فشلت الترجمة (راجع الخطأ أعلاه)", state="error")

        # === القص ===
        with col_cut:
            render_header("clipping.png", "✂️ القص الذكي")
            st.markdown("---")
            num_clips = st.number_input("عدد المقاطع المطلوبة", 1, 5, 2)
            clip_dur = st.number_input("مدة المقطع (ثانية)", 10, 60, 20)
            
            if st.button("🚀 تنفيذ القص فقط", use_container_width=True):
                st.session_state['generated_clips'] = []
                with st.status("جاري القص...", expanded=True) as status:
                    status.write(f"تحليل واختيار أفضل {num_clips} لقطات...")
                    clips = analyze_and_cut_specific(video_path, num_clips, clip_dur)
                    if clips:
                        st.session_state['generated_clips'] = clips
                        status.update(label="✅ تم القص!", state="complete")
                    else:
                        status.update(label="❌ خطأ في القص", state="error")

        st.divider()
        st.header("النتائج")

        if st.session_state['dubbed_video']:
            st.subheader("🎥 الفيديو المدبلج")
            st.video(st.session_state['dubbed_video'])

        if st.session_state['generated_clips']:
            st.subheader("✂️ المقاطع المستخرجة")
            cols = st.columns(len(st.session_state['generated_clips']))
            for i, clip in enumerate(st.session_state['generated_clips']):
                with cols[i]:
                    st.write(f"**{clip['label']}**")
                    st.video(clip['path'])