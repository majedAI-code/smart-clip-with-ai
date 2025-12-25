import streamlit as st
import os
import json
import time
import socket
import google.generativeai as genai
from elevenlabs.client import ElevenLabs
from moviepy.editor import VideoFileClip, AudioFileClip
import yt_dlp

# --- إعدادات Timeout ---
socket.setdefaulttimeout(600) 

# --- 1. إعدادات الصفحة ---
st.set_page_config(page_title="استوديو القص والترجمة", layout="centered", page_icon="✂️")

# --- 2. تحميل المفاتيح بشكل آمن (من Secrets) ---
def load_api_keys():
    try:
        # الكود هنا يطلب المفاتيح من سيرفر الاستضافة وليس من ملف مكتوب
        GOOGLE_KEY = st.secrets["GOOGLE_API_KEY"]
        ELEVEN_KEY = st.secrets["ELEVENLABS_API_KEY"]
        return GOOGLE_KEY, ELEVEN_KEY
    except Exception as e:
        # رسالة لطيفة تخبرك أين تضع المفاتيح عند الرفع
        st.error("⚠️ لم يتم العثور على المفاتيح! يرجى إضافتها في إعدادات Streamlit Cloud (Secrets).")
        st.stop()

GOOGLE_API_KEY, ELEVENLABS_API_KEY = load_api_keys()

genai.configure(api_key=GOOGLE_API_KEY)
eleven_client = ElevenLabs(api_key=ELEVENLABS_API_KEY)

# --- بقية الكود والدوال (كما هي في النسخة 6.0) ---

def get_best_model():
    try:
        for m in genai.list_models():
            if 'flash' in m.name and 'generateContent' in m.supported_generation_methods:
                return m.name
        return 'models/gemini-1.5-flash'
    except:
        return 'models/gemini-1.5-flash'

CURRENT_MODEL_NAME = get_best_model()

def detect_speaker_gender(audio_path):
    model = genai.GenerativeModel(CURRENT_MODEL_NAME)
    try:
        with open(audio_path, "rb") as f:
            audio_data = f.read()
        prompt = "Listen to this audio. Identify the gender of the MAIN speaker. Return ONLY one word: 'Male' or 'Female'."
        response = model.generate_content([prompt, {"mime_type": "audio/mp3", "data": audio_data}])
        result = response.text.strip().lower()
        if "female" in result: return "female"
        return "male"
    except: return "male"

def download_youtube_video(url):
    output_filename = "downloaded_video.mp4"
    if os.path.exists(output_filename):
        os.remove(output_filename)
    
    # خيارات متقدمة لخداع الحماية
    ydl_opts = {
        'format': 'best[ext=mp4]/best',
        'outtmpl': 'downloaded_video.%(ext)s',
        'quiet': True,
        'no_warnings': True,
        # هذا السطر مهم جداً: يجعل الطلب يبدو كأنه من متصفح كروم
        'http_headers': {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        }
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        return output_filename
    except Exception as e:
        # في حال الفشل، نعيد الخطأ لنراه
        raise e

def extract_audio(video_path):
    video = VideoFileClip(video_path)
    audio_path = "temp_audio.mp3"
    video.audio.write_audiofile(audio_path, logger=None)
    return audio_path

def transcribe_and_translate(audio_path, source_lang, target_lang):
    model = genai.GenerativeModel(CURRENT_MODEL_NAME)
    with open(audio_path, "rb") as f: audio_data = f.read()
    prompt = f"Listen to this audio. Source language: {source_lang}. Transcribe and translate to {target_lang}. Return ONLY the translation."
    result = model.generate_content([prompt, {"mime_type": "audio/mp3", "data": audio_data}])
    return result.text

def generate_dubbed_audio(text, voice_id):
    audio_generator = eleven_client.text_to_speech.convert(text=text, voice_id=voice_id, model_id="eleven_multilingual_v2")
    save_path = "dubbed_audio.mp3"
    with open(save_path, "wb") as f:
        for chunk in audio_generator: f.write(chunk)
    return save_path

def merge_audio_video(video_path, audio_path):
    video = VideoFileClip(video_path)
    new_audio = AudioFileClip(audio_path)
    final_video = video.set_audio(new_audio)
    if new_audio.duration < video.duration: final_video = final_video.subclip(0, new_audio.duration)
    output_path = "final_dubbed_video.mp4"
    final_video.write_videofile(output_path, codec="libx264", audio_codec="aac", logger=None)
    return output_path

def get_viral_clips(video_path, num_clips, clip_duration):
    model = genai.GenerativeModel(CURRENT_MODEL_NAME)
    myfile = genai.upload_file(video_path)
    while myfile.state.name == "PROCESSING":
        time.sleep(2)
        myfile = genai.get_file(myfile.name)
    prompt = f"""
    Analyze video. Identify exactly {num_clips} viral segments.
    Each segment approx {clip_duration} seconds. No overlap.
    Return JSON list: [{{"start": 10, "end": 40, "label": "Clip 1"}}]
    """
    try:
        response = model.generate_content([prompt, myfile])
        text = response.text.replace("```json", "").replace("```", "").strip()
        return json.loads(text)
    except: return []

def cut_clips(original_video, timestamps):
    if isinstance(original_video, str): original_video = VideoFileClip(original_video)
    clips = []
    for item in timestamps:
        start, end = 0, 0
        if isinstance(item, dict):
            start = item.get('start') or item.get('start_time')
            end = item.get('end') or item.get('end_time')
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            start, end = item[0], item[1]
        try:
            start, end = float(start), float(end)
            if start < end and end <= original_video.duration:
                clips.append(original_video.subclip(start, end))
        except: continue
    return clips

# --- واجهة المستخدم ---
if 'video_path' not in st.session_state: st.session_state['video_path'] = None

st.header("1. رفع الفيديو")
input_method = st.radio("الطريقة:", ["رفع ملف", "رابط يوتيوب"], horizontal=True)

if input_method == "رفع ملف":
    uploaded_file = st.file_uploader("ملف MP4", type=["mp4"])
    if uploaded_file:
        with open("uploaded_temp.mp4", "wb") as f: f.write(uploaded_file.getbuffer())
        st.session_state['video_path'] = "uploaded_temp.mp4"
        st.video(st.session_state['video_path'])
elif input_method == "رابط يوتيوب":
    yt_url = st.text_input("الرابط:")
    if yt_url and st.button("تحميل"):
        with st.spinner("جاري التحميل..."):
            try:
                path = download_youtube_video(yt_url)
                st.session_state['video_path'] = path
                st.success("تم!")
            except: st.error("خطأ في الرابط")

if st.session_state['video_path'] and input_method == "رابط يوتيوب": st.video(st.session_state['video_path'])

if st.session_state['video_path']:
    st.divider()
    st.header("2. خيارات المعالجة")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("🎙️ الدبلجة")
        enable_dubbing = st.checkbox("تفعيل")
        if enable_dubbing:
            s_lang = st.selectbox("من", ["English", "Arabic", "Spanish", "French"])
            t_lang = st.selectbox("إلى", ["Arabic", "English", "Spanish", "French"])
            st.caption("اختيار الصوت تلقائي (رجل/امرأة)")
    with c2:
        st.subheader("✂️ القص")
        enable_clipping = st.checkbox("تفعيل القص")
        num = st.number_input("العدد", 1, 10, 3)
        dur = st.slider("المدة", 15, 120, 60)

    if st.button("🚀 ابدأ"):
        vid = st.session_state['video_path']
        if enable_dubbing:
            with st.status("🎙️ جاري الدبلجة...") as s:
                aud = extract_audio(vid)
                gender = detect_speaker_gender(aud)
                # Voice IDs are hardcoded here safely
                voice = "21m00Tcm4TlvDq8ikWAM" if gender == "female" else "pNInz6obpgDQGcFmaJgB"
                st.info(f"تم اكتشاف صوت: {gender} - Voice ID: {voice}")
                txt = transcribe_and_translate(aud, s_lang, t_lang)
                new_aud = generate_dubbed_audio(txt, voice)
                final = merge_audio_video(vid, new_aud)
                st.video(final)
                s.update(label="✅ تم!", state="complete")
        if enable_clipping:
            with st.status("✂️ جاري القص...") as s:
                st.write("تحليل Gemini...")
                times = get_viral_clips(vid, num, dur)
                if times:
                    clips = cut_clips(vid, times)
                    s.update(label="✅ تم!", state="complete")
                    for i, c in enumerate(clips):
                        name = f"clip_{i}.mp4"
                        c.write_videofile(name, codec="libx264", audio_codec="aac", verbose=False, logger=None)
                        with open(name, "rb") as f: st.download_button(f"تحميل {i+1}", f, file_name=name)
                else: st.error("فشل التحليل")