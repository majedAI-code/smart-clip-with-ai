import streamlit as st
import os
import json
import time
import socket
import google.generativeai as genai
from elevenlabs.client import ElevenLabs
from moviepy.editor import VideoFileClip, AudioFileClip
import yt_dlp

# --- إعدادات الصفحة والهوية البصرية ---
st.set_page_config(
    page_title="استوديو المحتوى الذكي",
    page_icon="🎬",
    layout="centered"
)

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

# --- إدارة الذاكرة (Session State) لحل مشكلة اختفاء التحميل ---
if 'processed' not in st.session_state:
    st.session_state['processed'] = False
if 'dubbed_video' not in st.session_state:
    st.session_state['dubbed_video'] = None
if 'clips_list' not in st.session_state:
    st.session_state['clips_list'] = []

# --- دوال المعالجة ---

def check_video_duration(video_path, max_minutes=5):
    try:
        clip = VideoFileClip(video_path)
        duration_sec = clip.duration
        clip.close()
        if duration_sec > (max_minutes * 60):
            return False, duration_sec
        return True, duration_sec
    except:
        return True, 0

def get_best_model():
    return 'models/gemini-1.5-flash'

CURRENT_MODEL_NAME = get_best_model()

def detect_speaker_gender(audio_path):
    model = genai.GenerativeModel(CURRENT_MODEL_NAME)
    try:
        with open(audio_path, "rb") as f:
            audio_data = f.read()
        prompt = "Listen to this audio. Identify the gender of the MAIN speaker. Return ONLY one word: 'Male' or 'Female'."
        response = model.generate_content([prompt, {"mime_type": "audio/mp3", "data": audio_data}])
        if "female" in response.text.strip().lower(): return "female"
        return "male"
    except: return "male"

def extract_audio(video_path):
    video = VideoFileClip(video_path)
    audio_path = "temp_audio.mp3"
    video.audio.write_audiofile(audio_path, logger=None)
    video.close()
    return audio_path

def transcribe_and_translate(audio_path, target_lang):
    model = genai.GenerativeModel(CURRENT_MODEL_NAME)
    with open(audio_path, "rb") as f: audio_data = f.read()
    prompt = f"Listen to this audio. Transcribe and translate the content to {target_lang}. Return ONLY the translated text suitable for dubbing."
    result = model.generate_content([prompt, {"mime_type": "audio/mp3", "data": audio_data}])
    return result.text

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
        st.warning(f"⚠️ تجاوز الدبلجة بسبب: {e}")
        return None

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

# --- الدالة الجديدة: القص الذكي الديناميكي ---
def get_viral_clips_dynamic(video_path):
    model = genai.GenerativeModel(CURRENT_MODEL_NAME)
    myfile = genai.upload_file(video_path)
    while myfile.state.name == "PROCESSING":
        time.sleep(2)
        myfile = genai.get_file(myfile.name)
    
    # هنا الذكاء: لا نطلب عدداً محدداً، نطلب منه اختيار الأفضل فقط
    prompt = """
    Analyze the video carefully. Identify the MOST viral and engaging segments (Shorts/Reels).
    - Select only segments that stand out (funny, insightful, shocking, or summarized).
    - Duration of each clip: between 15 to 60 seconds.
    - Return a JSON list: [{"start": 10, "end": 40, "label": "Topic 1"}, ...]
    - If the video is boring, return an empty list.
    """
    try:
        response = model.generate_content([prompt, myfile])
        text = response.text.replace("```json", "").replace("```", "").strip()
        return json.loads(text)
    except: return []

def cut_clips_processing(original_video_path, timestamps):
    video = VideoFileClip(original_video_path)
    generated_files = []
    for i, item in enumerate(timestamps):
        start = item.get('start')
        end = item.get('end')
        label = item.get('label', f'Clip {i+1}')
        try:
            clip = video.subclip(start, end)
            # تحويل للأبعاد الرأسية للموبايل (اختياري، هنا نقص فقط)
            output_name = f"clip_{i+1}_{label}.mp4"
            clip.write_videofile(output_name, codec="libx264", audio_codec="aac", logger=None)
            generated_files.append({"path": output_name, "label": label})
        except: continue
    video.close()
    return generated_files

# --- واجهة التطبيق ---

st.title("استوديو المحتوى الذكي")
st.caption("أتمتة صناعة المحتوى الإعلامي باستخدام الذكاء الاصطناعي التوليدي")

# 1. قسم الرفع
st.header("1. رفع الفيديو")
upload_option = st.radio("الطريقة:", ["رفع ملف", "فيديو تجريبي (Demo)"], horizontal=True)
video_path = None

if upload_option == "رفع ملف":
    uploaded_file = st.file_uploader("اختر ملف MP4", type=["mp4"])
    if uploaded_file:
        with open("temp_video.mp4", "wb") as f: f.write(uploaded_file.getbuffer())
        video_path = "temp_video.mp4"
elif upload_option == "فيديو تجريبي (Demo)":
    if os.path.exists("sample.mp4"):
        video_path = "sample.mp4"
        st.success("✅ تم تحميل الفيديو التجريبي")
    else:
        st.error("لم يتم العثور على sample.mp4")

if video_path:
    # فحص المدة
    valid, dur = check_video_duration(video_path, 5)
    if not valid:
        st.error(f"فيديو طويل جداً ({int(dur/60)} دقيقة). الحد الأقصى 5 دقائق.")
        st.stop()
    
    st.video(video_path)
    
    st.divider()
    st.header("2. خيارات المعالجة")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("الدبلجة الذكية")
        st.caption("Smart Dubbing")
        enable_dubbing = st.checkbox("تفعيل الدبلجة")
        target_lang = st.selectbox("اللغة المستهدفة", ["English", "Arabic", "French", "Spanish", "Chinese"])
        
    with col2:
        st.subheader("القص الذكي")
        st.caption("Viral Clipping")
        enable_clipping = st.checkbox("استخراج المقاطع الأكثر رواجاً")
        st.info("🤖 سيقوم الذكاء الاصطناعي بتحديد عدد ومدّة المقاطع تلقائياً حسب المحتوى.")

    # زر التشغيل
    if st.button("🚀 بدء التحليل والمعالجة"):
        st.session_state['processed'] = True # حفظ حالة أننا بدأنا
        st.session_state['dubbed_video'] = None
        st.session_state['clips_list'] = [] # تصفير القائمة القديمة
        
        with st.status("جاري العمل في الاستوديو...", expanded=True) as status:
            
            # --- معالجة الدبلجة ---
            if enable_dubbing:
                status.write("🎙️ تحليل الصوت والهوية...")
                audio_path = extract_audio(video_path)
                gender = detect_speaker_gender(audio_path)
                
                status.write(f"detected gender: {gender}. Translating...")
                translated_text = transcribe_and_translate(audio_path, target_lang)
                
                # اختيار الصوت بناءً على الجنس
                voice_id = "21m00Tcm4TlvDq8ikWAM" if gender == "female" else "pNInz6obpgDQGcFmaJgB"
                
                status.write("🗣️ توليد الدبلجة...")
                dubbed_audio = generate_dubbed_audio(translated_text, voice_id)
                
                if dubbed_audio:
                    final_video = merge_audio_video(video_path, dubbed_audio)
                    st.session_state['dubbed_video'] = final_video
                else:
                    st.warning("تم تخطي الدبلجة لسبب تقني، سيتم عرض الفيديو الأصلي.")

            # --- معالجة القص الذكي ---
            if enable_clipping:
                status.write("🧠 Gemini يشاهد الفيديو ويحدد اللقطات الفيروسية...")
                # هنا لا نحدد العدد، نترك Gemini يقرر
                clips_data = get_viral_clips_dynamic(video_path)
                
                if clips_data:
                    status.write(f"✂️ وجدنا {len(clips_data)} مقاطع مميزة. جاري القص...")
                    generated_clips = cut_clips_processing(video_path, clips_data)
                    st.session_state['clips_list'] = generated_clips
                else:
                    st.warning("لم يجد الذكاء الاصطناعي مقاطع فيروسية واضحة في هذا الفيديو.")
            
            status.update(label="✅ تمت المهمة!", state="complete")

# --- عرض النتائج (مفصول عن الزر لضمان الثبات) ---
if st.session_state['processed']:
    st.divider()
    st.header("3. النتائج")
    
    # 1. عرض الدبلجة
    if st.session_state['dubbed_video']:
        st.subheader("🎥 الفيديو المدبلج")
        st.video(st.session_state['dubbed_video'])
        with open(st.session_state['dubbed_video'], "rb") as f:
            st.download_button("تحميل الفيديو المدبلج", f, file_name="dubbed_video.mp4")
    
    # 2. عرض المقاطع المقصوصة
    if st.session_state['clips_list']:
        st.subheader("🔥 المقاطع الأكثر رواجاً (Viral Clips)")
        cols = st.columns(len(st.session_state['clips_list'])) if len(st.session_state['clips_list']) > 0 else [st.container()]
        
        for i, clip in enumerate(st.session_state['clips_list']):
            # عرض المقاطع بشكل جميل
            st.write(f"**📌 {clip['label']}**")
            st.video(clip['path'])
            with open(clip['path'], "rb") as f:
                # المفتاح (key) هنا هو السر لعدم اختفاء الأزرار
                st.download_button(
                    label=f"تحميل المقطع {i+1}",
                    data=f,
                    file_name=clip['path'],
                    key=f"btn_{i}" 
                )
            st.divider()