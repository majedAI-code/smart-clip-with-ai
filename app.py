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

# --- 3. اختيار الموديل تلقائياً ---
@st.cache_resource
def get_working_model_name():
    try:
        models = list(genai.list_models())
        generation_models = [m.name for m in models if 'generateContent' in m.supported_generation_methods]
        
        for m in generation_models:
            if 'gemini-1.5-flash' in m: return m
        for m in generation_models:
            if 'gemini-1.5-pro' in m: return m
        if generation_models:
            return generation_models[0]
    except: pass
    return "models/gemini-1.5-flash"

CURRENT_MODEL_NAME = get_working_model_name()
# st.sidebar.success(f"الموديل: {CURRENT_MODEL_NAME}") # (اختياري للعرض)

# --- 4. إدارة الحالة (Session State) ---
if 'dubbed_video' not in st.session_state: st.session_state['dubbed_video'] = None
if 'generated_clips' not in st.session_state: st.session_state['generated_clips'] = []
if 'dubbed_clips_results' not in st.session_state: st.session_state['dubbed_clips_results'] = []

# --- 5. دوال المعالجة ---

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
    video.audio.write_audiofile(
        audio_path, 
        bitrate="64k",
        fps=22050,          
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
        st.error(f"❌ خطأ الترجمة: {e}")
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
        error_msg = str(e)
        if "payment_issue" in error_msg:
            st.warning("⚠️ تنبيه: مشكلة في الدفع (Payment Issue) في حساب ElevenLabs.")
        elif "quota_exceeded" in error_msg:
            st.warning("⚠️ تنبيه: انتهى رصيد الحروف في ElevenLabs.")
        else:
            st.error(f"خطأ في الصوت: {e}")
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

# --- دالة القص ---
def analyze_and_cut_specific(video_path, num_clips, clip_duration, prefix="clip"):
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
        Timestamps must be strictly within 0 and {total_duration}.
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
            # التأكد من عدم تجاوز مدة الفيديو
            if end > video.duration: end = video.duration
            if start >= end: continue

            label = item.get('label', f'{prefix} {i+1}')
            
            clip = video.subclip(start, end)
            name = f"{prefix}_{i}_{label}.mp4".replace(" ", "_")
            clip.write_videofile(
                name, codec="libx264", audio_codec="aac", preset="ultrafast", threads=4, logger=None
            )
            generated_files.append({"path": name, "label": label})
        except: continue
        
    video.close()
    return generated_files

# --- الواجهة (UI) ---
render_header("banner.jpg", "استوديو المحتوى الذكي")
st.caption("أتمتة صناعة المحتوى الإعلامي")

st.markdown("### 1. رفع الفيديو")
# التعديل الأول: تغيير اسم الخيار
upload_option = st.radio("المصدر:", ["رفع ملف", "رابط يوتيوب (فيديو Demo)"], horizontal=True)
video_path = None

if upload_option == "رفع ملف":
    uploaded_file = st.file_uploader("ملف MP4", type=["mp4"])
    if uploaded_file:
        with open("temp_video.mp4", "wb") as f: f.write(uploaded_file.getbuffer())
        video_path = "temp_video.mp4"

elif upload_option == "رابط يوتيوب (فيديو Demo)":
    # إضافة خانة شكلية للرابط لتبدو وكأنها يوتيوب حقيقي
    yt_url = st.text_input("أدخل رابط الفيديو:", placeholder="https://www.youtube.com/watch?v=...")
    
    # زر التحميل يقوم بتحميل الفيديو التجريبي المضمن لضمان نجاح العرض
    if st.button("تحميل الفيديو") and os.path.exists("sample.mp4"):
        video_path = "sample.mp4"
        st.success("تم جلب الفيديو بنجاح! (Demo Mode)")
    elif not os.path.exists("sample.mp4"):
        st.error("ملف sample.mp4 غير موجود في المجلد.")

if video_path:
    valid, dur = check_video_duration(video_path, 5)
    if not valid:
        st.error(f"الفيديو طويل ({int(dur/60)} دقيقة).")
    else:
        st.video(video_path)
        st.divider()
        col_dub, col_cut = st.columns(2)

        # === العمود 1: الدبلجة ===
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
                    
                    status.write("2. تحليل الهوية والترجمة...")
                    txt = transcribe_and_translate(aud, target_lang)
                    gend = detect_speaker_gender(aud)
                    
                    if txt:
                        status.write("3. توليد الصوت (ElevenLabs)...")
                        voice = "21m00Tcm4TlvDq8ikWAM" if gend == "female" else "pNInz6obpgDQGcFmaJgB"
                        dub = generate_dubbed_audio(txt, voice)
                        
                        if dub:
                            status.write("4. دمج الفيديو النهائي...")
                            st.session_state['dubbed_video'] = merge_audio_video(video_path, dub)
                            status.update(label="✅ تمت الدبلجة!", state="complete")
                        else:
                            status.update(label="❌ فشل توليد الصوت", state="error")
                    else:
                        status.update(label="❌ فشلت الترجمة", state="error")

        # === العمود 2: القص (للفيديو الأصلي المرفوع) ===
        with col_cut:
            render_header("clipping.png", "✂️ القص الذكي (للأصل)")
            st.markdown("---")
            num_clips = st.number_input("عدد المقاطع", 1, 5, 2, key="orig_num")
            clip_dur = st.number_input("المدة (ثانية)", 10, 60, 20, key="orig_dur")
            
            if st.button("🚀 قص الفيديو الأصلي", use_container_width=True):
                st.session_state['generated_clips'] = []
                with st.status("جاري القص...", expanded=True) as status:
                    status.write(f"تحليل واختيار أفضل {num_clips} لقطات من الأصل...")
                    clips = analyze_and_cut_specific(video_path, num_clips, clip_dur, prefix="orig_clip")
                    if clips:
                        st.session_state['generated_clips'] = clips
                        status.update(label="✅ تم القص!", state="complete")
                    else:
                        status.update(label="❌ خطأ في القص", state="error")

        st.divider()
        st.header("النتائج")

        # 1. نتائج الدبلجة + خيار قص المدبلج (التعديل الثالث)
        if st.session_state['dubbed_video']:
            st.subheader("🎥 الفيديو المدبلج")
            st.video(st.session_state['dubbed_video'])
            with open(st.session_state['dubbed_video'], "rb") as f:
                st.download_button("تحميل المدبلج", f, file_name="dubbed.mp4")
            
            # --- قسم جديد: قص الفيديو المدبلج ---
            st.markdown("---")
            st.markdown("#### ✂️ استخراج مقاطع من هذا الفيديو المدبلج")
            c_d1, c_d2, c_d3 = st.columns([1, 1, 1])
            with c_d1:
                d_num = st.number_input("عدد المقاطع", 1, 5, 2, key="dub_num")
            with c_d2:
                d_dur = st.number_input("المدة (ث)", 10, 60, 20, key="dub_dur")
            with c_d3:
                st.write("") # spacer
                st.write("")
                if st.button("قص المدبلج الآن", type="primary"):
                    st.session_state['dubbed_clips_results'] = []
                    with st.spinner("جاري تحليل وقص الفيديو المدبلج..."):
                        # نرسل الفيديو المدبلج للتحليل والقص
                        d_clips = analyze_and_cut_specific(
                            st.session_state['dubbed_video'], 
                            d_num, 
                            d_dur, 
                            prefix="dub_clip"
                        )
                        if d_clips:
                            st.session_state['dubbed_clips_results'] = d_clips
                            st.success("تم قص الفيديو المدبلج!")

            # عرض مقاطع المدبلج إن وجدت
            if st.session_state['dubbed_clips_results']:
                st.write("**المقاطع المدبلجة:**")
                d_cols = st.columns(len(st.session_state['dubbed_clips_results']))
                for i, clip in enumerate(st.session_state['dubbed_clips_results']):
                    with d_cols[i]:
                        st.caption(f"{clip['label']}")
                        st.video(clip['path'])
                        with open(clip['path'], "rb") as f:
                            st.download_button("تحميل", f, file_name=clip['path'], key=f"dl_dub_{i}")

        st.divider()

        # 2. نتائج قص الفيديو الأصلي
        if st.session_state['generated_clips']:
            st.subheader("✂️ مقاطع من الفيديو الأصلي (بلغته الأم)")
            cols = st.columns(len(st.session_state['generated_clips']))
            for i, clip in enumerate(st.session_state['generated_clips']):
                with cols[i]:
                    st.write(f"**{clip['label']}**")
                    st.video(clip['path'])
                    with open(clip['path'], "rb") as f:
                        st.download_button("تحميل", f, file_name=clip['path'], key=f"dl_orig_{i}")