import streamlit as st
import os
import json
import time
import requests  # مكتبة الاتصال المباشر
import google.generativeai as genai
from elevenlabs.client import ElevenLabs
from moviepy.editor import VideoFileClip

# --- 1. إعدادات الصفحة ---
st.set_page_config(
    page_title="استوديو المحتوى الذكي",
    page_icon="🎬",
    layout="centered",
    initial_sidebar_state="collapsed"
)

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

# --- 3. موديل Gemini ---
@st.cache_resource
def get_working_model_name():
    try:
        models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        for m in models: 
            if 'gemini-1.5-flash' in m: return m
        if models: return models[0]
    except: pass
    return "models/gemini-1.5-flash"

CURRENT_MODEL_NAME = get_working_model_name()

# --- 4. إدارة الحالة ---
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

def render_header(image_name, alt_text):
    if os.path.exists(image_name):
        st.image(image_name, use_column_width=True)
    else:
        st.header(alt_text)

# === (الحل الجذري) دبلجة بالاتصال المباشر API Request ===
def process_full_dubbing(video_path, target_lang_code):
    try:
        # رابط خدمة الدبلجة المباشر
        url = "https://api.elevenlabs.io/v1/dubbing"
        
        headers = {
            "xi-api-key": ELEVENLABS_API_KEY
        }
        
        # 1. إرسال ملف الفيديو (POST)
        with open(video_path, "rb") as f:
            data = {
                "target_lang": target_lang_code,
                "mode": "automatic",
                "source_lang": "auto",
                "num_speakers": "0",
                "watermark": "false"
            }
            files = {"file": f}
            
            response = requests.post(url, headers=headers, data=data, files=files)
        
        if response.status_code != 200:
            st.error(f"خطأ في الاتصال بالسيرفر: {response.text}")
            return None
            
        dubbing_id = response.json().get("dubbing_id")
        
        # 2. انتظار المعالجة (Polling)
        progress_text = "جاري الدبلجة والمزامنة (Pro)..."
        my_bar = st.progress(0, text=progress_text)
        
        while True:
            # التحقق من الحالة
            status_url = f"https://api.elevenlabs.io/v1/dubbing/{dubbing_id}"
            status_response = requests.get(status_url, headers=headers)
            status_data = status_response.json()
            
            status = status_data.get("status")
            
            if status == "dubbed":
                my_bar.progress(100, text="تمت الدبلجة بنجاح!")
                break
            elif status == "failed":
                st.error("فشلت عملية الدبلجة في المصدر.")
                return None
            else:
                time.sleep(2) # انتظار
        
        # 3. تحميل الفيديو الجاهز
        # ملاحظة: لتحميل الفيديو نستخدم رابط خاص
        download_url = f"https://api.elevenlabs.io/v1/dubbing/{dubbing_id}/audio/{target_lang_code}"
        
        dl_response = requests.get(download_url, headers=headers, stream=True)
        
        output_path = "final_dubbed_video.mp4"
        
        if dl_response.status_code == 200:
            with open(output_path, "wb") as f:
                for chunk in dl_response.iter_content(chunk_size=1024):
                    f.write(chunk)
            return output_path
        else:
            st.error(f"فشل تحميل الملف النهائي: {dl_response.text}")
            return None

    except Exception as e:
        st.error(f"حدث خطأ غير متوقع: {e}")
        return None

# === دالة القص الذكي ===
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
        Timestamps must be within 0 and {total_duration}.
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
            if end > video.duration: end = video.duration
            if start >= end: continue
            label = item.get('label', f'{prefix} {i+1}')
            clip = video.subclip(start, end)
            name = f"{prefix}_{i}_{label}.mp4".replace(" ", "_")
            clip.write_videofile(name, codec="libx264", audio_codec="aac", preset="ultrafast", threads=4, logger=None)
            generated_files.append({"path": name, "label": label})
        except: continue
    video.close()
    return generated_files

# --- 6. الواجهة (UI) ---
render_header("banner.jpg", "استوديو المحتوى الذكي")
st.caption("أتمتة صناعة المحتوى الإعلامي (Pro Edition)")

st.markdown("### 1. رفع الفيديو")
upload_option = st.radio("المصدر:", ["رفع ملف", "رابط يوتيوب (فيديو Demo)"], horizontal=True)
video_path = None

if upload_option == "رفع ملف":
    uploaded_file = st.file_uploader("ملف MP4", type=["mp4"])
    if uploaded_file:
        with open("temp_video.mp4", "wb") as f: f.write(uploaded_file.getbuffer())
        video_path = "temp_video.mp4"
elif upload_option == "رابط يوتيوب (فيديو Demo)":
    yt_url = st.text_input("أدخل رابط الفيديو:", placeholder="https://www.youtube.com/watch?v=...")
    if st.button("تحميل الفيديو") and os.path.exists("sample.mp4"):
        video_path = "sample.mp4"
        st.success("تم جلب الفيديو بنجاح! (Demo Mode)")

if video_path:
    valid, dur = check_video_duration(video_path, 5)
    if not valid:
        st.error(f"الفيديو طويل ({int(dur/60)} دقيقة).")
    else:
        st.video(video_path)
        st.divider()
        col_dub, col_cut = st.columns(2)

        with col_dub:
            render_header("dubbing.png", "🎙️ الدبلجة")
            st.markdown("---")
            lang_map = {"Arabic": "ar", "English": "en", "French": "fr", "Spanish": "es", "German": "de", "Chinese": "zh"}
            target_lang_name = st.selectbox("اللغة المستهدفة", list(lang_map.keys()))
            
            if st.button("🚀 تنفيذ الدبلجة (Pro)", use_container_width=True):
                st.session_state['dubbed_video'] = None
                with st.status("جاري الاتصال باستوديو الدبلجة...", expanded=True) as status:
                    status.write("1. رفع الفيديو وتحليل المتحدثين...")
                    final_vid = process_full_dubbing(video_path, lang_map[target_lang_name])
                    if final_vid:
                        st.session_state['dubbed_video'] = final_vid
                        status.update(label="✅ تمت الدبلجة والمزامنة!", state="complete")
                    else:
                        status.update(label="❌ فشلت العملية", state="error")

        with col_cut:
            render_header("clipping.png", "✂️ القص الذكي (للأصل)")
            st.markdown("---")
            num_clips = st.number_input("العدد", 1, 5, 2, key="on")
            clip_dur = st.number_input("المدة", 10, 60, 20, key="od")
            if st.button("🚀 قص الفيديو الأصلي", use_container_width=True):
                st.session_state['generated_clips'] = []
                with st.status("جاري القص...", expanded=True) as status:
                    clips = analyze_and_cut_specific(video_path, num_clips, clip_dur, prefix="orig")
                    if clips:
                        st.session_state['generated_clips'] = clips
                        status.update(label="✅ تم القص!", state="complete")
                    else:
                        status.update(label="❌ خطأ", state="error")

        st.divider()
        st.header("النتائج")

        if st.session_state['dubbed_video']:
            st.subheader("🎥 الفيديو المدبلج")
            st.video(st.session_state['dubbed_video'])
            with open(st.session_state['dubbed_video'], "rb") as f:
                st.download_button("تحميل المدبلج", f, "dubbed_pro.mp4")
            
            st.markdown("---")
            st.write("✂️ **قص المدبلج:**")
            c1, c2, c3 = st.columns([1,1,1])
            with c1: dn = st.number_input("العدد", 1, 5, 2, key="dn")
            with c2: dd = st.number_input("المدة", 10, 60, 20, key="dd")
            with c3:
                st.write("")
                st.write("")
                if st.button("قص الآن"):
                    st.session_state['dubbed_clips_results'] = analyze_and_cut_specific(st.session_state['dubbed_video'], dn, dd, prefix="dub")
            
            if st.session_state['dubbed_clips_results']:
                dc = st.columns(len(st.session_state['dubbed_clips_results']))
                for i, clip in enumerate(st.session_state['dubbed_clips_results']):
                    with dc[i]:
                        st.video(clip['path'])
                        with open(clip['path'], "rb") as f: st.download_button("📥", f, clip['path'], key=f"d{i}")

        if st.session_state['generated_clips']:
            st.subheader("✂️ مقاطع الأصل")
            oc = st.columns(len(st.session_state['generated_clips']))
            for i, clip in enumerate(st.session_state['generated_clips']):
                with oc[i]:
                    st.video(clip['path'])
                    with open(clip['path'], "rb") as f: st.download_button("📥", f, clip['path'], key=f"o{i}")