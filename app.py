import streamlit as st
import os
import json
import time
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
        st.error("⚠️ يرجى وضع المفاتيح في ملف secrets.toml")
        st.stop()

GOOGLE_API_KEY, ELEVENLABS_API_KEY = load_api_keys()
genai.configure(api_key=GOOGLE_API_KEY)
eleven_client = ElevenLabs(api_key=ELEVENLABS_API_KEY)

# --- 3. إدارة الحالة (لمنع اختفاء النتائج عند التحميل) ---
if 'dubbed_video' not in st.session_state: st.session_state['dubbed_video'] = None
if 'generated_clips' not in st.session_state: st.session_state['generated_clips'] = []
if 'dubbed_clips_results' not in st.session_state: st.session_state['dubbed_clips_results'] = []
if 'processing_dub' not in st.session_state: st.session_state['processing_dub'] = False

# --- 4. دوال المعالجة الجديدة (الاحترافية) ---

def check_video_duration(video_path, max_minutes=5):
    try:
        clip = VideoFileClip(video_path)
        dur = clip.duration
        clip.close()
        if dur > max_minutes * 60: return False, dur
        return True, dur
    except: return True, 0

# === السلاح السري: دبلجة احترافية عبر ElevenLabs Dubbing API ===
def process_full_dubbing(video_path, target_lang_code):
    try:
        # 1. رفع الفيديو لاستوديو الدبلجة
        with open(video_path, "rb") as f:
            response = eleven_client.dubbing.dub_a_video_or_an_audio_file(
                file=f,
                target_lang=target_lang_code,
                mode="automatic", # هذا الوضع يضبط التزامن والأصوات تلقائياً
                source_lang="auto", # اكتشاف تلقائي للغة الأصلية
                num_speakers=0, # 0 تعني اكتشاف تلقائي لعدد المتحدثين
                watermark=False
            )
        
        dubbing_id = response.dubbing_id
        
        # 2. انتظار المعالجة (Polling)
        # هذه الخطوة تتم في سحابة ElevenLabs القوية
        progress_text = "جاري الدبلجة والمزامنة (قد يستغرق وقتاً حسب طول الفيديو)..."
        my_bar = st.progress(0, text=progress_text)
        
        while True:
            status = eleven_client.dubbing.get_dubbing_project_metadata(dubbing_id).status
            if status == "dubbed":
                my_bar.progress(100, text="تمت الدبلجة بنجاح!")
                break
            elif status == "failed":
                st.error("فشلت عملية الدبلجة من المصدر.")
                return None
            else:
                # تحديث وهمي للانتظار
                time.sleep(2)
        
        # 3. تحميل الفيديو الجاهز
        output_path = "final_dubbed_video.mp4"
        # ملاحظة: اللغة هنا يجب أن تكون كود (ar, en, etc)
        video_bytes = eleven_client.dubbing.get_dubbed_file(dubbing_id, target_lang_code)
        
        with open(output_path, "wb") as f:
            for chunk in video_bytes:
                f.write(chunk)
                
        return output_path

    except Exception as e:
        st.error(f"خطأ في خدمة الدبلجة: {e}")
        return None

# === دالة القص الذكي (Gemini) ===
def analyze_and_cut_specific(video_path, num_clips, clip_duration, prefix="clip"):
    # نستخدم فلاش لأنه أسرع وأرخص للتحليل
    model = genai.GenerativeModel("models/gemini-1.5-flash")
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
            if end > video.duration: end = video.duration
            if start >= end: continue

            label = item.get('label', f'{prefix} {i+1}')
            
            # قص سريع جداً
            clip = video.subclip(start, end)
            name = f"{prefix}_{i}_{label}.mp4".replace(" ", "_")
            clip.write_videofile(
                name, codec="libx264", audio_codec="aac", preset="ultrafast", threads=4, logger=None
            )
            generated_files.append({"path": name, "label": label})
        except: continue
        
    video.close()
    return generated_files

def render_header(image_name, alt_text):
    if os.path.exists(image_name):
        st.image(image_name, use_column_width=True)
    else:
        st.header(alt_text)

# --- 5. الواجهة (UI) ---
render_header("banner.jpg", "استوديو المحتوى الذكي")
st.caption("أتمتة صناعة المحتوى الإعلامي (Pro Edition)")

# قسم الرفع
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
    elif not os.path.exists("sample.mp4"):
        st.warning("ملف sample.mp4 غير موجود.")

if video_path:
    valid, dur = check_video_duration(video_path, 5)
    if not valid:
        st.error(f"الفيديو طويل ({int(dur/60)} دقيقة).")
    else:
        st.video(video_path)
        st.divider()
        col_dub, col_cut = st.columns(2)

        # === الدبلجة الاحترافية ===
        with col_dub:
            render_header("dubbing.png", "🎙️ الدبلجة")
            st.markdown("---")
            # خريطة لغات ElevenLabs
            lang_map = {
                "Arabic": "ar", "English": "en", "French": "fr", "Spanish": "es", 
                "German": "de", "Chinese": "zh", "Japanese": "ja", "Russian": "ru"
            }
            target_lang_name = st.selectbox("اللغة المستهدفة", list(lang_map.keys()))
            target_code = lang_map[target_lang_name]
            
            if st.button("🚀 تنفيذ الدبلجة (Pro)", use_container_width=True):
                st.session_state['dubbed_video'] = None
                # نستخدم status container لمنع التداخل
                with st.status("جاري الاتصال باستوديو الدبلجة...", expanded=True) as status:
                    status.write("1. رفع الفيديو وتحليل المتحدثين...")
                    # استدعاء الدالة الجديدة
                    final_vid = process_full_dubbing(video_path, target_code)
                    
                    if final_vid:
                        st.session_state['dubbed_video'] = final_vid
                        status.update(label="✅ تمت الدبلجة والمزامنة!", state="complete")
                    else:
                        status.update(label="❌ فشلت العملية", state="error")

        # === القص ===
        with col_cut:
            render_header("clipping.png", "✂️ القص الذكي (للأصل)")
            st.markdown("---")
            num_clips = st.number_input("عدد المقاطع", 1, 5, 2, key="orig_num")
            clip_dur = st.number_input("المدة (ثانية)", 10, 60, 20, key="orig_dur")
            
            if st.button("🚀 قص الفيديو الأصلي", use_container_width=True):
                st.session_state['generated_clips'] = []
                with st.status("جاري القص...", expanded=True) as status:
                    clips = analyze_and_cut_specific(video_path, num_clips, clip_dur, prefix="orig_clip")
                    if clips:
                        st.session_state['generated_clips'] = clips
                        status.update(label="✅ تم القص!", state="complete")
                    else:
                        status.update(label="❌ خطأ في القص", state="error")

        st.divider()
        st.header("النتائج")

        # 1. عرض الدبلجة (مع حفظ الحالة عند التحميل)
        if st.session_state['dubbed_video']:
            st.subheader("🎥 الفيديو المدبلج (متزامن + متعدد الأصوات)")
            st.video(st.session_state['dubbed_video'])
            
            # زر التحميل (مفتاح فريد لمنع التعارض)
            with open(st.session_state['dubbed_video'], "rb") as f:
                st.download_button(
                    label="تحميل الفيديو المدبلج",
                    data=f,
                    file_name="dubbed_pro.mp4",
                    mime="video/mp4",
                    key="dl_main_dub"
                )
            
            # --- قص الفيديو المدبلج ---
            st.markdown("---")
            st.markdown("#### ✂️ قص الفيديو المدبلج")
            c1, c2, c3 = st.columns([1,1,1])
            with c1: d_num = st.number_input("العدد", 1, 5, 2, key="d_n")
            with c2: d_dur = st.number_input("المدة", 10, 60, 20, key="d_d")
            with c3:
                st.write("")
                st.write("")
                if st.button("قص المدبلج الآن"):
                    with st.spinner("جاري قص النسخة المدبلجة..."):
                        d_clips = analyze_and_cut_specific(
                            st.session_state['dubbed_video'], d_num, d_dur, prefix="dub_clip"
                        )
                        if d_clips:
                            st.session_state['dubbed_clips_results'] = d_clips
            
            # عرض مقاطع المدبلج
            if st.session_state['dubbed_clips_results']:
                st.write("**المقاطع المدبلجة:**")
                dc_cols = st.columns(len(st.session_state['dubbed_clips_results']))
                for i, clip in enumerate(st.session_state['dubbed_clips_results']):
                    with dc_cols[i]:
                        st.caption(clip['label'])
                        st.video(clip['path'])
                        with open(clip['path'], "rb") as f:
                            st.download_button("📥", f, file_name=clip['path'], key=f"dl_dclip_{i}")

        st.divider()

        # 2. عرض قص الأصل
        if st.session_state['generated_clips']:
            st.subheader("✂️ مقاطع من الفيديو الأصلي")
            oc_cols = st.columns(len(st.session_state['generated_clips']))
            for i, clip in enumerate(st.session_state['generated_clips']):
                with oc_cols[i]:
                    st.caption(clip['label'])
                    st.video(clip['path'])
                    with open(clip['path'], "rb") as f:
                        st.download_button("📥", f, file_name=clip['path'], key=f"dl_oclip_{i}")