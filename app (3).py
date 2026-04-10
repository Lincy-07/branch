import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import os
import tempfile
import time
from collections import deque
from mediapipe.python.solutions import pose as mp_pose
from mediapipe.python.solutions import drawing_utils as mp_drawing

# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────
st.set_page_config(page_title="AI Fitness Trainer", layout="wide")
st.title("🏋️‍♂️ AI Virtual Fitness Trainer")

actions = ['barbell biceps curl', 'push-up', 'shoulder press', 'squat']

THRESHOLDS = {
    'squat':               {'down': 85,  'up': 155, 'joints': (23, 25, 27)},
    'barbell biceps curl': {'down': 45,  'up': 145, 'joints': (11, 13, 15)},
    'push-up':             {'down': 65,  'up': 145, 'joints': (11, 13, 15)},
    'shoulder press':      {'down': 65,  'up': 145, 'joints': (13, 11, 23)},
}

FORM_RULES = {
    'squat': {
        'good_range': (70, 95),
        'good_msg':  '✅ Great squat depth! Keep chest up.',
        'high_msg':  '⚠️ Go deeper — thighs should reach parallel.',
        'low_msg':   '⚠️ Too deep — stop at parallel to protect knees.',
        'tip':       '💡 Keep knees over toes, weight in heels.',
    },
    'barbell biceps curl': {
        'good_range': (35, 55),
        'good_msg':  '✅ Full range curl! Great squeeze.',
        'high_msg':  '⚠️ Curl higher — squeeze at the top.',
        'low_msg':   '⚠️ Lower fully to complete the rep.',
        'tip':       '💡 Keep elbows pinned to your sides.',
    },
    'push-up': {
        'good_range': (55, 80),
        'good_msg':  '✅ Good push-up depth!',
        'high_msg':  '⚠️ Lower chest closer to the ground.',
        'low_msg':   "⚠️ Don't collapse — keep core tight.",
        'tip':       '💡 Body must be a straight line.',
    },
    'shoulder press': {
        'good_range': (55, 80),
        'good_msg':  '✅ Great press! Full extension.',
        'high_msg':  '⚠️ Press fully overhead — extend arms.',
        'low_msg':   '⚠️ Lower to shoulder level between reps.',
        'tip':       '💡 Brace core to protect lower back.',
    },
}

CALORIES_PER_REP = {
    'squat': 0.32,
    'barbell biceps curl': 0.14,
    'push-up': 0.29,
    'shoulder press': 0.19,
}

# ─────────────────────────────────────────
# LOAD RESOURCES
# ─────────────────────────────────────────
@st.cache_resource
def load_model():
    if not os.path.exists("fitness_trainer_model.h5"):
        st.error("❌ Model not found!")
        st.stop()
    return tf.keras.models.load_model("fitness_trainer_model.h5")

@st.cache_resource
def load_pose():
    return mp_pose.Pose(
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
        model_complexity=1
    )

model      = load_model()
pose_model = load_pose()

# ─────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────
defaults = {
    "count": 0, "stage": None,
    "last_feedback": "", "last_exercise": "",
    "rep_confirmed": False,
    "angle_buffer": deque(maxlen=5),
    "chat_history": [],
    "total_calories": 0.0,
    "session_start": time.time(),
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ─────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────
def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return 360 - angle if angle > 180 else angle

def get_angle(exercise, lm):
    if exercise not in THRESHOLDS:
        return 0
    i, j, k = THRESHOLDS[exercise]['joints']
    return calculate_angle([lm[i].x, lm[i].y], [lm[j].x, lm[j].y], [lm[k].x, lm[k].y])

def smooth_angle(raw_angle):
    st.session_state.angle_buffer.append(raw_angle)
    return float(np.mean(st.session_state.angle_buffer))

def get_form_feedback(exercise, angle):
    if exercise not in FORM_RULES:
        return ""
    r = FORM_RULES[exercise]
    lo, hi = r['good_range']
    if lo <= angle <= hi:
        return r['good_msg'] + "\n" + r['tip']
    elif angle > hi:
        return r['high_msg'] + "\n" + r['tip']
    else:
        return r['low_msg'] + "\n" + r['tip']

def update_rep_count(exercise, smoothed_angle):
    thresh      = THRESHOLDS.get(exercise, {})
    down_thresh = thresh.get('down', 90)
    up_thresh   = thresh.get('up',  150)

    if exercise != st.session_state.last_exercise:
        st.session_state.count         = 0
        st.session_state.stage         = None
        st.session_state.rep_confirmed = False
        st.session_state.last_exercise = exercise
        st.session_state.angle_buffer.clear()
        st.session_state.total_calories = 0.0

    feedback = st.session_state.last_feedback

    if smoothed_angle <= down_thresh:
        if st.session_state.stage != "down":
            st.session_state.stage         = "down"
            st.session_state.rep_confirmed = False
            feedback = get_form_feedback(exercise, smoothed_angle)
            st.session_state.last_feedback = feedback

    elif smoothed_angle >= up_thresh:
        if st.session_state.stage == "down" and not st.session_state.rep_confirmed:
            st.session_state.count        += 1
            st.session_state.stage         = "up"
            st.session_state.rep_confirmed = True
            st.session_state.total_calories += CALORIES_PER_REP.get(exercise, 0.2)

    return feedback

def draw_overlay(frame, exercise, confidence, angle, count, feedback, stage):
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w, 55), (20, 20, 20), -1)
    cv2.putText(frame, f"{exercise.upper()}  {confidence:.0%}",
                (10, 38), cv2.FONT_HERSHEY_DUPLEX, 0.85, (0, 255, 255), 2)
    cv2.rectangle(frame, (w-150, 0), (w, 95), (180, 60, 20), -1)
    cv2.putText(frame, "REPS",  (w-130, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
    cv2.putText(frame, str(count), (w-110, 85), cv2.FONT_HERSHEY_DUPLEX, 2.2, (255,255,255), 3)
    cv2.rectangle(frame, (0, h-85), (400, h), (20, 20, 20), -1)
    cv2.putText(frame, f"Angle: {angle:.1f}  Stage: {stage or 'ready'}",
                (8, h-52), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200,200,0), 2)
    if feedback:
        line  = feedback.split('\n')[0]
        color = (0, 210, 0) if "✅" in line else (0, 140, 255)
        cv2.putText(frame, line, (8, h-18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return frame

def run_inference(frame):
    rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose_model.process(rgb)
    exercise, confidence, angle, feedback = "No pose detected", 0.0, 0, ""

    if results.pose_landmarks:
        lm     = results.pose_landmarks.landmark
        coords = np.array([[l.x, l.y, l.z] for l in lm]).flatten()
        pred       = model.predict(coords.reshape(1, 1, 99), verbose=0)
        exercise   = actions[np.argmax(pred)]
        confidence = float(np.max(pred))
        raw_angle  = get_angle(exercise, lm)
        angle      = smooth_angle(raw_angle)
        feedback   = update_rep_count(exercise, angle)

        mp_drawing.draw_landmarks(
            rgb, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing.DrawingSpec(color=(200,0,0), thickness=2, circle_radius=3),
            connection_drawing_spec=mp_drawing.DrawingSpec(color=(255,255,255), thickness=2)
        )

    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    bgr = draw_overlay(bgr, exercise, confidence, angle,
                       st.session_state.count, feedback, st.session_state.stage)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), exercise, confidence, feedback

# ─────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────
st.sidebar.header("🎛️ Controls")
if st.sidebar.button("🔄 Reset Session"):
    st.session_state.count          = 0
    st.session_state.stage          = None
    st.session_state.last_exercise  = ""
    st.session_state.last_feedback  = ""
    st.session_state.rep_confirmed  = False
    st.session_state.total_calories = 0.0
    st.session_state.session_start  = time.time()
    st.session_state.angle_buffer.clear()
    st.sidebar.success("Session reset!")

st.sidebar.markdown("---")
elapsed = int(time.time() - st.session_state.session_start)
mins, secs = divmod(elapsed, 60)
st.sidebar.markdown("### 📊 Workout Dashboard")
st.sidebar.markdown(
    f"""<div style='background:#1e1e2e;padding:14px;border-radius:12px;color:white;font-family:sans-serif'>
      <div style='display:flex;justify-content:space-between;margin-bottom:10px'>
        <div style='text-align:center'>
          <div style='font-size:28px;font-weight:bold;color:#4ade80'>{st.session_state.count}</div>
          <div style='font-size:11px;color:#aaa'>REPS</div>
        </div>
        <div style='text-align:center'>
          <div style='font-size:28px;font-weight:bold;color:#f97316'>{st.session_state.total_calories:.1f}</div>
          <div style='font-size:11px;color:#aaa'>CALORIES</div>
        </div>
        <div style='text-align:center'>
          <div style='font-size:28px;font-weight:bold;color:#60a5fa'>{mins:02d}:{secs:02d}</div>
          <div style='font-size:11px;color:#aaa'>TIME</div>
        </div>
      </div>
      <div style='border-top:1px solid #333;padding-top:8px;margin-top:4px'>
        <div style='font-size:12px;color:#aaa'>Current Exercise</div>
        <div style='font-size:16px;font-weight:bold;color:#e879f9'>
          {st.session_state.last_exercise.title() if st.session_state.last_exercise else "Not detected yet"}
        </div>
        <div style='font-size:12px;color:#aaa;margin-top:6px'>Stage</div>
        <div style='font-size:16px;font-weight:bold;color:#facc15'>
          {(st.session_state.stage or "Ready").upper()}
        </div>
      </div>
    </div>""", unsafe_allow_html=True)

if st.session_state.last_feedback:
    bg = "#e6ffe6" if "✅" in st.session_state.last_feedback else "#fff3cd"
    bd = "green"   if "✅" in st.session_state.last_feedback else "orange"
    st.sidebar.markdown("**💬 Last Form Feedback:**")
    st.sidebar.markdown(
        f"<div style='padding:8px;border-radius:6px;background:{bg};border-left:4px solid {bd}'>"
        f"{st.session_state.last_feedback.replace(chr(10),'<br>')}</div>",
        unsafe_allow_html=True)

st.sidebar.markdown("---")
st.sidebar.markdown("**🏋️ Supported Exercises:**")
for a in actions:
    st.sidebar.markdown(f"- {a.title()}")

# ─────────────────────────────────────────
# TABS
# ─────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📹 Video Analysis", "🎥 Live Webcam", "🤖 Fitness Chatbot"])

# ══════════════════════════════════════════
# TAB 1 — VIDEO UPLOAD
# ══════════════════════════════════════════
with tab1:
    st.subheader("📹 Upload a Workout Video")
    st.info("Upload your workout video. AI classifies exercise, counts reps, and gives form feedback.")

    video_file = st.file_uploader("Choose video", type=["mp4", "mov", "avi"], key="vid")

    if video_file:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tmp.write(video_file.read())
        tmp.close()

        cap          = cv2.VideoCapture(tmp.name)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_ph    = st.empty()
        progress_ph = st.progress(0)
        c1, c2, c3  = st.columns(3)
        feedback_ph = st.empty()
        frame_idx = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % 2 == 0:
                annotated, exercise, conf, feedback = run_inference(frame)
                frame_ph.image(annotated, width=720)
                c1.metric("🏷️ Exercise",   exercise.title())
                c2.metric("📊 Confidence", f"{conf:.0%}")
                c3.metric("🔢 Reps",       st.session_state.count)
                if feedback:
                    bg = "#e6ffe6" if "✅" in feedback else "#fff3cd"
                    bd = "green"   if "✅" in feedback else "orange"
                    feedback_ph.markdown(
                        f"<div style='padding:10px;border-radius:8px;"
                        f"background:{bg};border-left:4px solid {bd}'>"
                        f"{feedback.replace(chr(10),'<br>')}</div>",
                        unsafe_allow_html=True)
            progress_ph.progress(min(frame_idx / max(total_frames, 1), 1.0))
            frame_idx += 1

        cap.release()
        os.unlink(tmp.name)
        st.balloons()
        st.success(f"🎉 Done! Total reps: **{st.session_state.count}** | "
                   f"Calories burned: **{st.session_state.total_calories:.1f} kcal**")

# ══════════════════════════════════════════
# TAB 2 — LIVE WEBCAM (FIXED — no JS, pure st.camera_input)
#
# WHY THE OLD JS VERSION DIDN'T WORK:
#   JavaScript inside a Streamlit iframe cannot trigger a Python rerun
#   or pass image data back to Python. postMessage and query params are
#   both blocked by Streamlit's sandbox. st.camera_input is the ONLY
#   reliable way to get camera frames into Python in Streamlit/Colab.
#
# HOW THIS VERSION WORKS:
#   st.camera_input captures a frame → Python decodes it → run_inference()
#   → results shown immediately. The "Auto-rerun" toggle calls st.rerun()
#   after a delay to simulate a live feed.
# ══════════════════════════════════════════
with tab2:
    st.subheader("🎥 Live Webcam — Real-Time Auto-Classification")

    st.info("📌 Allow camera access → stand **2–3 metres back** (full body visible) → click **Take Photo**")

    st.markdown(
        "<div style='background:#fff3cd;border-left:5px solid orange;padding:10px 14px;"
        "border-radius:6px;margin-bottom:12px'>⚠️ <b>Full body must be in frame — head to feet!</b> "
        "Place your device on a shelf or table 2–3 m away. "
        "Upper-body-only frames = no pose detected.</div>",
        unsafe_allow_html=True)

    # Auto-rerun controls
    ctrl1, ctrl2 = st.columns([1, 1])
    auto_rerun  = ctrl1.toggle("🔄 Auto-rerun (live mode)", value=False)
    rerun_speed = ctrl2.selectbox("Interval", ["Fast (1s)", "Normal (2s)", "Slow (3s)"],
                                  index=1, label_visibility="collapsed")
    delay = {"Fast (1s)": 1.0, "Normal (2s)": 2.0, "Slow (3s)": 3.0}[rerun_speed]

    cam_col, res_col = st.columns([1, 1])

    with cam_col:
        live_frame = st.camera_input(
            "📷 Take photo — ensure FULL BODY is visible",
            key="live_cam"
        )

    with res_col:
        if live_frame:
            file_bytes = np.asarray(bytearray(live_frame.read()), dtype=np.uint8)
            frame      = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            if frame is not None:
                annotated, exercise, conf, feedback = run_inference(frame)

                st.image(annotated, use_column_width=True)

                m1, m2 = st.columns(2)
                m1.metric("🏷️ Exercise",   exercise.title())
                m2.metric("📊 Confidence", f"{conf:.0%}")
                m3, m4 = st.columns(2)
                m3.metric("🔢 Reps",       st.session_state.count)
                m4.metric("🔥 Calories",   f"{st.session_state.total_calories:.1f} kcal")

                if exercise == "No pose detected":
                    st.error("❌ No pose detected — stand further back so your full body fits in frame.")
                else:
                    stage_val = st.session_state.stage or "ready"
                    color_map = {"down": "#f97316", "up": "#4ade80", "ready": "#60a5fa"}
                    sc = color_map.get(stage_val, "#aaa")
                    st.markdown(
                        f"<div style='display:inline-block;padding:4px 14px;border-radius:999px;"
                        f"background:{sc};color:white;font-weight:bold;font-size:13px'>"
                        f"Stage: {stage_val.upper()}</div>", unsafe_allow_html=True)

                    if st.session_state.angle_buffer:
                        ang = list(st.session_state.angle_buffer)[-1]
                        st.markdown(f"**Joint angle: `{ang:.1f}°`**")
                        st.progress(min(int(ang), 180) / 180)

                    if feedback:
                        bg = "#e6ffe6" if "✅" in feedback else "#fff3cd"
                        bd = "green"   if "✅" in feedback else "orange"
                        st.markdown(
                            f"<div style='padding:12px;border-radius:8px;"
                            f"background:{bg};border-left:5px solid {bd};margin-top:8px'>"
                            f"<b>Form Feedback:</b><br>{feedback.replace(chr(10),'<br>')}</div>",
                            unsafe_allow_html=True)

                    thresh = THRESHOLDS.get(exercise, {})
                    st.caption(
                        f"Rep counter: angle ≤{thresh.get('down','?')}° = DOWN  |  "
                        f"angle ≥{thresh.get('up','?')}° = UP  →  DOWN + UP = 1 rep")
        else:
            st.markdown("""
            **Setup checklist:**
            - ☐ Device placed on shelf/table (not hand-held)
            - ☐ Standing 2–3 metres away
            - ☐ Full body head-to-feet visible in preview
            - ☐ Good lighting (face a window)
            - ☐ Camera allowed in browser

            **To count reps:**
            Take a photo at the **bottom** of the movement (DOWN),
            then at the **top** (UP). Each complete cycle = 1 rep.
            Enable **Auto-rerun** to do this automatically.
            """)

    # Auto-rerun loop — simulates live feed
    if auto_rerun and live_frame is not None:
        time.sleep(delay)
        st.rerun()

# ══════════════════════════════════════════
# TAB 3 — CHATBOT
# ══════════════════════════════════════════
with tab3:
    st.subheader("🤖 Fitness Assistant Chatbot")
    st.info("Ask about form, exercises, diet, recovery, or rep counting!")

    def chatbot_reply(msg):
        m = msg.lower()
        if any(w in m for w in ["squat","leg","knee","depth"]):
            return ("🦵 **Squat Form:**\n- Chest up, back straight\n- Knees track over toes\n"
                    "- Thighs parallel at bottom\n- Weight in heels\n\n💡 AI counts rep only when angle <85° then >155°.")
        elif any(w in m for w in ["curl","bicep","biceps","arm"]):
            return ("💪 **Bicep Curl Form:**\n- Elbows pinned to sides\n- Squeeze at top 1 sec\n"
                    "- Lower slowly 3 sec\n\n💡 AI uses elbow angle: <45° = down, >145° = up.")
        elif any(w in m for w in ["push","pushup","chest","tricep"]):
            return ("🤸 **Push-Up Form:**\n- Hands shoulder-width\n- Body straight line\n"
                    "- Chest nearly touches floor\n\n💡 Shoulder angle: <65° = down, >145° = up.")
        elif any(w in m for w in ["shoulder","press","overhead"]):
            return ("🏋️ **Shoulder Press Form:**\n- Start at shoulder height\n- Press straight overhead\n"
                    "- Core braced throughout\n\n💡 Angle: <65° = down, >145° = up.")
        elif any(w in m for w in ["count","rep","wrong","overcount"]):
            return ("🔢 **Rep Counting:**\n- Full DOWN then UP = 1 rep\n- Angles smoothed over 5 frames\n"
                    "- Rep counted only once per cycle\n\n💡 Full body must be visible in frame.")
        elif any(w in m for w in ["calorie","burn","energy"]):
            return ("🔥 **Calorie Estimates per rep:**\n- Squat: ~0.32 kcal\n- Push-up: ~0.29 kcal\n"
                    "- Shoulder Press: ~0.19 kcal\n- Bicep Curl: ~0.14 kcal\n\n💡 Check your sidebar dashboard!")
        elif any(w in m for w in ["diet","protein","food","eat","nutrition"]):
            return ("🥗 **Nutrition:**\n- Protein: 1.6–2.2g per kg bodyweight\n- Muscle gain: +200–300 kcal surplus\n"
                    "- Fat loss: -300–500 kcal deficit\n- Best sources: eggs, chicken, fish, legumes")
        elif any(w in m for w in ["rest","recovery","sleep","sore"]):
            return ("😴 **Recovery:**\n- 7–9 hours sleep\n- 48hr rest per muscle group\n"
                    "- Stay hydrated and hit protein targets")
        elif any(w in m for w in ["warm","warmup","stretch"]):
            return ("🔥 **Warm-Up (5–10 min):**\n- 2 min light cardio\n- Arm circles, leg swings\n"
                    "- Bodyweight reps of main lift\n\n⚠️ Dynamic BEFORE, static AFTER.")
        else:
            return ("🤖 I can help with:\n- **Form**: squat, curl, push-up, shoulder press\n"
                    "- **Calories** burned\n- **Diet & nutrition**\n- **Recovery & sleep**\n- **Warm-up** routines\n\n"
                    "Try: *'Fix my squat form'* or *'How many calories did I burn?'*")

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = st.chat_input("Ask your fitness question...")
    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)
        reply = chatbot_reply(user_input)
        st.session_state.chat_history.append({"role": "assistant", "content": reply})
        with st.chat_message("assistant"):
            st.markdown(reply)

    if st.button("🗑️ Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()
