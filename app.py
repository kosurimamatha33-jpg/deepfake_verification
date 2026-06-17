import streamlit as st
import cv2
import numpy as np
from utils import load_image, load_video
from liveness import liveness_check
from eyebrow import detect_eyebrow_anomaly, reset_frame_history
from deepfake import analyze_frame
import os, time

st.set_page_config(page_title="Deepfake Verification System", layout="wide")
st.title("🔍 AI-Powered Content Authenticity System")
st.markdown("**Main Feature: Eyebrow Pattern Analysis — 5-Factor Deep Inspection**")

mode = st.sidebar.radio("Select Input Mode",
                         ["Live Camera Verification", "Upload Image", "Upload Video"])


# ─────────────────────────────────────────────────────────────
# SHARED: eyebrow card renderer
# ─────────────────────────────────────────────────────────────
def render_eyebrow_panel(ea, show_title=True):
    """Render the full 5-factor eyebrow breakdown from an analysis dict."""
    if show_title:
        st.subheader("👁️ EYEBROW PATTERN ANALYSIS — 5-Factor Report")

    # Factor tiles (row 1: quality factors)
    st.markdown("**🔎 Quality Factors (how well the image allows detection)**")
    c1, c2, c3, c4, c5 = st.columns(5)

    def tile(col, icon, label, score, msg):
        pct = int(score * 100)
        colour = "🟢" if score >= 0.75 else ("🟡" if score >= 0.55 else "🔴")
        col.metric(f"{icon} {label}", f"{colour} {pct}%")
        col.caption(msg)

    tile(c1, "💡", "Lighting",    ea["lighting"],    ea["lighting_msg"])
    tile(c2, "📷", "Image Quality",ea["quality"],    ea["quality_msg"])
    tile(c3, "📐", "Face Angle",   ea["angle"],       ea["angle_msg"])
    tile(c4, "👤", "Face Visible", ea["visibility"],  ea["visibility_msg"])
    tile(c5, "🎞️", "Consistency",  ea["consistency"], ea["consistency_msg"])

    st.markdown("---")
    # Row 2: eyebrow features
    st.markdown("**🧬 Eyebrow Feature Analysis**")
    f1, f2, f3, f4, f5 = st.columns(5)
    tile(f1, "🌀", "Shape",       ea["shape"],      ea["shape_msg"])
    tile(f2, "🔬", "Hair Density",ea["density"],    ea["density_msg"])
    tile(f3, "〰️", "Continuity",  ea["continuity"], ea["continuity_msg"])
    tile(f4, "🪡", "Texture",     ea["texture"],    ea["texture_msg"])
    tile(f5, "⚖️", "Symmetry",    ea["symmetry"],   ea["symmetry_msg"])

    st.markdown("---")
    overall = ea["overall_score"]
    if overall >= 0.75:
        st.success(f"### 👁️ Eyebrow Overall: AUTHENTIC — {overall*100:.0f}%\n"
                   f"Natural eyebrow patterns confirmed across all 5 factors.")
    elif overall >= 0.55:
        st.warning(f"### 👁️ Eyebrow Overall: MIXED SIGNALS — {overall*100:.0f}%\n"
                   f"Some factors flagged — manual review recommended.")
    else:
        st.error(f"### 👁️ Eyebrow Overall: ANOMALY — {overall*100:.0f}%\n"
                 f"Significant abnormalities detected — likely AI/deepfake.")


# ─────────────────────────────────────────────────────────────
# CORE FRAME PROCESSOR
# ─────────────────────────────────────────────────────────────
def process_frame(frame, prev_face=None, prev_ear=None, track_history=False):
    liveness_result, face, ear = liveness_check(frame, prev_face, prev_ear)
    liveness_ok     = liveness_result["liveness"]
    liveness_reason = liveness_result["reason"]

    deepfake_result = analyze_frame(frame)
    fake_prob       = deepfake_result["fake_probability"]

    eyebrow_score, eyebrow_analysis = detect_eyebrow_anomaly(
        frame, track_history=track_history)
    eyebrow_is_anomaly = eyebrow_analysis["is_anomaly"]

    liveness_score  = 0.9 if liveness_ok else 0.1
    deepfake_score  = 1.0 - fake_prob

    # Eyebrows 45 %, Deepfake 30 %, Liveness 25 %
    overall_score = (eyebrow_score   * 0.45 +
                     deepfake_score  * 0.30 +
                     liveness_score  * 0.25)

    confidence    = round(min(0.7 + 0.3 * overall_score, 1.0), 2)

    warnings = []
    if eyebrow_is_anomaly and fake_prob > 0.5:
        decision     = "FAKE";       content_type = "AI-GENERATED"
        warnings.append("❌ Eyebrow anomaly + deepfake signal — high risk")
    elif not liveness_ok:
        decision     = "FAKE";       content_type = "AI-GENERATED"
        warnings.append("❌ Liveness failed: " + liveness_reason)
    elif overall_score < 0.50:
        decision     = "FAKE";       content_type = "AI-GENERATED"
        warnings.append("❌ Multiple authenticity indicators failed")
    elif overall_score < 0.68:
        decision     = "SUSPICIOUS"; content_type = "POTENTIALLY AI-GENERATED"
        warnings.append("⚠️ Mixed signals — manual review recommended")
    else:
        decision     = "REAL";       content_type = "HUMAN-CREATED"
        warnings.append("✅ All checks passed")

    if eyebrow_is_anomaly:
        warnings.append(f"⚠️ Eyebrow issues: {eyebrow_score*100:.0f}% authenticity")
    if fake_prob > 0.6:
        warnings.append("⚠️ Deepfake heuristic indicates manipulation")

    return {
        "decision":     decision,
        "content_type": content_type,
        "confidence":   confidence,
        "warnings":     warnings,
        "details": {
            "liveness":          liveness_ok,
            "liveness_reason":   liveness_reason,
            "deepfake_prob":     fake_prob,
            "eyebrow_score":     eyebrow_score,
            "eyebrow_analysis":  eyebrow_analysis,
            "eyebrow_is_anomaly":eyebrow_is_anomaly
        },
        "face_bbox": face,
        "ear":       ear
    }


# ═════════════════════════════════════════════════════════════
# LIVE CAMERA
# ═════════════════════════════════════════════════════════════
if mode == "Live Camera Verification":
    st.write("### 👤 Live Person Verification")

    col_inst, col_stat = st.columns([2, 1])
    with col_inst:
        st.info("""
**📋 Instructions for Best Eyebrow Analysis:**

1. 💡 **Good Lighting** — Face a window or lamp; avoid back-lighting
2. 👤 **Full Face Visible** — Keep eyebrows fully in frame, nothing covering them
3. 📐 **Direct Angle** — Look straight at the camera (slight tilt is fine)
4. 📷 **Stay Still** — Hold steady so the camera captures sharp detail
5. 👁️ **Blink Naturally** — Blink 2-3 times during verification
6. ↔️ **Slight Head Turn** — A small left/right turn helps depth analysis

The system captures **multiple frames** to build a consistency score.
        """)
    with col_stat:
        st.markdown("**⏱️ Live Status**")
        status_ph      = st.empty()
        feedback_ph    = st.empty()

    run = st.checkbox("🎥 Start Live Verification", value=False)
    frame_window = st.image([])

    if run:
        reset_frame_history()
        cap = cv2.VideoCapture(0)
        prev_face = prev_ear = None
        frame_count  = 0
        eyebrow_scores = []
        last_analysis  = None

        status_ph.info("🔴 Initializing camera…")

        while True:
            ret, frame = cap.read()
            if not ret:
                st.warning("❌ Camera not accessible.")
                break

            frame_window.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                               use_column_width=True)

            if frame_count % 4 == 0:
                result       = process_frame(frame, prev_face, prev_ear,
                                             track_history=True)
                prev_face    = result["face_bbox"]
                prev_ear     = result["ear"]
                ea           = result["details"]["eyebrow_analysis"]
                escore       = result["details"]["eyebrow_score"]
                last_analysis = ea
                eyebrow_scores.append(escore)

                # Build live status
                lines = [
                    f"👤 Face:      {'✅ Detected' if prev_face else '❌ Not found'}",
                    f"💡 Lighting:  {'✅' if ea['lighting'] >= 0.6 else '⚠️'} {int(ea['lighting']*100)}%",
                    f"📐 Angle:     {'✅' if ea['angle']   >= 0.7 else '⚠️'} {int(ea['angle']*100)}%",
                    f"👁️ Eyebrows:  {'✅' if escore       >= 0.6 else '⚠️'} {int(escore*100)}%",
                    f"🎞️ Frames:    {len(eyebrow_scores)}",
                ]
                status_ph.code("\n".join(lines))

                # Contextual feedback tip
                if ea["lighting"] < 0.55:
                    feedback_ph.warning("💡 Tip: Move to better lighting")
                elif ea["angle"] < 0.55:
                    feedback_ph.warning("📐 Tip: Face the camera more directly")
                elif ea["visibility"] < 0.55:
                    feedback_ph.warning("👤 Tip: Make sure both eyebrows are visible")
                elif escore < 0.55:
                    feedback_ph.error("⚠️ Eyebrow anomaly detected — potential deepfake")
                else:
                    feedback_ph.success("✅ Good — keep steady")

            frame_count += 1
            if frame_count > 200:
                break

        cap.release()

        # ── Final verdict ────────────────────────────────────
        if eyebrow_scores and last_analysis:
            st.write("---")
            st.subheader("✅ Verification Complete")

            avg_eb  = np.mean(eyebrow_scores)
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("👁️ Avg Eyebrow Score", f"{avg_eb*100:.0f}%")
            c2.metric("🎞️ Frames Analysed",   len(eyebrow_scores))
            c3.metric("💡 Lighting",           f"{last_analysis['lighting']*100:.0f}%")
            c4.metric("📐 Face Angle",         f"{last_analysis['angle']*100:.0f}%")

            render_eyebrow_panel(last_analysis)

            st.write("---")
            if avg_eb >= 0.70:
                st.success("✅ **HUMAN VERIFIED** — Eyebrow patterns authentic across all 5 factors.")
            elif avg_eb >= 0.55:
                st.warning("⚠️ **UNCERTAIN** — Eyebrow patterns show mixed signals.")
            else:
                st.error("❌ **FAKE / AI-GENERATED** — Eyebrow anomalies detected.")


# ═════════════════════════════════════════════════════════════
# UPLOAD IMAGE
# ═════════════════════════════════════════════════════════════
elif mode == "Upload Image":
    st.write("### 📸 Image Authenticity Analysis")
    st.caption("👁️ Main Feature: Eyebrow Pattern Analysis across all 5 quality factors")

    uploaded = st.file_uploader("📁 Choose an image…", type=["jpg", "jpeg", "png"])

    if uploaded is not None:
        img = load_image(uploaded)

        col_img, col_res = st.columns([1, 1])
        with col_img:
            st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
                     caption="Uploaded Image", use_column_width=True)

        with col_res:
            with st.spinner("🔍 Running 5-factor eyebrow analysis…"):
                reset_frame_history()
                result = process_frame(img, track_history=False)
                time.sleep(0.8)

            st.subheader("📊 Overall Verdict")
            ct = result["content_type"]
            conf_pct = result["confidence"] * 100

            if "HUMAN" in ct:
                st.success(f"### ✅ {ct}")
                st.write("This image appears to be **human-created**.")
            elif "AI" in ct:
                st.error(f"### ❌ {ct}")
                st.write("This image shows **signs of AI generation or deepfake**.")
            else:
                st.warning(f"### ⚠️ {ct}")
                st.write("Mixed signals — **manual review recommended**.")

            st.metric("Overall Confidence", f"{conf_pct:.1f}%")

            if result["warnings"]:
                for w in result["warnings"]:
                    st.write(f"• {w}")

        # Full eyebrow panel below the image
        st.write("---")
        render_eyebrow_panel(result["details"]["eyebrow_analysis"])

        # Supporting checks
        with st.expander("🔬 Supporting Checks (Liveness & Deepfake)"):
            c1, c2 = st.columns(2)
            with c1:
                st.info("**Liveness Check**\n"
                        + ("✅ Passed" if result["details"]["liveness"] else "❌ Failed"))
            with c2:
                dp = result["details"]["deepfake_prob"] * 100
                label = "⚠️ Likely Deepfake" if dp > 60 else "✅ Looks Natural"
                st.info(f"**Deepfake Score**\n{label} — {dp:.1f}%")


# ═════════════════════════════════════════════════════════════
# UPLOAD VIDEO
# ═════════════════════════════════════════════════════════════
elif mode == "Upload Video":
    st.write("### 🎬 Video Authenticity Analysis")
    st.caption("👁️ Main Feature: Eyebrow Pattern Analysis — frame-by-frame across all 5 factors")

    uploaded = st.file_uploader("📁 Choose a video…", type=["mp4", "avi", "mov"])

    if uploaded is not None:
        with open("temp_video.mp4", "wb") as f:
            f.write(uploaded.getbuffer())
        st.video("temp_video.mp4")

        with st.spinner("🔍 Analysing eyebrow patterns frame-by-frame…"):
            reset_frame_history()
            cap          = cv2.VideoCapture("temp_video.mp4")
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            sample_rate  = max(1, total_frames // 30)
            decisions, confidences, warnings_set = [], [], set()
            eb_scores, lighting_scores = [], []
            quality_scores, angle_scores, visibility_scores = [], [], []
            shape_s, density_s, continuity_s, texture_s, symmetry_s = [], [], [], [], []
            analyses = []

            prog = st.progress(0)
            for i in range(0, total_frames, sample_rate):
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ret, frame = cap.read()
                if not ret:
                    break
                r  = process_frame(frame, track_history=True)
                ea = r["details"]["eyebrow_analysis"]
                decisions.append(r["decision"])
                confidences.append(r["confidence"])
                for w in r["warnings"]:
                    warnings_set.add(w)
                eb_scores.append(r["details"]["eyebrow_score"])
                lighting_scores.append(ea["lighting"])
                quality_scores.append(ea["quality"])
                angle_scores.append(ea["angle"])
                visibility_scores.append(ea["visibility"])
                shape_s.append(ea["shape"])
                density_s.append(ea["density"])
                continuity_s.append(ea["continuity"])
                texture_s.append(ea["texture"])
                symmetry_s.append(ea["symmetry"])
                analyses.append(ea)
                prog.progress(min(len(eb_scores) / 30, 1.0))

            cap.release()
            time.sleep(0.5)

        st.write("---")
        st.subheader("📊 Video Analysis Results")

        from collections import Counter
        if decisions:
            final_dec    = Counter(decisions).most_common(1)[0][0]
            avg_conf     = np.mean(confidences)
            avg_eb       = np.mean(eb_scores)
            n_frames     = len(eb_scores)

            # Verdict
            if final_dec == "REAL" and avg_eb >= 0.65:
                st.success(f"### ✅ HUMAN-CREATED  ({avg_eb*100:.0f}% eyebrow authenticity)")
            elif final_dec == "FAKE" or avg_eb < 0.50:
                st.error(f"### ❌ AI-GENERATED / DEEPFAKE  ({avg_eb*100:.0f}% eyebrow authenticity)")
            else:
                st.warning(f"### ⚠️ UNCERTAIN — Manual Review  ({avg_eb*100:.0f}% eyebrow authenticity)")

            # Summary metrics
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("👁️ Avg Eyebrow",   f"{avg_eb*100:.0f}%")
            c2.metric("📊 Confidence",     f"{avg_conf*100:.0f}%")
            c3.metric("🎞️ Frames Checked", n_frames)
            c4.metric("✅ Real Frames",
                      f"{decisions.count('REAL')}/{n_frames}")

            # ── 5-Factor averages ────────────────────────────
            st.write("---")
            st.subheader("👁️ 5-Factor Eyebrow Summary (video average)")

            st.markdown("**🔎 Quality Factors**")
            q1, q2, q3, q4, q5 = st.columns(5)

            def avg_tile(col, icon, label, scores):
                avg = np.mean(scores)
                clr = "🟢" if avg >= 0.75 else ("🟡" if avg >= 0.55 else "🔴")
                col.metric(f"{icon} {label}", f"{clr} {avg*100:.0f}%")

            avg_tile(q1, "💡", "Lighting",     lighting_scores)
            avg_tile(q2, "📷", "Image Quality",quality_scores)
            avg_tile(q3, "📐", "Face Angle",   angle_scores)
            avg_tile(q4, "👤", "Face Visible", visibility_scores)
            consistency_avgs = [a["consistency"] for a in analyses]
            avg_tile(q5, "🎞️", "Consistency",  consistency_avgs)

            st.markdown("**🧬 Eyebrow Features**")
            f1, f2, f3, f4, f5 = st.columns(5)
            avg_tile(f1, "🌀", "Shape",      shape_s)
            avg_tile(f2, "🔬", "Density",    density_s)
            avg_tile(f3, "〰️", "Continuity", continuity_s)
            avg_tile(f4, "🪡", "Texture",    texture_s)
            avg_tile(f5, "⚖️", "Symmetry",   symmetry_s)

            # ── Frame-by-frame chart ─────────────────────────
            st.write("---")
            st.subheader("📈 Eyebrow Score per Frame")
            st.line_chart({"Eyebrow Authenticity": eb_scores,
                           "Lighting":             lighting_scores,
                           "Face Angle":           angle_scores})

            # ── Issues ───────────────────────────────────────
            if warnings_set:
                st.write("---")
                st.subheader("⚠️ Issues Detected")
                for w in list(warnings_set)[:6]:
                    st.write(f"• {w}")

            # ── Most recent frame detail ──────────────────────
            if analyses:
                with st.expander("🔬 Last Frame — Full Detail"):
                    render_eyebrow_panel(analyses[-1], show_title=False)

        if os.path.exists("temp_video.mp4"):
            os.remove("temp_video.mp4")