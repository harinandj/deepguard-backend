from flask import Flask, render_template, request, jsonify
import os
import cv2
import librosa
import subprocess
import shutil
import tempfile
import numpy as np
import torch

from PIL import Image
from transformers import (
    VideoMAEForVideoClassification,
    VideoMAEImageProcessor,
)

app = Flask(__name__)

# ============================================================
# FOLDERS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
FRAMES_FOLDER = os.path.join(BASE_DIR, "static", "frames")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(FRAMES_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

ALLOWED_EXTENSIONS = {"mp4", "mov", "avi", "webm"}

# ============================================================
# FACE DETECTOR
# ============================================================

FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)

# ============================================================
# VIDEO DEEPFAKE MODEL
# ============================================================
# This is a VIDEO model, not the previous single-image model.
#
# It samples 16 frames from the video and uses VideoMAE's
# spatiotemporal representation to classify the video.
#
# Model:
# Vansh180/VideoMae-ffc23-deepfake-detector
#
# The model card reports 88% validation accuracy on its
# FaceForensics++ validation setup. This does NOT mean 88%
# accuracy on every real-world video.
# ============================================================

VIDEO_MODEL_NAME = "Vansh180/VideoMae-ffc23-deepfake-detector"

print("Loading VideoMAE deepfake detector...")
print("The first run may download the model (~345 MB).")

processor = VideoMAEImageProcessor.from_pretrained(
    VIDEO_MODEL_NAME
)

video_model = VideoMAEForVideoClassification.from_pretrained(
    VIDEO_MODEL_NAME
)

video_model.eval()

print("VideoMAE model loaded successfully!")

# ============================================================
# HELPERS
# ============================================================

def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower()
        in ALLOWED_EXTENSIONS
    )


def get_label_name(model, index):
    """Get a readable label from the model configuration."""
    label = model.config.id2label.get(index, str(index))
    return str(label).lower()


# ============================================================
# FACE DETECTION / CONSISTENCY
# ============================================================

def analyze_faces(
    video_path,
    output_frame_path,
    frame_interval=15,
    max_frames_to_check=40
):
    """
    Classic OpenCV face detection.

    IMPORTANT:
    This is NOT a deepfake classifier.
    The ratio only means how often a face was detected.
    """

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        cap.release()

        return {
            "readable": False,
            "frames_analyzed": 0,
            "frames_with_face": 0,
            "face_ratio_percent": 0,
            "consistent": False,
            "preview_saved": False
        }

    frame_index = 0
    frames_analyzed = 0
    frames_with_face = 0
    preview_saved = False

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        if frame_index % frame_interval == 0:

            frames_analyzed += 1

            gray = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2GRAY
            )

            faces = FACE_CASCADE.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(60, 60)
            )

            if len(faces) > 0:

                frames_with_face += 1

                if not preview_saved:

                    annotated = frame.copy()

                    for (x, y, w, h) in faces:

                        cv2.rectangle(
                            annotated,
                            (x, y),
                            (x + w, y + h),
                            (0, 200, 0),
                            3
                        )

                    cv2.imwrite(
                        output_frame_path,
                        annotated
                    )

                    preview_saved = True

            if frames_analyzed >= max_frames_to_check:
                break

        frame_index += 1

    cap.release()

    face_ratio_percent = (
        round(
            frames_with_face /
            frames_analyzed *
            100,
            1
        )
        if frames_analyzed
        else 0
    )

    # Detection consistency only. NOT authenticity.
    consistent = face_ratio_percent >= 80

    return {
        "readable": True,
        "frames_analyzed": frames_analyzed,
        "frames_with_face": frames_with_face,
        "face_ratio_percent": face_ratio_percent,
        "consistent": consistent,
        "preview_saved": preview_saved
    }


# ============================================================
# FFMPEG
# ============================================================

def find_ffmpeg():

    ffmpeg = shutil.which("ffmpeg")

    if ffmpeg:
        return ffmpeg

    possible_paths = [
        r"C:\ffmpeg\bin\ffmpeg.exe",
        r"C:\Program Files\ffmpeg\bin\ffmpeg.exe",
        r"C:\Program Files (x86)\ffmpeg\bin\ffmpeg.exe",
    ]

    for path in possible_paths:

        if os.path.exists(path):
            return path

    return None


# ============================================================
# AUDIO ANALYSIS
# ============================================================

def analyze_voice(video_path):

    ffmpeg = find_ffmpeg()

    if ffmpeg is None:

        return {
            "audio_detected": False,
            "duration": 0,
            "sample_rate": 0,
            "rms_energy": 0,
            "zero_crossing_rate": 0,
            "spectral_centroid": 0,
            "error": "FFmpeg was not found."
        }

    temp_wav = os.path.join(
        tempfile.gettempdir(),
        "deepguard_audio.wav"
    )

    try:

        command = [
            ffmpeg,
            "-y",
            "-i",
            video_path,
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-acodec",
            "pcm_s16le",
            temp_wav
        ]

        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        if (
            result.returncode != 0
            or not os.path.exists(temp_wav)
        ):

            return {
                "audio_detected": False,
                "duration": 0,
                "sample_rate": 0,
                "rms_energy": 0,
                "zero_crossing_rate": 0,
                "spectral_centroid": 0,
                "error": "Could not extract audio."
            }

        y, sr = librosa.load(
            temp_wav,
            sr=None,
            mono=True
        )

        duration = librosa.get_duration(
            y=y,
            sr=sr
        )

        rms = float(
            librosa.feature.rms(
                y=y
            ).mean()
        )

        zcr = float(
            librosa.feature.zero_crossing_rate(
                y
            ).mean()
        )

        spectral_centroid = float(
            librosa.feature.spectral_centroid(
                y=y,
                sr=sr
            ).mean()
        )

        return {
            "audio_detected": True,
            "duration": round(duration, 2),
            "sample_rate": sr,
            "rms_energy": round(rms, 5),
            "zero_crossing_rate": round(zcr, 5),
            "spectral_centroid": round(spectral_centroid, 2),
            "error": None
        }

    except Exception as e:

        return {
            "audio_detected": False,
            "duration": 0,
            "sample_rate": 0,
            "rms_energy": 0,
            "zero_crossing_rate": 0,
            "spectral_centroid": 0,
            "error": str(e)
        }

    finally:

        if os.path.exists(temp_wav):

            try:
                os.remove(temp_wav)
            except Exception:
                pass


# ============================================================
# VIDEO DEEPFAKE ANALYSIS
# ============================================================

@torch.no_grad()
def analyze_ai(video_path):
    """
    Uses a true video classification model.

    The model expects 16 frames sampled from the video.
    We sample 16 frames uniformly across the entire video,
    exactly matching the approach described in the model card.

    This is still a prototype and cannot guarantee perfect
    real-world accuracy.
    """

    try:

        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():

            return {
                "available": False,
                "prediction": "UNAVAILABLE",
                "confidence": 0,
                "fake_score": 0,
                "real_score": 0,
                "frames_used": 0,
                "error": "Could not open video."
            }

        total_frames = int(
            cap.get(cv2.CAP_PROP_FRAME_COUNT)
        )

        if total_frames <= 0:

            cap.release()

            return {
                "available": False,
                "prediction": "UNAVAILABLE",
                "confidence": 0,
                "fake_score": 0,
                "real_score": 0,
                "frames_used": 0,
                "error": "Could not determine video length."
            }

        # VideoMAE model card uses 16 uniformly sampled frames.
        num_frames = 16

        indices = np.linspace(
            0,
            total_frames - 1,
            num_frames
        ).astype(int)

        frames = []

        for index in indices:

            cap.set(
                cv2.CAP_PROP_POS_FRAMES,
                int(index)
            )

            success, frame = cap.read()

            if not success:
                continue

            frame_rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            frames.append(
                Image.fromarray(frame_rgb)
            )

        cap.release()

        if len(frames) == 0:

            return {
                "available": False,
                "prediction": "UNAVAILABLE",
                "confidence": 0,
                "fake_score": 0,
                "real_score": 0,
                "frames_used": 0,
                "error": "No video frames could be read."
            }

        # If a very short/damaged video gives fewer than 16 frames,
        # repeat the last usable frame so the model still receives
        # the expected number of frames.
        while len(frames) < num_frames:
            frames.append(frames[-1].copy())

        # Keep exactly 16 frames.
        frames = frames[:num_frames]

        print(
            f"VideoMAE: using {len(frames)} frames"
        )

        inputs = processor(
            frames,
            return_tensors="pt"
        )

        outputs = video_model(
            **inputs
        )

        probabilities = torch.softmax(
            outputs.logits,
            dim=1
        )[0]

        # Read the model's own label mapping.
        labels = {
            i: get_label_name(video_model, i)
            for i in range(len(probabilities))
        }

        print("Model labels:", labels)

        real_index = None
        fake_index = None

        for index, label in labels.items():

            if "real" in label:
                real_index = index

            if "fake" in label:
                fake_index = index

        # The model card specifies:
        # index 0 = Real, index 1 = Fake.
        if real_index is None and len(probabilities) > 0:
            real_index = 0

        if fake_index is None and len(probabilities) > 1:
            fake_index = 1

        real_score = float(
            probabilities[real_index]
        )

        fake_score = float(
            probabilities[fake_index]
        )

        if fake_score >= real_score:

            prediction = "FAKE"
            confidence = fake_score

        else:

            prediction = "REAL"
            confidence = real_score

        result = {
            "available": True,
            "prediction": prediction,
            "confidence": round(
                confidence * 100,
                1
            ),
            "fake_score": round(
                fake_score * 100,
                1
            ),
            "real_score": round(
                real_score * 100,
                1
            ),
            "frames_used": len(frames),
            "error": None
        }

        print("VideoMAE result:")
        print(result)

        return result

    except Exception as e:

        print(
            "VideoMAE ERROR:",
            str(e)
        )

        return {
            "available": False,
            "prediction": "UNAVAILABLE",
            "confidence": 0,
            "fake_score": 0,
            "real_score": 0,
            "frames_used": 0,
            "error": str(e)
        }


# ============================================================
# HOME
# ============================================================

@app.route("/")
def index():

    return render_template(
        "index.html"
    )


# ============================================================
# UPLOAD
# ============================================================

@app.route(
    "/upload",
    methods=["POST"]
)
def upload():

    if "video" not in request.files:

        return render_template(
            "result.html",
            error="No video was sent."
        )

    file = request.files["video"]

    if file.filename == "":

        return render_template(
            "result.html",
            error="No video was selected."
        )

    if not allowed_file(
        file.filename
    ):

        return render_template(
            "result.html",
            error=(
                "Unsupported video format. "
                "Please upload mp4, mov, avi, or webm."
            )
        )

    save_path = os.path.join(
        UPLOAD_FOLDER,
        file.filename
    )

    file.save(
        save_path
    )

    preview_filename = (
        os.path.splitext(
            file.filename
        )[0]
        + "_face_preview.jpg"
    )

    preview_path = os.path.join(
        FRAMES_FOLDER,
        preview_filename
    )

    print("Analyzing face...")
    face_results = analyze_faces(
        save_path,
        preview_path
    )

    print("Analyzing audio...")
    voice_results = analyze_voice(
        save_path
    )

    print("Running VIDEO deepfake model...")
    ai_results = analyze_ai(
        save_path
    )

    print("FINAL AI RESULT:")
    print(ai_results)

    return render_template(
        "result.html",
        filename=file.filename,
        face_results=face_results,
        voice_results=voice_results,
        ai_results=ai_results,
        preview_filename=(
            preview_filename
            if face_results["preview_saved"]
            else None
        )
    )
# ============================================================
# LOVABLE API UPLOAD
# ============================================================

@app.route("/api/upload", methods=["POST"])
def api_upload():

    if "video" not in request.files:
        return jsonify({
            "error": "No video was sent."
        }), 400

    file = request.files["video"]

    if file.filename == "":
        return jsonify({
            "error": "No video was selected."
        }), 400

    if not allowed_file(file.filename):
        return jsonify({
            "error": "Unsupported video format. Please upload mp4, mov, avi, or webm."
        }), 400

    save_path = os.path.join(
        UPLOAD_FOLDER,
        file.filename
    )

    file.save(save_path)

    preview_filename = (
        os.path.splitext(file.filename)[0]
        + "_face_preview.jpg"
    )

    preview_path = os.path.join(
        FRAMES_FOLDER,
        preview_filename
    )

    print("Analyzing face...")
    face_results = analyze_faces(
        save_path,
        preview_path
    )

    print("Analyzing audio...")
    voice_results = analyze_voice(
        save_path
    )

    print("Running VIDEO deepfake model...")
    ai_results = analyze_ai(
        save_path
    )

    print("FINAL AI RESULT:")
    print(ai_results)

    return jsonify({
        "filename": file.filename,
        "ai": ai_results,
        "face": face_results,
        "voice": voice_results,
        "preview_filename": (
            preview_filename
            if face_results.get("preview_saved")
            else None
        )
    })

# ============================================================
# START SERVER
# ============================================================

if __name__ == "__main__":

    app.run(
        debug=True
    )
