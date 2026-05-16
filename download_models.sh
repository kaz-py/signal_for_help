#!/usr/bin/env bash
# Downloads the MediaPipe hand-pose ONNX model from the OpenCV model zoo.
# The model detects 21 hand landmarks and provides a hand-presence score
# that we use to reject false detections from non-hand objects.

set -e

MODELS_DIR="$(cd "$(dirname "$0")" && pwd)/models"
mkdir -p "$MODELS_DIR"

MODEL_FILE="handpose_estimation_mediapipe_2023feb.onnx"
MODEL_PATH="$MODELS_DIR/$MODEL_FILE"

# Primary source: OpenCV model zoo (GitHub releases)
PRIMARY_URL="https://github.com/opencv/opencv_zoo/raw/main/models/handpose_estimation_mediapipe/${MODEL_FILE}"
# Mirror: HuggingFace (opencv/handpose_estimation_mediapipe)
MIRROR_URL="https://huggingface.co/opencv/handpose_estimation_mediapipe/resolve/main/${MODEL_FILE}"

download_model() {
    local url="$1"
    echo "Downloading from: $url"
    if command -v wget &>/dev/null; then
        wget -q --show-progress -O "$MODEL_PATH" "$url"
    elif command -v curl &>/dev/null; then
        curl -L --progress-bar -o "$MODEL_PATH" "$url"
    else
        echo "ERROR: Neither wget nor curl is installed." >&2
        exit 1
    fi
}

if [ -f "$MODEL_PATH" ]; then
    echo "Model already present: $MODEL_PATH"
    echo "Delete it and re-run this script to force a re-download."
    exit 0
fi

echo "=== Downloading MediaPipe hand-pose model ==="
if ! download_model "$PRIMARY_URL"; then
    echo "Primary download failed, trying mirror..."
    download_model "$MIRROR_URL"
fi

echo ""
echo "Model saved to: $MODEL_PATH"
echo ""
echo "Build and run:"
echo "  mkdir -p build && cd build"
echo "  cmake .. -DCMAKE_BUILD_TYPE=Release"
echo "  cmake --build . -j\$(nproc)"
echo "  ./signal_for_help"
