# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project does

Detects the "Signal for Help" domestic-violence hand gesture via webcam (OpenCV + MediaPipe ONNX model) and sends `SFH\n` over a serial port to an Arduino, which lights an LED. The gesture must be completed 3 times within 10 seconds to trigger the alert.

## Build and run (C++ desktop app)

```bash
# First-time: download the ~4 MB ONNX model
bash download_models.sh

# Build
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . -j$(nproc)

# Run (from project root or build/)
./build/signal_for_help
./build/signal_for_help models/handpose_estimation_mediapipe_2023feb.onnx 0 /dev/ttyACM0 9600
```

Requires OpenCV 4 with the `dnn` module: `pkg-config --modversion opencv4`.

## Arduino firmware (PlatformIO)

The firmware lives in `arduino/sfh_led/`. It listens on serial for `SFH\n` and pulses pin 11 for 3 seconds.

```bash
# Upload to Arduino Uno on /dev/ttyACM0
cd arduino/sfh_led
pio run --target upload

# Open serial monitor
pio device monitor
```

## Architecture

The pipeline runs per-frame in `src/main.cpp`:

1. **`SkinDetector`** — converts frame to YCrCb, thresholds Cr/Cb channels, applies morphological open+close, returns candidate `cv::Rect` ROIs.
2. **`HandPoseEstimator`** — crops each ROI, runs the MediaPipe ONNX model (224×224 **NHWC** input — must be built manually, not with `blobFromImage`), returns 21 landmarks + confidence score. Detections below 0.35 confidence are discarded.
3. **`GestureDetector`** — 3-state machine (IDLE → OPEN_HAND → THUMB_TUCKED → SIGNAL_COMPLETE). Uses wrist-relative distances to classify finger state; tolerates up to 8 noisy frames before resetting.
4. **`AlertSystem`** — renders a red overlay + plays `\a` + calls `notify-send`. Active for 8 seconds.
5. **`SerialComm`** — POSIX `termios` / Win32 `CreateFile` wrapper; sends `SFH\n` when 3 gestures complete.

### Key tunable constants in `src/main.cpp`

| Constant | Default | Effect |
|---|---|---|
| `GESTURES_TO_ALERT` | `3` | Gestures required before alert fires |
| `COUNTER_RESET_SEC` | `10.0f` | Seconds of inactivity before counter resets |

### Key tunable constants in `src/gesture_detector.cpp`

| Value | Meaning |
|---|---|
| `1.15f` | Finger-extended threshold (tip/wrist vs MCP/wrist ratio) |
| `1.25f` | Finger-curled threshold |
| `0.85f` | Thumb-tucked threshold (tip-to-palm-center / palmLen) |
| `openHoldSec=0.8f` | Seconds open palm must be held |
| `thumbHoldSec=0.5f` | Seconds thumb-tucked must be held |
| `MAX_FAIL_FRAMES=8` | Noisy frames tolerated before state regression |

## Runtime keyboard controls

| Key | Action |
|---|---|
| `Q` / `Esc` | Quit |
| `R` | Reset gesture state machine and counter |
| `G` | Toggle debug panel (per-finger state) |
| `D` | Toggle skin-mask overlay |
| `V` | Toggle verbose geometric values in console |
