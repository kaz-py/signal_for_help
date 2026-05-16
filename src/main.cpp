#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>

#include "skin_detector.hpp"
#include "hand_pose.hpp"
#include "gesture_detector.hpp"
#include "alert_system.hpp"

// ---------------------------------------------------------------------------
// Expand ROI with asymmetric padding:
//   - sideF  : fraction added to left/right/top
//   - bottomF: fraction added to bottom (more space for the wrist)
// This ensures the wrist landmark is always inside the crop, which is
// critical for the distance-from-wrist gesture calculations.
// ---------------------------------------------------------------------------
static cv::Rect expandROI(const cv::Rect& r, const cv::Size& imgSz,
                          float sideF = 0.25f, float bottomF = 0.40f) {
    int dx    = static_cast<int>(r.width  * sideF);
    int dyTop = static_cast<int>(r.height * sideF);
    int dyBot = static_cast<int>(r.height * bottomF);

    int x = std::max(0, r.x - dx);
    int y = std::max(0, r.y - dyTop);
    int w = std::min(imgSz.width  - x, r.width  + 2 * dx);
    int h = std::min(imgSz.height - y, r.height + dyTop + dyBot);
    return {x, y, w, h};
}

// ---------------------------------------------------------------------------
// Draw the 21-point MediaPipe skeleton
// ---------------------------------------------------------------------------
static void drawSkeleton(cv::Mat& frame,
                         const std::vector<cv::Point2f>& kp) {
    const cv::Scalar boneColor{0, 200, 80};
    const cv::Scalar jointColor{0, 255, 120};

    for (int i = 0; i < HAND_CONNECTIONS_COUNT; ++i) {
        int a = HAND_CONNECTIONS[i][0], b = HAND_CONNECTIONS[i][1];
        cv::line(frame,
                 cv::Point(static_cast<int>(kp[a].x), static_cast<int>(kp[a].y)),
                 cv::Point(static_cast<int>(kp[b].x), static_cast<int>(kp[b].y)),
                 boneColor, 2, cv::LINE_AA);
    }
    for (int i = 0; i < MP::COUNT; ++i)
        cv::circle(frame,
                   cv::Point(static_cast<int>(kp[i].x), static_cast<int>(kp[i].y)),
                   4, jointColor, cv::FILLED, cv::LINE_AA);
}

// ---------------------------------------------------------------------------
// Draw a progress bar for the current gesture phase timer
// ---------------------------------------------------------------------------
static void drawProgressBar(cv::Mat& frame, float progress, int x, int y,
                            int w, int h, const cv::Scalar& color) {
    cv::rectangle(frame, {x, y, w, h}, {80, 80, 80}, cv::FILLED);
    cv::rectangle(frame, {x, y, static_cast<int>(w * progress), h}, color, cv::FILLED);
    cv::rectangle(frame, {x, y, w, h}, {200, 200, 200}, 1);
}

// ---------------------------------------------------------------------------
// Draw the bottom HUD bar + debug finger-state panel
// ---------------------------------------------------------------------------
static void drawHUD(cv::Mat& frame,
                    const GestureResult& gr,
                    const GestureDebug&  dbg,
                    float confidence,
                    bool  modelLoaded,
                    bool  showDebug) {
    const int barH = 60;
    int barY = frame.rows - barH;
    cv::rectangle(frame, {0, barY, frame.cols, barH}, cv::Scalar(20, 20, 20), cv::FILLED);

    // Phase color and text
    cv::Scalar phaseColor;
    switch (gr.phase) {
        case GesturePhase::IDLE:            phaseColor = {160, 160, 160}; break;
        case GesturePhase::OPEN_HAND:       phaseColor = {80,  220,  80}; break;
        case GesturePhase::THUMB_TUCKED:    phaseColor = {80,  180, 220}; break;
        case GesturePhase::SIGNAL_COMPLETE: phaseColor = {60,   60, 255}; break;
    }

    cv::putText(frame, gr.label,
                {10, barY + 22},
                cv::FONT_HERSHEY_SIMPLEX, 0.60, phaseColor, 2, cv::LINE_AA);

    // Progress bar (shows how long current phase has been held)
    if (gr.phase == GesturePhase::OPEN_HAND || gr.phase == GesturePhase::THUMB_TUCKED)
        drawProgressBar(frame, gr.progress, 10, barY + 32, 200, 14, phaseColor);

    // Confidence and keys
    std::string confStr = modelLoaded
        ? "Conf: " + std::to_string(static_cast<int>(confidence * 100)) + "%"
        : "[Sin modelo]";
    cv::putText(frame, confStr,
                {frame.cols - 130, barY + 22},
                cv::FONT_HERSHEY_SIMPLEX, 0.50,
                modelLoaded ? cv::Scalar(180, 180, 180) : cv::Scalar(60, 60, 255),
                1, cv::LINE_AA);

    cv::putText(frame, "Q:Salir  R:Reset  V:Debug  D:Mascara",
                {frame.cols / 2 - 155, barY + 50},
                cv::FONT_HERSHEY_SIMPLEX, 0.40, {120, 120, 120}, 1, cv::LINE_AA);

    // ── Debug finger-state panel (top-right) ─────────────────────────────────
    if (showDebug) {
        struct FingerInfo { const char* name; bool extended; bool curled; };
        FingerInfo fingers[] = {
            {"IDX", dbg.indexExtended,  !dbg.indexExtended},
            {"MED", dbg.middleExtended, !dbg.middleExtended},
            {"ANL", dbg.ringExtended,   !dbg.ringExtended},
            {"MNQ", dbg.pinkyExtended,  !dbg.pinkyExtended},
        };

        int px = frame.cols - 160, py = 10;
        cv::rectangle(frame, {px - 5, py - 5, 165, 140}, {20, 20, 20}, cv::FILLED);
        cv::putText(frame, "=== DEBUG GESTO ===", {px, py + 12},
                    cv::FONT_HERSHEY_SIMPLEX, 0.38, {200, 200, 200}, 1);

        for (int i = 0; i < 4; ++i) {
            cv::Scalar col = fingers[i].extended ? cv::Scalar{60, 220, 60}
                                                 : cv::Scalar{60, 60, 220};
            std::string s = std::string(fingers[i].name) + ": "
                          + (fingers[i].extended ? "EXTENDIDO" : "DOBLADO  ");
            cv::putText(frame, s, {px, py + 30 + i * 18},
                        cv::FONT_HERSHEY_SIMPLEX, 0.38, col, 1);
        }

        cv::Scalar tCol = dbg.thumbOut    ? cv::Scalar{60, 220, 60}
                        : dbg.thumbTucked ? cv::Scalar{220, 180, 60}
                                          : cv::Scalar{60, 60, 220};
        std::string ts = "PLG: " + std::string(dbg.thumbOut    ? "FUERA"
                                             : dbg.thumbTucked ? "ADENTRO"
                                                               : "NEUTRO ");
        cv::putText(frame, ts, {px, py + 30 + 4 * 18},
                    cv::FONT_HERSHEY_SIMPLEX, 0.38, tCol, 1);

        char palmStr[64];
        std::snprintf(palmStr, sizeof(palmStr),
                      "Palm=%.0fpx ratio=%.2f", dbg.palmSize, dbg.thumbToPalmRatio);
        cv::putText(frame, palmStr, {px, py + 30 + 5 * 18},
                    cv::FONT_HERSHEY_SIMPLEX, 0.35, {160, 160, 160}, 1);

        // Step indicators: circle = completed, empty = pending
        auto stepDot = [&](int sx, int sy, bool done, const char* label) {
            cv::circle(frame, {sx, sy}, 7,
                       done ? cv::Scalar{60, 220, 60} : cv::Scalar{80, 80, 80},
                       cv::FILLED);
            cv::putText(frame, label, {sx + 10, sy + 4},
                        cv::FONT_HERSHEY_SIMPLEX, 0.33, {200, 200, 200}, 1);
        };
        stepDot(px,     py + 120,
                gr.phase != GesturePhase::IDLE, "Abierta");
        stepDot(px + 60, py + 120,
                gr.phase == GesturePhase::THUMB_TUCKED ||
                gr.phase == GesturePhase::SIGNAL_COMPLETE, "Pulgar");
        stepDot(px + 115, py + 120,
                gr.phase == GesturePhase::SIGNAL_COMPLETE, "Cerrada");
    }
}

// ---------------------------------------------------------------------------
int main(int argc, char* argv[]) {
    std::string modelPath = "models/handpose_estimation_mediapipe_2023feb.onnx";
    if (argc > 1) modelPath = argv[1];

    int camIndex = 0;
    if (argc > 2) camIndex = std::atoi(argv[2]);

    std::cout << "=== Signal for Help Detector ===\n";
    std::cout << "Modelo : " << modelPath << "\n";
    std::cout << "Camara : " << camIndex  << "\n\n";
    std::cout << "Controles:\n"
              << "  Q  - Salir\n"
              << "  R  - Reiniciar estado del gesto\n"
              << "  V  - Activar/desactivar modo verbose (consola)\n"
              << "  D  - Activar/desactivar mascara de piel\n"
              << "  G  - Activar/desactivar panel debug\n\n";

    // ── Subsystems ──────────────────────────────────────────────────────────
    // Confidence threshold lowered to 0.35 — real webcam images have more
    // noise and lighting variation than the model's training data.
    HandPoseEstimator poseEstimator(modelPath, /*confThreshold=*/0.35f);
    SkinDetector      skinDetector;
    GestureDetector   gestureDetector(/*openHold=*/0.8f,
                                     /*thumbHold=*/0.5f,
                                     /*reset=*/1.5f);
    AlertSystem       alertSystem;

    if (!poseEstimator.isLoaded()) {
        std::cerr << "\n[ADVERTENCIA] Modelo ONNX no encontrado.\n"
                  << "Ejecute: bash download_models.sh\n"
                  << "Continuando solo con deteccion de piel (menos preciso).\n\n";
    }

    // ── Camera ──────────────────────────────────────────────────────────────
    cv::VideoCapture cap(camIndex, cv::CAP_V4L2);
    if (!cap.isOpened()) { cap.open(camIndex); }
    if (!cap.isOpened()) {
        std::cerr << "[ERROR] No se puede abrir la camara " << camIndex << "\n";
        return 1;
    }
    cap.set(cv::CAP_PROP_FRAME_WIDTH,  640);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
    cap.set(cv::CAP_PROP_FPS, 30);

    std::cout << "Resolucion: "
              << cap.get(cv::CAP_PROP_FRAME_WIDTH) << "×"
              << cap.get(cv::CAP_PROP_FRAME_HEIGHT) << "\n\n";

    cv::namedWindow("Signal for Help", cv::WINDOW_NORMAL);
    cv::resizeWindow("Signal for Help", 800, 600);

    bool showDebugPanel = false;
    bool showMask       = false;
    bool verbose        = false;

    cv::Mat frame, display;
    GestureDebug lastDebug{};

    while (true) {
        cap >> frame;
        if (frame.empty()) { std::cerr << "[ERROR] Frame vacio.\n"; break; }

        cv::flip(frame, frame, 1);  // mirror — more natural for the user
        frame.copyTo(display);

        // ── Step 1: Skin-color candidate detection ───────────────────────────
        auto candidates = skinDetector.detect(frame);

        // ── Step 2: Hand pose estimation on each candidate ───────────────────
        HandPoseResult bestPose;
        bool           handConfirmed = false;

        for (auto& region : candidates) {
            // Expand ROI asymmetrically: more space below for the wrist
            cv::Rect roi = expandROI(region.bbox, frame.size(),
                                     /*sideF=*/0.25f, /*bottomF=*/0.40f);
            if (roi.width < 20 || roi.height < 20) continue;

            // Draw candidate bbox in gray
            cv::rectangle(display, region.bbox, {140, 140, 140}, 1);

            if (poseEstimator.isLoaded()) {
                cv::Mat crop = frame(roi);
                HandPoseResult pose = poseEstimator.estimate(crop, roi, frame.size());

                if (pose.valid && pose.confidence > 0.35f) {
                    if (!handConfirmed || pose.confidence > bestPose.confidence) {
                        bestPose      = pose;
                        handConfirmed = true;
                    }
                }
            } else {
                // Fallback: accept skin region as hand (no landmark check)
                bestPose.valid      = true;
                bestPose.confidence = 1.0f;
                bestPose.landmarks.clear();
                handConfirmed = true;
            }
        }

        // ── Step 3: Draw skeleton if hand detected ───────────────────────────
        if (handConfirmed && !bestPose.landmarks.empty()) {
            drawSkeleton(display, bestPose.landmarks);
            // Bounding box around all landmarks
            std::vector<cv::Point> pts;
            for (auto& lm : bestPose.landmarks)
                pts.emplace_back(static_cast<int>(lm.x), static_cast<int>(lm.y));
            cv::Rect lmBox = expandROI(cv::boundingRect(pts), frame.size(), 0.05f, 0.05f);
            cv::rectangle(display, lmBox, {0, 220, 80}, 2);
        }

        // ── Step 4: Gesture state machine ────────────────────────────────────
        GestureResult gr = gestureDetector.update(
            handConfirmed ? bestPose.landmarks : std::vector<cv::Point2f>{},
            handConfirmed ? bestPose.confidence : 0.0f,
            handConfirmed);

        if (handConfirmed && !bestPose.landmarks.empty())
            lastDebug = gestureDetector.getDebug(bestPose.landmarks);

        // ── Step 5: Alert ────────────────────────────────────────────────────
        if (gr.alertFired)
            alertSystem.trigger();
        if (alertSystem.isActive())
            alertSystem.render(display);

        // ── Step 6: HUD + debug panel ────────────────────────────────────────
        drawHUD(display, gr, lastDebug,
                handConfirmed ? bestPose.confidence : 0.0f,
                poseEstimator.isLoaded(), showDebugPanel);

        // ── Step 7: Skin mask overlay ─────────────────────────────────────────
        if (showMask) {
            cv::Mat mask = skinDetector.getLastMask();
            if (!mask.empty()) {
                cv::Mat cm;
                cv::cvtColor(mask, cm, cv::COLOR_GRAY2BGR);
                cv::resize(cm, cm, {frame.cols / 4, frame.rows / 4});
                cv::addWeighted(display(cv::Rect(0, 0, cm.cols, cm.rows)),
                                0.3, cm, 0.7, 0,
                                display(cv::Rect(0, 0, cm.cols, cm.rows)));
            }
        }

        cv::imshow("Signal for Help", display);

        int key = cv::waitKey(1);
        if (key == 'q' || key == 27) break;
        if (key == 'r') {
            gestureDetector.reset();
            alertSystem.deactivate();
            std::cout << "[Main] Reinicio manual.\n";
        }
        if (key == 'g') {
            showDebugPanel = !showDebugPanel;
            std::cout << "[Main] Debug panel: " << (showDebugPanel ? "ON" : "OFF") << "\n";
        }
        if (key == 'd') {
            showMask = !showMask;
            std::cout << "[Main] Mascara de piel: " << (showMask ? "ON" : "OFF") << "\n";
        }
        if (key == 'v') {
            verbose = !verbose;
            gestureDetector.setVerbose(verbose);
            std::cout << "[Main] Modo verbose: " << (verbose ? "ON" : "OFF") << "\n";
        }
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}
