#include <opencv2/opencv.hpp>
#include <cstdio>
#include <iostream>
#include <string>

#include "skin_detector.hpp"
#include "hand_pose.hpp"
#include "gesture_detector.hpp"
#include "alert_system.hpp"
#include "serial_comm.hpp"

using namespace cv;
using namespace std;

static constexpr int   GESTURES_TO_ALERT  = 3;
static constexpr int   TRACK_MAX_AGE      = 30;   // ~1 s at 30 fps
static constexpr float CONF_THRESHOLD     = 0.32f;
static constexpr float COUNTER_RESET_SEC  = 10.0f;

// ---------------------------------------------------------------------------
static Size detectScreenSize() {
    FILE* f = popen("xrandr 2>/dev/null | awk '/\\*/{print $1; exit}'", "r");
    if (f) {
        char buf[32]{};
        bool ok = (fgets(buf, sizeof(buf), f) != nullptr);
        pclose(f);
        if (ok) {
            int w = 0, h = 0;
            if (sscanf(buf, "%dx%d", &w, &h) == 2 && w > 100 && h > 100)
                return {w, h};
        }
    }
    return {1920, 1080};
}

// ---------------------------------------------------------------------------
// Asymmetric ROI padding: more bottom space keeps the wrist landmark inside
// the crop (wrist = scale reference for all gesture geometry).
// ---------------------------------------------------------------------------
static Rect expandROI(const Rect& r, const Size& imgSz,
                      float sideF = 0.25f, float bottomF = 0.40f) {
    int dx    = static_cast<int>(r.width  * sideF);
    int dyTop = static_cast<int>(r.height * sideF);
    int dyBot = static_cast<int>(r.height * bottomF);
    int x = max(0, r.x - dx);
    int y = max(0, r.y - dyTop);
    int w = min(imgSz.width  - x, r.width  + 2 * dx);
    int h = min(imgSz.height - y, r.height + dyTop + dyBot);
    return {x, y, w, h};
}

// ---------------------------------------------------------------------------
static void drawSkeleton(Mat& frame, const vector<Point2f>& kp) {
    const Scalar bone{0, 200, 80};
    const Scalar joint{0, 255, 120};
    for (int i = 0; i < HAND_CONNECTIONS_COUNT; ++i) {
        int a = HAND_CONNECTIONS[i][0], b = HAND_CONNECTIONS[i][1];
        line(frame,
             {static_cast<int>(kp[a].x), static_cast<int>(kp[a].y)},
             {static_cast<int>(kp[b].x), static_cast<int>(kp[b].y)},
             bone, 2, LINE_AA);
    }
    for (int i = 0; i < MP::COUNT; ++i)
        circle(frame,
               {static_cast<int>(kp[i].x), static_cast<int>(kp[i].y)},
               4, joint, FILLED, LINE_AA);
}

static void drawProgressBar(Mat& frame, float progress, int x, int y,
                            int w, int h, const Scalar& color) {
    rectangle(frame, {x, y, w, h}, Scalar(80, 80, 80), FILLED);
    rectangle(frame, {x, y, static_cast<int>(w * progress), h}, color, FILLED);
    rectangle(frame, {x, y, w, h}, Scalar(200, 200, 200), 1);
}

// ---------------------------------------------------------------------------
static void drawHUD(Mat& frame,
                    const GestureResult& gr,
                    const GestureDebug&  dbg,
                    float confidence,
                    bool  modelLoaded,
                    bool  showDebug,
                    int   gestureCount,
                    bool  serialOpen,
                    bool  tracking) {
    const int barH = 60;
    int barY = frame.rows - barH;
    rectangle(frame, {0, barY, frame.cols, barH}, Scalar(20, 20, 20), FILLED);

    Scalar phaseColor;
    switch (gr.phase) {
        case GesturePhase::IDLE:            phaseColor = {160, 160, 160}; break;
        case GesturePhase::OPEN_HAND:       phaseColor = {80,  220,  80}; break;
        case GesturePhase::THUMB_TUCKED:    phaseColor = {80,  180, 220}; break;
        case GesturePhase::SIGNAL_COMPLETE: phaseColor = {60,   60, 255}; break;
    }

    string label = gr.label;
    if (gr.phase == GesturePhase::SIGNAL_COMPLETE && gestureCount > 0)
        label = "Gesto " + to_string(gestureCount)
              + "/" + to_string(GESTURES_TO_ALERT) + " completado";

    putText(frame, label, {10, barY + 22},
            FONT_HERSHEY_SIMPLEX, 0.60, phaseColor, 2, LINE_AA);

    if (gr.phase == GesturePhase::OPEN_HAND || gr.phase == GesturePhase::THUMB_TUCKED)
        drawProgressBar(frame, gr.progress, 10, barY + 32, 200, 14, phaseColor);

    if (gestureCount > 0)
        putText(frame,
                "Gestos: " + to_string(gestureCount) + "/" + to_string(GESTURES_TO_ALERT),
                {10, barY + 50},
                FONT_HERSHEY_SIMPLEX, 0.42, Scalar(60, 220, 255), 1, LINE_AA);

    putText(frame, tracking ? "TRACK" : "SCAN ",
            {frame.cols - 200, barY + 50}, FONT_HERSHEY_SIMPLEX, 0.38,
            tracking ? Scalar(60,220,255) : Scalar(200,140,60), 1, LINE_AA);

    putText(frame, serialOpen ? "COM:OK" : "COM:--",
            {frame.cols - 130, barY + 50}, FONT_HERSHEY_SIMPLEX, 0.38,
            serialOpen ? Scalar(60,220,60) : Scalar(80,80,80), 1, LINE_AA);

    putText(frame,
            modelLoaded ? "Conf: " + to_string(static_cast<int>(confidence * 100)) + "%"
                        : "[Sin modelo]",
            {frame.cols - 130, barY + 22}, FONT_HERSHEY_SIMPLEX, 0.50,
            modelLoaded ? Scalar(180,180,180) : Scalar(60,60,255), 1, LINE_AA);

    putText(frame, "Q:Salir  R:Reset  V:Verbose  G:Panel",
            {frame.cols / 2 - 140, barY + 50},
            FONT_HERSHEY_SIMPLEX, 0.40, Scalar(120, 120, 120), 1, LINE_AA);

    if (!showDebug) return;

    struct FI { const char* name; bool ext; };
    FI fi[] = {{"IDX",dbg.indexExtended},{"MED",dbg.middleExtended},
               {"ANL",dbg.ringExtended}, {"MNQ",dbg.pinkyExtended}};
    int px = frame.cols - 160, py = 10;
    rectangle(frame, {px-5, py-5, 165, 140}, Scalar(20,20,20), FILLED);
    putText(frame, "=== DEBUG GESTO ===", {px, py+12},
            FONT_HERSHEY_SIMPLEX, 0.38, Scalar(200,200,200), 1);
    for (int i = 0; i < 4; ++i) {
        putText(frame,
                string(fi[i].name) + ": " + (fi[i].ext ? "EXTENDIDO" : "DOBLADO  "),
                {px, py+30+i*18}, FONT_HERSHEY_SIMPLEX, 0.38,
                fi[i].ext ? Scalar{60,220,60} : Scalar{60,60,220}, 1);
    }
    putText(frame,
            "PLG: " + string(dbg.thumbOut ? "FUERA" : dbg.thumbTucked ? "ADENTRO" : "NEUTRO "),
            {px, py+30+4*18}, FONT_HERSHEY_SIMPLEX, 0.38,
            dbg.thumbOut ? Scalar{60,220,60} : dbg.thumbTucked ? Scalar{220,180,60}
                                                                : Scalar{60,60,220}, 1);
    char ps[64];
    snprintf(ps, sizeof(ps), "Palm=%.0fpx ratio=%.2f", dbg.palmSize, dbg.thumbToPalmRatio);
    putText(frame, ps, {px, py+30+5*18}, FONT_HERSHEY_SIMPLEX, 0.35, Scalar(160,160,160), 1);

    auto dot = [&](int sx, int sy, bool done, const char* lbl) {
        circle(frame, {sx,sy}, 7,
               done ? Scalar{60,220,60} : Scalar{80,80,80}, FILLED);
        putText(frame, lbl, {sx+10,sy+4}, FONT_HERSHEY_SIMPLEX, 0.33, Scalar(200,200,200), 1);
    };
    dot(px,     py+120, gr.phase != GesturePhase::IDLE,            "Abierta");
    dot(px+60,  py+120, gr.phase == GesturePhase::THUMB_TUCKED ||
                        gr.phase == GesturePhase::SIGNAL_COMPLETE,  "Pulgar");
    dot(px+115, py+120, gr.phase == GesturePhase::SIGNAL_COMPLETE, "Cerrada");
}

// ---------------------------------------------------------------------------
// TiledScanner: divides the frame into overlapping crops whose size is
// ~55% of the shorter frame dimension (~264 px for 640×480).
// At that crop size a hand fills ≈35-50% of the 224×224 model input —
// the "sweet spot" where the MediaPipe landmark model is most reliable.
//
// Tiles cycle across calls so only `tilesPerCall` inferences run per frame,
// spreading the cost over multiple frames while still scanning everywhere.
// ---------------------------------------------------------------------------
struct TiledScanner {
    vector<Rect> tiles;
    int idx{0};

    void init(Size sz) {
        tiles.clear();
        idx = 0;
        // Crop ≈ 55 % of the short side: large enough to include the whole hand,
        // small enough that the hand fills a meaningful fraction of 224×224.
        int crop = static_cast<int>(min(sz.width, sz.height) * 0.55f);
        crop     = max(crop, 160);  // never smaller than 160 px
        // 3 columns × 2 rows = 6 tiles, uniformly distributed across the frame
        for (int r = 0; r < 2; ++r) {
            for (int c = 0; c < 3; ++c) {
                int x = (sz.width  > crop) ? c * (sz.width  - crop) / 2 : 0;
                int y = (sz.height > crop) ? r * (sz.height - crop) / 1 : 0;
                tiles.push_back({x, y, crop, crop});
            }
        }
    }

    // Try up to `n` tiles this call. Returns the best valid result found.
    HandPoseResult scan(HandPoseEstimator& est,
                        const Mat& frame,
                        const Rect& fROI,
                        int n) {
        if (tiles.empty()) init(frame.size());
        HandPoseResult best;
        for (int t = 0; t < n; ++t) {
            Rect tile = tiles[idx % (int)tiles.size()] & fROI;
            ++idx;
            if (tile.width < 60 || tile.height < 60) continue;
            HandPoseResult pose = est.estimate(frame(tile), tile, frame.size());
            if (pose.valid && pose.confidence > CONF_THRESHOLD)
                if (!best.valid || pose.confidence > best.confidence)
                    best = pose;
        }
        return best;
    }

    void reset() { idx = 0; }
};

// ---------------------------------------------------------------------------
int main(int argc, char* argv[]) {
    string modelPath = "models/handpose_estimation_mediapipe_2023feb.onnx";
    if (argc > 1) modelPath = argv[1];
    int camIndex = 0;
    if (argc > 2) camIndex = atoi(argv[2]);

    string serialPort;
    if (argc > 3) { serialPort = argv[3]; }
#ifdef _WIN32
    else { serialPort = "COM3"; }
#else
    else { serialPort = "/dev/ttyACM0"; }
#endif
    int serialBaud = 9600;
    if (argc > 4) serialBaud = atoi(argv[4]);

    const Size screenSize = detectScreenSize();

    cout << "=== Signal for Help Detector ===\n"
         << "Modelo       : " << modelPath  << "\n"
         << "Camara       : " << camIndex   << "\n"
         << "Puerto serial: " << serialPort << " @ " << serialBaud << " bps\n"
         << "Pantalla     : " << screenSize.width << "x" << screenSize.height << "\n\n"
         << "Controles:\n"
         << "  Q  - Salir\n"
         << "  R  - Reiniciar estado del gesto\n"
         << "  V  - Activar/desactivar modo verbose\n"
         << "  G  - Activar/desactivar panel debug\n\n";

    HandPoseEstimator poseEstimator(modelPath, CONF_THRESHOLD);
    SkinDetector      skinDetector;
    GestureDetector   gestureDetector(/*openHold=*/1.0f,
                                      /*thumbHold=*/0.7f,
                                      /*reset=*/1.5f);
    AlertSystem       alertSystem;

    SerialComm serial;
    serial.open(serialPort, serialBaud);
    if (!serial.isOpen())
        cout << "[Serial] Puerto no disponible — continuando sin serial.\n\n";

    int gestureCount = 0;
    using Clock = chrono::steady_clock;
    auto lastGestureTime = Clock::now();

    if (!poseEstimator.isLoaded())
        cerr << "\n[ADVERTENCIA] Modelo ONNX no encontrado. Ejecute: bash download_models.sh\n\n";

    // ── Camera ──────────────────────────────────────────────────────────────
    VideoCapture cap(camIndex, CAP_V4L2);
    if (!cap.isOpened()) cap.open(camIndex);
    if (!cap.isOpened()) {
        cerr << "[ERROR] No se puede abrir la camara " << camIndex << "\n";
        return 1;
    }
    cap.set(CAP_PROP_FOURCC, VideoWriter::fourcc('M','J','P','G'));
    cap.set(CAP_PROP_FRAME_WIDTH,  640);
    cap.set(CAP_PROP_FRAME_HEIGHT, 480);
    cap.set(CAP_PROP_FPS, 30);
    cout << "Captura: " << cap.get(CAP_PROP_FRAME_WIDTH)
         << "x"         << cap.get(CAP_PROP_FRAME_HEIGHT) << "\n\n";

    namedWindow("Signal for Help", WINDOW_NORMAL);
    setWindowProperty("Signal for Help", WND_PROP_FULLSCREEN, WINDOW_FULLSCREEN);

    bool showDebugPanel = false;
    bool verbose        = false;

    // ── Detection state ──────────────────────────────────────────────────────
    // SCAN mode : skin detector → candidate ROIs → pose estimator on each crop.
    //             Fallback tiled scan (260 px crops) when skin detector fails.
    // TRACK mode: pose estimator runs directly on last known expanded bbox.
    //             Skips skin detector entirely — one inference per frame.
    Rect         trackedROI;
    int          trackAge = TRACK_MAX_AGE + 1;
    TiledScanner scanner;

    Mat frame, display;
    GestureDebug lastDebug{};

    while (true) {
        cap >> frame;
        if (frame.empty()) { cerr << "[ERROR] Frame vacio.\n"; break; }

        flip(frame, frame, 1);
        frame.copyTo(display);

        const Rect fROI{0, 0, frame.cols, frame.rows};
        bool isTracking = (trackAge <= TRACK_MAX_AGE) && (trackedROI.area() > 0);

        HandPoseResult bestPose{};
        bool           handConfirmed = false;

        if (isTracking) {
            // ── Track mode ────────────────────────────────────────────────────
            Rect tROI = expandROI(trackedROI, frame.size(), 0.35f, 0.50f);
            tROI &= fROI;
            rectangle(display, tROI, Scalar(160, 200, 0), 1);

            if (poseEstimator.isLoaded() && tROI.width >= 20 && tROI.height >= 20) {
                HandPoseResult pose = poseEstimator.estimate(frame(tROI), tROI, frame.size());
                if (pose.valid && pose.confidence > CONF_THRESHOLD) {
                    bestPose = pose; handConfirmed = true;
                }
            }
        } else {
            // ── Scan mode ─────────────────────────────────────────────────────
            auto regions = skinDetector.detect(frame);

            for (auto& r : regions) {
                rectangle(display, r.bbox, Scalar(140, 140, 140), 1);

                Rect roi = expandROI(r.bbox, frame.size(), 0.25f, 0.40f);
                roi &= fROI;
                if (roi.width < 20 || roi.height < 20) continue;

                if (poseEstimator.isLoaded()) {
                    HandPoseResult pose = poseEstimator.estimate(frame(roi), roi, frame.size());
                    if (pose.valid && pose.confidence > CONF_THRESHOLD) {
                        if (!handConfirmed || pose.confidence > bestPose.confidence) {
                            bestPose = pose; handConfirmed = true;
                        }
                    }
                } else if (!handConfirmed) {
                    bestPose.valid = true; bestPose.confidence = 1.0f; handConfirmed = true;
                }
            }

            // ── Fallback: tiled scan when skin detector found nothing ──────────
            // Tries 3 crops of ~260 px per frame, cycling through 6 tiles.
            // At that crop size the hand fills ~35-50 % of the 224×224 input
            // (vs ~15 % for a full 640×480 frame — the old broken approach).
            if (!handConfirmed && poseEstimator.isLoaded()) {
                if (scanner.tiles.empty()) scanner.init(frame.size());
                HandPoseResult fb = scanner.scan(poseEstimator, frame, fROI, 3);
                if (fb.valid && fb.confidence > CONF_THRESHOLD) {
                    bestPose = fb; handConfirmed = true;
                    // Draw the active tile in orange so the user can see it
                    int ti = (scanner.idx - 1 + (int)scanner.tiles.size())
                             % (int)scanner.tiles.size();
                    rectangle(display, scanner.tiles[ti], Scalar(255, 100, 0), 1);
                }
            }
        }

        // ── Update tracking state ────────────────────────────────────────────
        if (handConfirmed && !bestPose.landmarks.empty()) {
            vector<Point> pts;
            pts.reserve(bestPose.landmarks.size());
            for (auto& lm : bestPose.landmarks)
                pts.emplace_back(static_cast<int>(lm.x), static_cast<int>(lm.y));
            trackedROI = boundingRect(pts);
            trackAge   = 0;
        } else {
            ++trackAge;
            if (trackAge > TRACK_MAX_AGE)
                trackedROI = Rect{};
        }

        // Reset tile cycling when a hand is confirmed so the next scan starts
        // from tile 0 (avoids starting mid-cycle after tracking is lost).
        if (handConfirmed) scanner.reset();

        // ── Draw skeleton ────────────────────────────────────────────────────
        if (handConfirmed && !bestPose.landmarks.empty()) {
            drawSkeleton(display, bestPose.landmarks);
            vector<Point> pts;
            pts.reserve(bestPose.landmarks.size());
            for (auto& lm : bestPose.landmarks)
                pts.emplace_back(static_cast<int>(lm.x), static_cast<int>(lm.y));
            Rect lmBox = expandROI(boundingRect(pts), frame.size(), 0.05f, 0.05f);
            rectangle(display, lmBox, Scalar(0, 220, 80), 2);
        }

        // ── Gesture state machine ────────────────────────────────────────────
        GestureResult gr = gestureDetector.update(
            handConfirmed ? bestPose.landmarks : vector<Point2f>{},
            handConfirmed ? bestPose.confidence : 0.0f,
            handConfirmed);

        if (handConfirmed && !bestPose.landmarks.empty())
            lastDebug = gestureDetector.getDebug(bestPose.landmarks);

        // ── Gesture counter + alert + serial ─────────────────────────────────
        if (gr.alertFired) {
            ++gestureCount;
            lastGestureTime = Clock::now();
            cout << "[Counter] Gesto: " << gestureCount << "/" << GESTURES_TO_ALERT << "\n";
            if (gestureCount >= GESTURES_TO_ALERT) {
                gestureCount = 0;
                alertSystem.trigger();
                if (serial.isOpen()) {
                    serial.send("SFH\n");
                    cout << "[Serial] 'SFH' enviado a " << serial.portName() << "\n";
                }
            }
        }

        if (gestureCount > 0) {
            float elapsed = chrono::duration<float>(Clock::now() - lastGestureTime).count();
            if (elapsed >= COUNTER_RESET_SEC) {
                cout << "[Counter] Tiempo agotado — reiniciado\n";
                gestureCount = 0;
            }
        }

        if (alertSystem.isActive())
            alertSystem.render(display);

        drawHUD(display, gr, lastDebug,
                handConfirmed ? bestPose.confidence : 0.0f,
                poseEstimator.isLoaded(), showDebugPanel,
                gestureCount, serial.isOpen(), isTracking);

        Mat displayFull;
        resize(display, displayFull, screenSize, 0, 0, INTER_LINEAR);
        imshow("Signal for Help", displayFull);

        int key = waitKey(1);
        if (key == 'q' || key == 27) break;
        if (key == 'r') {
            gestureDetector.reset();
            alertSystem.deactivate();
            gestureCount = 0;
            trackedROI   = Rect{};
            trackAge     = TRACK_MAX_AGE + 1;
            scanner.reset();
            cout << "[Main] Reinicio manual.\n";
        }
        if (key == 'g') {
            showDebugPanel = !showDebugPanel;
            cout << "[Main] Debug panel: " << (showDebugPanel ? "ON" : "OFF") << "\n";
        }
        if (key == 'v') {
            verbose = !verbose;
            gestureDetector.setVerbose(verbose);
            cout << "[Main] Verbose: " << (verbose ? "ON" : "OFF") << "\n";
        }
    }

    cap.release();
    destroyAllWindows();
    return 0;
}
