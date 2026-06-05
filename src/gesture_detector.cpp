#include "gesture_detector.hpp"
#include <cmath>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <ctime>

using namespace cv;
using namespace std;

// ============================================================================
// MediaPipe 21-point hand landmarks (visual reference):
//
//                  8   12  16  20        <- FINGERTIPS
//                  |    |   |   |
//                  7   11  15  19        <- DIP joints
//                  |    |   |   |
//                  6   10  14  18        <- PIP joints
//                  |    |   |   |
//                  5    9  13  17        <- MCP joints (knuckles)
//              4   |    |   |   |
//             (thumb tip)
//          3    \__|    |   |   |
//          2        \   |   |   |
//          1    (thumb CMC)
//               0   <-- WRIST
//
// Distance from WRIST to MIDDLE_MCP = "palm length" (our scale reference).
// ============================================================================

GestureDetector::GestureDetector(float openHoldSec,
                                 float thumbHoldSec,
                                 float resetSec)
    : openHoldSec_(openHoldSec),
      thumbHoldSec_(thumbHoldSec),
      resetSec_(resetSec)
{}

void GestureDetector::reset() {
    phase_             = GesturePhase::IDLE;
    alertAlreadyFired_ = false;
    initialized_       = false;
    openFailCount_     = 0;
    thumbFailCount_    = 0;
}

// ============================================================================
GestureResult GestureDetector::update(const vector<Point2f>& kp,
                                      float confidence, bool handPresent) {
    TP now = Clock::now();

    if (!initialized_) {
        phaseStart_   = now;
        lastHandSeen_ = now;
        initialized_  = true;
    }

    if (handPresent && confidence > 0.0f)
        lastHandSeen_ = now;

    float noHandSec = chrono::duration<float>(now - lastHandSeen_).count();
    if (noHandSec > resetSec_ && phase_ != GesturePhase::IDLE) {
        if (verbose_) cout << "[Gesture] Hand lost — resetting to IDLE\n";
        reset();
        initialized_  = true;
        phaseStart_   = now;
        lastHandSeen_ = now;
    }

    auto elapsed = [&](TP start) {
        return chrono::duration<float>(now - start).count();
    };

    float progress = 0.0f;
    GestureResult result{phase_, false, "", 0.0f};

    if (!handPresent || static_cast<int>(kp.size()) < 21) {
        result.label = "Sin mano";
        return result;
    }

    if (verbose_) {
        GestureDebug dbg = getDebug(kp);
        cout << fixed << setprecision(2)
             << "[Gesto] palmLen=" << dbg.palmSize
             << " thumbRatio=" << dbg.thumbToPalmRatio
             << " idx=" << dbg.indexExtended
             << " mid=" << dbg.middleExtended
             << " rng=" << dbg.ringExtended
             << " pnk=" << dbg.pinkyExtended
             << " thumbOut=" << dbg.thumbOut
             << " thumbTucked=" << dbg.thumbTucked
             << " fingersClosed=" << dbg.fingersClosed
             << "\n";
    }

    switch (phase_) {

        case GesturePhase::IDLE: {
            // 3e: Use minExtended=3 when entering from IDLE (more tolerant for
            //     slightly-rotated hands), strict=4 for continuity checks.
            if (checkOpenHand(kp, 3)) {
                phase_         = GesturePhase::OPEN_HAND;
                phaseStart_    = now;
                openFailCount_ = 0;
                if (verbose_) cout << "[Gesture] → OPEN_HAND\n";
            } else if (fastMode_ && checkThumbTucked(kp)) {
                phase_          = GesturePhase::THUMB_TUCKED;
                phaseStart_     = now;
                thumbFailCount_ = 0;
                if (verbose_) cout << "[Gesture] → THUMB_TUCKED (modo rapido)\n";
            }
            result.label = fastMode_ ? "Listo — haz paso 2 directo" : "Esperando gesto...";
            break;
        }

        case GesturePhase::OPEN_HAND: {
            if (fastMode_ && checkThumbTucked(kp)) {
                phase_          = GesturePhase::THUMB_TUCKED;
                phaseStart_     = now;
                thumbFailCount_ = 0;
                if (verbose_) cout << "[Gesture] → THUMB_TUCKED (modo rapido)\n";
                result.label = "Pulgar adentro (paso 2/3)";
                break;
            }

            if (!checkOpenHand(kp)) {
                ++openFailCount_;
                if (openFailCount_ > MAX_FAIL_FRAMES) {
                    reset(); initialized_ = true; phaseStart_ = now; lastHandSeen_ = now;
                    result.label = "Esperando gesto...";
                    break;
                }
            } else {
                openFailCount_ = 0;
            }

            float held = elapsed(phaseStart_);
            progress = min(1.0f, held / openHoldSec_);

            if (held >= openHoldSec_ && checkThumbTucked(kp)) {
                phase_          = GesturePhase::THUMB_TUCKED;
                phaseStart_     = now;
                thumbFailCount_ = 0;
                if (verbose_) cout << "[Gesture] → THUMB_TUCKED\n";
            }
            result.label = "Mano abierta (paso 1/3)";
            break;
        }

        case GesturePhase::THUMB_TUCKED: {
            if (!checkThumbTucked(kp)) {
                ++thumbFailCount_;
                if (thumbFailCount_ > MAX_FAIL_FRAMES) {
                    if (checkOpenHand(kp)) {
                        phase_         = GesturePhase::OPEN_HAND;
                        phaseStart_    = now;
                        openFailCount_ = 0;
                        if (verbose_) cout << "[Gesture] → OPEN_HAND (pulgar extendido de nuevo)\n";
                    } else {
                        reset(); initialized_ = true; phaseStart_ = now; lastHandSeen_ = now;
                    }
                    result.label = "Pulgar adentro (paso 2/3)";
                    break;
                }
            } else {
                thumbFailCount_ = 0;
            }

            float held = elapsed(phaseStart_);
            progress = min(1.0f, held / thumbHoldSec_);

            if (held >= thumbHoldSec_ && checkFingersClosed(kp)) {
                phase_     = GesturePhase::SIGNAL_COMPLETE;
                phaseStart_= now;

                if (!alertAlreadyFired_) {
                    alertAlreadyFired_ = true;
                    result.alertFired  = true;
                    fastMode_ = true;

                    auto t  = time(nullptr);
                    auto tm = *localtime(&t);
                    ostringstream oss;
                    oss << put_time(&tm, "%Y-%m-%d %H:%M:%S");
                    cout << "\n*** SIGNAL FOR HELP DETECTADO a las " << oss.str() << " ***\n\n";
                }
                if (verbose_) cout << "[Gesture] → SIGNAL_COMPLETE\n";
            }
            result.label = "Pulgar adentro (paso 2/3)";
            break;
        }

        case GesturePhase::SIGNAL_COMPLETE: {
            progress = 1.0f;
            bool released = fastMode_
                ? !checkFingersClosed(kp)
                : (!checkFingersClosed(kp) && checkOpenHand(kp));
            if (released) {
                reset(); initialized_ = true; phaseStart_ = now; lastHandSeen_ = now;
                if (verbose_) cout << "[Gesture] → IDLE (gesto liberado)\n";
            }
            result.label = "Gesto completo! (paso 3/3)";
            break;
        }
    }

    result.phase    = phase_;
    result.progress = progress;
    return result;
}

// ============================================================================
// Geometry helpers
// ============================================================================

float GestureDetector::dist(Point2f a, Point2f b) const {
    float dx = a.x - b.x, dy = a.y - b.y;
    return sqrt(dx * dx + dy * dy);
}

float GestureDetector::palmLen(const vector<Point2f>& kp) const {
    return dist(kp[WRIST], kp[MIDDLE_MCP]);
}

Point2f GestureDetector::palmCenter(const vector<Point2f>& kp) const {
    return (kp[INDEX_MCP] + kp[MIDDLE_MCP] + kp[RING_MCP] + kp[PINKY_MCP]) * 0.25f;
}

bool GestureDetector::fingerExtended(const vector<Point2f>& kp,
                                     int tip, int mcp) const {
    return dist(kp[tip], kp[WRIST]) > dist(kp[mcp], kp[WRIST]) * 1.20f;
}

bool GestureDetector::fingerCurled(const vector<Point2f>& kp,
                                   int tip, int mcp) const {
    return dist(kp[tip], kp[WRIST]) < dist(kp[mcp], kp[WRIST]) * 1.10f;
}

// ============================================================================
// Gesture phase checks
// ============================================================================

bool GestureDetector::checkOpenHand(const vector<Point2f>& kp, int minExtended) const {
    int extCount = 0;
    if (fingerExtended(kp, INDEX_TIP,  INDEX_MCP))  ++extCount;
    if (fingerExtended(kp, MIDDLE_TIP, MIDDLE_MCP)) ++extCount;
    if (fingerExtended(kp, RING_TIP,   RING_MCP))   ++extCount;
    if (fingerExtended(kp, PINKY_TIP,  PINKY_MCP))  ++extCount;

    float   pl     = palmLen(kp);
    Point2f center = palmCenter(kp);
    bool    thumbOut = dist(kp[THUMB_TIP], center) > pl * 0.88f;

    return extCount >= minExtended && thumbOut;
}

bool GestureDetector::checkThumbTucked(const vector<Point2f>& kp) const {
    float   pl     = palmLen(kp);
    Point2f center = palmCenter(kp);
    return dist(kp[THUMB_TIP], center) < pl * 0.68f;
}

bool GestureDetector::checkFingersClosed(const vector<Point2f>& kp) const {
    int curledCount = 0;
    if (fingerCurled(kp, INDEX_TIP,  INDEX_MCP))  ++curledCount;
    if (fingerCurled(kp, MIDDLE_TIP, MIDDLE_MCP)) ++curledCount;
    if (fingerCurled(kp, RING_TIP,   RING_MCP))   ++curledCount;
    if (fingerCurled(kp, PINKY_TIP,  PINKY_MCP))  ++curledCount;
    return curledCount >= 3;
}

// ============================================================================
GestureDebug GestureDetector::getDebug(const vector<Point2f>& kp) const {
    GestureDebug d{};
    if (static_cast<int>(kp.size()) < 21) return d;

    d.palmSize         = palmLen(kp);
    Point2f center     = palmCenter(kp);
    d.thumbToPalmRatio = (d.palmSize > 0.0f)
                             ? dist(kp[THUMB_TIP], center) / d.palmSize
                             : 0.0f;

    d.indexExtended  = fingerExtended(kp, INDEX_TIP,  INDEX_MCP);
    d.middleExtended = fingerExtended(kp, MIDDLE_TIP, MIDDLE_MCP);
    d.ringExtended   = fingerExtended(kp, RING_TIP,   RING_MCP);
    d.pinkyExtended  = fingerExtended(kp, PINKY_TIP,  PINKY_MCP);
    d.thumbOut       = dist(kp[THUMB_TIP], center) > d.palmSize * 0.80f;
    d.thumbTucked    = dist(kp[THUMB_TIP], center) < d.palmSize * 0.85f;
    d.fingersClosed  = checkFingersClosed(kp);
    return d;
}
