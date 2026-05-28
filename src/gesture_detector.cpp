#include "gesture_detector.hpp"
#include <cmath>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <ctime>

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
// Main update — call once per frame
// ============================================================================
GestureResult GestureDetector::update(const std::vector<cv::Point2f>& kp,
                                      float confidence, bool handPresent) {
    TP now = Clock::now();

    if (!initialized_) {
        phaseStart_   = now;
        lastHandSeen_ = now;
        initialized_  = true;
    }

    if (handPresent && confidence > 0.0f)
        lastHandSeen_ = now;

    // Reset to IDLE if hand has been absent too long
    float noHandSec = std::chrono::duration<float>(now - lastHandSeen_).count();
    if (noHandSec > resetSec_ && phase_ != GesturePhase::IDLE) {
        if (verbose_) std::cout << "[Gesture] Hand lost — resetting to IDLE\n";
        reset();
        initialized_  = true;
        phaseStart_   = now;
        lastHandSeen_ = now;
    }

    auto elapsed = [&](TP start) {
        return std::chrono::duration<float>(now - start).count();
    };

    float progress = 0.0f;
    GestureResult result{phase_, false, "", 0.0f};

    if (!handPresent || static_cast<int>(kp.size()) < 21) {
        result.label = "Sin mano";
        return result;
    }

    if (verbose_) {
        GestureDebug dbg = getDebug(kp);
        std::cout << std::fixed << std::setprecision(2)
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

        // ── IDLE: wait for an open hand ──────────────────────────────────────
        case GesturePhase::IDLE: {
            if (checkOpenHand(kp)) {
                phase_     = GesturePhase::OPEN_HAND;
                phaseStart_= now;
                openFailCount_ = 0;
                if (verbose_) std::cout << "[Gesture] → OPEN_HAND\n";
            }
            result.label = "Esperando gesto...";
            break;
        }

        // ── OPEN_HAND: hold open palm for openHoldSec_ ───────────────────────
        case GesturePhase::OPEN_HAND: {
            if (!checkOpenHand(kp)) {
                ++openFailCount_;
                if (openFailCount_ > MAX_FAIL_FRAMES) {
                    // Gave enough chances — hand is no longer open
                    reset(); initialized_ = true; phaseStart_ = now; lastHandSeen_ = now;
                    result.label = "Esperando gesto...";
                    break;
                }
            } else {
                openFailCount_ = 0;  // reset tolerance counter on success
            }

            float held = elapsed(phaseStart_);
            progress = std::min(1.0f, held / openHoldSec_);

            if (held >= openHoldSec_ && checkThumbTucked(kp)) {
                phase_        = GesturePhase::THUMB_TUCKED;
                phaseStart_   = now;
                thumbFailCount_= 0;
                if (verbose_) std::cout << "[Gesture] → THUMB_TUCKED\n";
            }
            result.label = "Mano abierta (paso 1/3)";
            break;
        }

        // ── THUMB_TUCKED: wait for fingers to close over thumb ────────────────
        case GesturePhase::THUMB_TUCKED: {
            if (!checkThumbTucked(kp)) {
                ++thumbFailCount_;
                if (thumbFailCount_ > MAX_FAIL_FRAMES) {
                    // Thumb came back out
                    if (checkOpenHand(kp)) {
                        phase_     = GesturePhase::OPEN_HAND;
                        phaseStart_= now;
                        openFailCount_ = 0;
                        if (verbose_) std::cout << "[Gesture] → OPEN_HAND (pulgar extendido de nuevo)\n";
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
            progress = std::min(1.0f, held / thumbHoldSec_);

            if (held >= thumbHoldSec_ && checkFingersClosed(kp)) {
                phase_     = GesturePhase::SIGNAL_COMPLETE;
                phaseStart_= now;

                if (!alertAlreadyFired_) {
                    alertAlreadyFired_ = true;
                    result.alertFired  = true;

                    auto t  = std::time(nullptr);
                    auto tm = *std::localtime(&t);
                    std::ostringstream oss;
                    oss << std::put_time(&tm, "%Y-%m-%d %H:%M:%S");
                    std::cout << "\n*** SIGNAL FOR HELP DETECTADO a las " << oss.str() << " ***\n\n";
                }
                if (verbose_) std::cout << "[Gesture] → SIGNAL_COMPLETE\n";
            }
            result.label = "Pulgar adentro (paso 2/3)";
            break;
        }

        // ── SIGNAL_COMPLETE: hold alert until gesture is released ─────────────
        case GesturePhase::SIGNAL_COMPLETE: {
            progress = 1.0f;
            // Allow re-trigger: reset when hand opens again
            if (!checkFingersClosed(kp) && checkOpenHand(kp)) {
                reset(); initialized_ = true; phaseStart_ = now; lastHandSeen_ = now;
                if (verbose_) std::cout << "[Gesture] → IDLE (gesto liberado)\n";
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

float GestureDetector::dist(cv::Point2f a, cv::Point2f b) const {
    float dx = a.x - b.x, dy = a.y - b.y;
    return std::sqrt(dx * dx + dy * dy);
}

// Palm length = wrist to base of middle finger — our global scale reference.
// All other thresholds are expressed as multiples of this value, so they
// automatically adapt to any hand size and camera distance.
float GestureDetector::palmLen(const std::vector<cv::Point2f>& kp) const {
    return dist(kp[WRIST], kp[MIDDLE_MCP]);
}

// Palm center = geometric center of the four MCP knuckles.
// When the thumb is tucked it ends up near this point.
cv::Point2f GestureDetector::palmCenter(const std::vector<cv::Point2f>& kp) const {
    return (kp[INDEX_MCP] + kp[MIDDLE_MCP] + kp[RING_MCP] + kp[PINKY_MCP]) * 0.25f;
}

// A finger is EXTENDED when its tip is at least 15% farther from the wrist
// than its own MCP knuckle. This works for any hand orientation.
bool GestureDetector::fingerExtended(const std::vector<cv::Point2f>& kp,
                                     int tip, int mcp) const {
    return dist(kp[tip], kp[WRIST]) > dist(kp[mcp], kp[WRIST]) * 1.15f;
}

// A finger is CURLED when its tip has returned to approximately the MCP level
// (i.e. the finger is bent into the palm). Threshold 1.25 gives tolerance
// for natural partial curling — NOT requiring a fully closed fist to trigger.
bool GestureDetector::fingerCurled(const std::vector<cv::Point2f>& kp,
                                   int tip, int mcp) const {
    return dist(kp[tip], kp[WRIST]) < dist(kp[mcp], kp[WRIST]) * 1.25f;
}

// ============================================================================
// Gesture phase checks
// ============================================================================

// OPEN_HAND: at least 3 of 4 fingers are extended AND the thumb is not
// pressed against the palm (its tip is far from the palm center).
bool GestureDetector::checkOpenHand(const std::vector<cv::Point2f>& kp) const {
    int extCount = 0;
    if (fingerExtended(kp, INDEX_TIP,  INDEX_MCP))  ++extCount;
    if (fingerExtended(kp, MIDDLE_TIP, MIDDLE_MCP)) ++extCount;
    if (fingerExtended(kp, RING_TIP,   RING_MCP))   ++extCount;
    if (fingerExtended(kp, PINKY_TIP,  PINKY_MCP))  ++extCount;

    // Thumb is "out" when its tip is far from the palm center.
    // Threshold 0.80× palm length = thumb must be clearly away from the palm.
    float       pl     = palmLen(kp);
    cv::Point2f center = palmCenter(kp);
    bool        thumbOut = dist(kp[THUMB_TIP], center) > pl * 0.80f;

    return extCount >= 3 && thumbOut;
}

// THUMB_TUCKED: the thumb tip has moved to the palm center area.
// We no longer require other fingers to be extended here — the state machine
// already ensured we arrived from OPEN_HAND, so the sequence is guaranteed.
bool GestureDetector::checkThumbTucked(const std::vector<cv::Point2f>& kp) const {
    float       pl     = palmLen(kp);
    cv::Point2f center = palmCenter(kp);
    // Thumb tip must be within 85% of palm length from the palm center.
    return dist(kp[THUMB_TIP], center) < pl * 0.85f;
}

// SIGNAL_COMPLETE (fingers closed): at least 3 of 4 non-thumb fingers
// are curled back toward the wrist level. 3-of-4 instead of 4-of-4 gives
// tolerance for small landmark errors on the pinky or ring finger.
bool GestureDetector::checkFingersClosed(const std::vector<cv::Point2f>& kp) const {
    int curledCount = 0;
    if (fingerCurled(kp, INDEX_TIP,  INDEX_MCP))  ++curledCount;
    if (fingerCurled(kp, MIDDLE_TIP, MIDDLE_MCP)) ++curledCount;
    if (fingerCurled(kp, RING_TIP,   RING_MCP))   ++curledCount;
    if (fingerCurled(kp, PINKY_TIP,  PINKY_MCP))  ++curledCount;
    return curledCount >= 3;
}

// ============================================================================
// Debug info (no side effects — safe to call every frame)
// ============================================================================
GestureDebug GestureDetector::getDebug(const std::vector<cv::Point2f>& kp) const {
    GestureDebug d{};
    if (static_cast<int>(kp.size()) < 21) return d;

    d.palmSize         = palmLen(kp);
    cv::Point2f center = palmCenter(kp);
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
