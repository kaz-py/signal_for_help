#pragma once
#include <opencv2/opencv.hpp>
#include <chrono>
#include <string>
#include <vector>

// The three observable phases of the "Signal for Help" gesture
// (defined by the Canadian Women's Foundation):
//   1. Open palm facing camera (all fingers extended)
//   2. Thumb folds across the palm
//   3. Fingers curl down over the thumb (completes the signal)
enum class GesturePhase {
    IDLE,           // No confident hand, or wrong gesture
    OPEN_HAND,      // Palm open, all fingers extended
    THUMB_TUCKED,   // Thumb folded in, other fingers still extended
    SIGNAL_COMPLETE // Fingers closed over thumb → emit alert
};

struct GestureResult {
    GesturePhase phase;
    bool         alertFired;  // true on the single frame the transition to SIGNAL_COMPLETE happens
    std::string  label;       // human-readable phase name
    float        progress;    // 0.0–1.0 progress toward next phase (for HUD timer bar)
};

// Per-frame diagnostic info (shown in the debug overlay)
struct GestureDebug {
    bool  indexExtended, middleExtended, ringExtended, pinkyExtended;
    bool  thumbOut;         // thumb tip far from palm center
    bool  thumbTucked;      // thumb tip near palm center
    bool  fingersClosed;
    float palmSize;         // wrist-to-middle-MCP distance in pixels
    float thumbToPalmRatio; // dist(thumb_tip, palm_center) / palmSize
};

// Finite-state machine that tracks the gesture sequence frame by frame.
// Each phase must be held for a minimum duration to advance to the next,
// preventing accidental triggers from natural hand movement.
// A "failure tolerance" counter prevents single bad frames from resetting state.
class GestureDetector {
public:
    explicit GestureDetector(float openHoldSec  = 0.8f,
                             float thumbHoldSec = 0.5f,
                             float resetSec     = 1.5f);

    // Call once per frame with the 21 MediaPipe landmarks (empty if no hand)
    GestureResult update(const std::vector<cv::Point2f>& kp,
                         float confidence, bool handPresent);

    void         reset();
    GesturePhase phase() const { return phase_; }

    // Set verbose=true to print frame-by-frame geometry values to stdout
    void setVerbose(bool v) { verbose_ = v; }

    // Returns diagnostic info for the current landmarks (safe to call with empty kp)
    GestureDebug getDebug(const std::vector<cv::Point2f>& kp) const;

private:
    using Clock = std::chrono::steady_clock;
    using TP    = Clock::time_point;

    // High-level gesture checks
    // minExtended: how many of the 4 fingers (index-pinky) must be extended.
    //   Use 3 when transitioning FROM IDLE (more tolerant for tilted hands),
    //   use 4 when validating an already-open hand (avoids false continuations).
    bool checkOpenHand   (const std::vector<cv::Point2f>& kp, int minExtended = 4) const;
    bool checkThumbTucked(const std::vector<cv::Point2f>& kp) const;
    bool checkFingersClosed(const std::vector<cv::Point2f>& kp) const;

    // Per-finger helpers
    // Extended: tip is clearly FARTHER from wrist than its MCP (finger points away from palm)
    bool  fingerExtended(const std::vector<cv::Point2f>& kp, int tip, int mcp) const;
    // Curled: tip has come back close to palm level (finger is bent into fist)
    bool  fingerCurled  (const std::vector<cv::Point2f>& kp, int tip, int mcp) const;

    float palmLen   (const std::vector<cv::Point2f>& kp) const;
    cv::Point2f palmCenter(const std::vector<cv::Point2f>& kp) const;
    float dist      (cv::Point2f a, cv::Point2f b) const;

    GesturePhase phase_{GesturePhase::IDLE};
    TP           phaseStart_;
    TP           lastHandSeen_;
    bool         initialized_{false};
    bool         alertAlreadyFired_{false};
    bool         verbose_{false};
    bool         fastMode_{false};  // true after first complete gesture — skip step 1

    // Failure-tolerance counters: how many consecutive frames the current check
    // has been failing. We only transition back when tolerance is exceeded.
    int  openFailCount_{0};
    int  thumbFailCount_{0};
    static constexpr int MAX_FAIL_FRAMES = 5;  // ~0.17 s at 30 fps

    float openHoldSec_;
    float thumbHoldSec_;
    float resetSec_;

    // MediaPipe index aliases
    static constexpr int WRIST      =  0;
    static constexpr int THUMB_CMC  =  1;
    static constexpr int THUMB_MCP  =  2;
    static constexpr int THUMB_TIP  =  4;
    static constexpr int INDEX_MCP  =  5;
    static constexpr int INDEX_TIP  =  8;
    static constexpr int MIDDLE_MCP =  9;
    static constexpr int MIDDLE_TIP = 12;
    static constexpr int RING_MCP   = 13;
    static constexpr int RING_TIP   = 16;
    static constexpr int PINKY_MCP  = 17;
    static constexpr int PINKY_TIP  = 20;
};
