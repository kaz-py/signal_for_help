#pragma once
#include <opencv2/opencv.hpp>
#include <chrono>

// Manages the visual alert overlay displayed when a Signal for Help is detected.
// The alert stays visible for ALERT_DURATION_SEC seconds after being triggered.
class AlertSystem {
public:
    AlertSystem() = default;

    // Activate the alert (call once when the gesture is confirmed)
    void trigger();

    // Deactivate manually (e.g., after a reset)
    void deactivate();

    bool isActive() const;

    // Draw the semi-transparent red overlay + warning text onto frame in-place
    void render(cv::Mat& frame) const;

private:
    bool active_{false};
    std::chrono::steady_clock::time_point triggerTime_;
    static constexpr double ALERT_DURATION_SEC = 8.0;
};
