#include "alert_system.hpp"
#include <iostream>
#include <cstdlib>  // std::system

void AlertSystem::trigger() {
    active_      = true;
    triggerTime_ = std::chrono::steady_clock::now();
    std::cout << "\a";  // terminal bell
    std::cout.flush();

    // Non-blocking system notification (Linux)
    std::system("notify-send -u critical 'Signal for Help' 'Gesture detected!' 2>/dev/null &");
}

void AlertSystem::deactivate() {
    active_ = false;
}

bool AlertSystem::isActive() const {
    if (!active_) return false;
    auto elapsed = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - triggerTime_).count();
    return elapsed < ALERT_DURATION_SEC;
}

void AlertSystem::render(cv::Mat& frame) const {
    if (!frame.empty() && isActive()) {
        // Semi-transparent red overlay
        cv::Mat overlay = frame.clone();
        cv::rectangle(overlay, {0, 0, frame.cols, frame.rows},
                      cv::Scalar(0, 0, 200), cv::FILLED);
        cv::addWeighted(overlay, 0.35, frame, 0.65, 0, frame);

        // Main warning text
        const std::string line1 = "SIGNAL FOR HELP DETECTED";
        const std::string line2 = "Requesting immediate assistance";

        int  fontFace  = cv::FONT_HERSHEY_DUPLEX;
        auto textScale = [&](const std::string& s, double scale) {
            int baseline = 0;
            cv::Size ts  = cv::getTextSize(s, fontFace, scale, 2, &baseline);
            cv::Point org((frame.cols - ts.width) / 2,
                          frame.rows / 2 - baseline);
            // Shadow
            cv::putText(frame, s, org + cv::Point(2, 2), fontFace, scale,
                        cv::Scalar(0, 0, 0), 3, cv::LINE_AA);
            // Text
            cv::putText(frame, s, org, fontFace, scale,
                        cv::Scalar(255, 255, 255), 2, cv::LINE_AA);
        };

        textScale(line1, 1.4);

        // Second line slightly lower
        {
            int baseline = 0;
            cv::Size ts  = cv::getTextSize(line1, fontFace, 1.4, 2, &baseline);
            int      yOff= frame.rows / 2 + ts.height + 16;
            cv::Size ts2 = cv::getTextSize(line2, fontFace, 0.8, 2, &baseline);
            cv::Point org2((frame.cols - ts2.width) / 2, yOff);
            cv::putText(frame, line2, org2 + cv::Point(1,1), fontFace, 0.8,
                        cv::Scalar(0,0,0), 3, cv::LINE_AA);
            cv::putText(frame, line2, org2, fontFace, 0.8,
                        cv::Scalar(255, 220, 80), 2, cv::LINE_AA);
        }

        // Pulsing red border
        auto elapsed = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - triggerTime_).count();
        bool blink = static_cast<int>(elapsed * 3) % 2 == 0;
        if (blink) {
            cv::rectangle(frame, {4, 4, frame.cols - 8, frame.rows - 8},
                          cv::Scalar(0, 0, 255), 6, cv::LINE_AA);
        }
    }
}
