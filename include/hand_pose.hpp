#pragma once
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <string>
#include <vector>

// MediaPipe hand landmark indices (21 keypoints)
namespace MP {
    enum Landmark : int {
        WRIST      =  0,
        THUMB_CMC  =  1, THUMB_MCP  =  2, THUMB_IP   =  3, THUMB_TIP  =  4,
        INDEX_MCP  =  5, INDEX_PIP  =  6, INDEX_DIP  =  7, INDEX_TIP  =  8,
        MIDDLE_MCP =  9, MIDDLE_PIP = 10, MIDDLE_DIP = 11, MIDDLE_TIP = 12,
        RING_MCP   = 13, RING_PIP   = 14, RING_DIP   = 15, RING_TIP   = 16,
        PINKY_MCP  = 17, PINKY_PIP  = 18, PINKY_DIP  = 19, PINKY_TIP  = 20,
        COUNT      = 21
    };
}

// Finger bone connectivity for drawing the hand skeleton
extern const int HAND_CONNECTIONS[][2];
extern const int HAND_CONNECTIONS_COUNT;

struct HandPoseResult {
    std::vector<cv::Point2f> landmarks;      // 21 points in original frame coordinates
    std::vector<float>       zCoords;        // relative depth (negative = closer to camera)
    float                    confidence{0.0f}; // hand presence score [0, 1]
    bool                     valid{false};
};

// Wraps the MediaPipe hand-pose ONNX model (handpose_estimation_mediapipe_2023feb.onnx).
// Input : BGR crop of the hand region (any size) → resized internally to 224×224.
// Output: 21 2-D landmarks mapped back to the original frame coordinate space.
class HandPoseEstimator {
public:
    explicit HandPoseEstimator(const std::string& modelPath,
                               float confThreshold = 0.5f,
                               int backend = cv::dnn::DNN_BACKEND_DEFAULT,
                               int target  = cv::dnn::DNN_TARGET_CPU);

    bool isLoaded() const { return loaded_; }

    // handCrop : sub-image cropped from the frame (BGR)
    // roi      : the bounding box of handCrop inside the full frame
    // frameSize: size of the full frame (for boundary clamping)
    HandPoseResult estimate(const cv::Mat& handCrop,
                            const cv::Rect& roi,
                            const cv::Size& frameSize);

private:
    cv::dnn::Net             net_;
    float                    confThreshold_;
    bool                     loaded_{false};
    std::vector<std::string> outNames_;

    static constexpr int MODEL_W = 224;
    static constexpr int MODEL_H = 224;
};
