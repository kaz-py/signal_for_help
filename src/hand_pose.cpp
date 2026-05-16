#include "hand_pose.hpp"
#include <iostream>
#include <stdexcept>

// Hand skeleton connections (pairs of landmark indices)
const int HAND_CONNECTIONS[][2] = {
    // Palm
    {0,1},{1,2},{2,3},{3,4},   // thumb
    {0,5},{5,6},{6,7},{7,8},   // index
    {0,9},{9,10},{10,11},{11,12},  // middle
    {0,13},{13,14},{14,15},{15,16},// ring
    {0,17},{17,18},{18,19},{19,20},// pinky
    {5,9},{9,13},{13,17}           // palm arc
};
const int HAND_CONNECTIONS_COUNT = static_cast<int>(
    sizeof(HAND_CONNECTIONS) / sizeof(HAND_CONNECTIONS[0]));

HandPoseEstimator::HandPoseEstimator(const std::string& modelPath,
                                     float confThreshold,
                                     int backend, int target)
    : confThreshold_(confThreshold)
{
    try {
        net_ = cv::dnn::readNet(modelPath);
        net_.setPreferableBackend(backend);
        net_.setPreferableTarget(target);

        // Discover output layer names so we don't hard-code them
        outNames_ = net_.getUnconnectedOutLayersNames();
        if (outNames_.empty())
            throw std::runtime_error("Model has no output layers");

        loaded_ = true;
        std::cout << "[HandPose] Model loaded: " << modelPath
                  << "  outputs: " << outNames_.size() << "\n";
        for (auto& n : outNames_) std::cout << "  - " << n << "\n";
    } catch (const std::exception& e) {
        std::cerr << "[HandPose] Failed to load model: " << e.what() << "\n";
    }
}

HandPoseResult HandPoseEstimator::estimate(const cv::Mat& handCrop,
                                           const cv::Rect& roi,
                                           const cv::Size& /*frameSize*/) {
    HandPoseResult result;
    if (!loaded_ || handCrop.empty()) return result;

    // --- Preprocess -------------------------------------------------------
    // Model was converted from TFLite (NHWC). It requires input shape (1,224,224,3)
    // in RGB order normalized to [0,1]. Standard blobFromImage produces NCHW and
    // causes a channel-mismatch error — we build the NHWC blob manually.
    cv::Mat resized;
    cv::resize(handCrop, resized, cv::Size(MODEL_W, MODEL_H));
    cv::cvtColor(resized, resized, cv::COLOR_BGR2RGB);
    cv::Mat imgF;
    resized.convertTo(imgF, CV_32F, 1.0 / 255.0);   // HWC float32 [0,1]
    // Add batch dim: (H,W,C) → (1,H,W,C) NHWC
    int sz[] = {1, MODEL_H, MODEL_W, 3};
    cv::Mat nhwcBlob(4, sz, CV_32F, imgF.data);      // wraps imgF memory (safe: imgF lives through forward())

    net_.setInput(nhwcBlob);

    // --- Forward pass -----------------------------------------------------
    std::vector<cv::Mat> outs;
    try {
        net_.forward(outs, outNames_);
    } catch (const cv::Exception& e) {
        std::cerr << "[HandPose] Inference error: " << e.what() << "\n";
        return result;
    }

    if (outs.empty()) return result;

    // --- Parse outputs ----------------------------------------------------
    // handpose_estimation_mediapipe_2023feb.onnx outputs (NHWC model):
    //   outs[0] Identity  : (1, 63) → 21 screen landmarks × (x, y, z), x,y in [0,224]
    //   outs[1] Identity_1: (1,  1) → hand-presence score (sigmoid, 0–1)
    //   outs[2] Identity_2: (1,  1) → handedness (0=left, 1=right) — unused
    //   outs[3] Identity_3: (1, 63) → 21 world landmarks (metric 3-D) — unused

    // Landmarks
    cv::Mat landmarkMat = outs[0].reshape(1, 1);  // ensure (1, 63)
    if (landmarkMat.total() < static_cast<size_t>(MP::COUNT * 3)) return result;

    const float* lmPtr = landmarkMat.ptr<float>(0);

    // Confidence
    float confidence = (outs.size() >= 2) ? outs[1].ptr<float>(0)[0] : 1.0f;
    // Apply sigmoid if score looks like logit (outside [0,1])
    if (confidence < 0.0f || confidence > 1.0f)
        confidence = 1.0f / (1.0f + std::exp(-confidence));

    result.confidence = confidence;
    if (confidence < confThreshold_) return result;  // not a confident hand detection

    // Map landmarks from model space (0..224) to ROI space to frame space
    float scaleX = static_cast<float>(roi.width)  / MODEL_W;
    float scaleY = static_cast<float>(roi.height) / MODEL_H;

    result.landmarks.resize(MP::COUNT);
    result.zCoords.resize(MP::COUNT);

    for (int i = 0; i < MP::COUNT; ++i) {
        float mx = lmPtr[i * 3 + 0];
        float my = lmPtr[i * 3 + 1];
        float mz = lmPtr[i * 3 + 2];

        result.landmarks[i] = cv::Point2f(
            roi.x + mx * scaleX,
            roi.y + my * scaleY);
        result.zCoords[i] = mz;
    }

    result.valid = true;
    return result;
}
