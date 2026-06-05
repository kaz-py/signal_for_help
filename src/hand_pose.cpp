#include "hand_pose.hpp"
#include <iostream>
#include <stdexcept>

using namespace cv;
using namespace cv::dnn;
using namespace std;

// Hand skeleton connections (pairs of landmark indices)
const int HAND_CONNECTIONS[][2] = {
    {0,1},{1,2},{2,3},{3,4},          // thumb
    {0,5},{5,6},{6,7},{7,8},          // index
    {0,9},{9,10},{10,11},{11,12},     // middle
    {0,13},{13,14},{14,15},{15,16},   // ring
    {0,17},{17,18},{18,19},{19,20},   // pinky
    {5,9},{9,13},{13,17}              // palm arc
};
const int HAND_CONNECTIONS_COUNT = static_cast<int>(
    sizeof(HAND_CONNECTIONS) / sizeof(HAND_CONNECTIONS[0]));

HandPoseEstimator::HandPoseEstimator(const string& modelPath,
                                     float confThreshold,
                                     int backend, int target)
    : confThreshold_(confThreshold)
{
    try {
        net_ = readNet(modelPath);
        net_.setPreferableBackend(backend);
        net_.setPreferableTarget(target);

        outNames_ = net_.getUnconnectedOutLayersNames();
        if (outNames_.empty())
            throw runtime_error("Model has no output layers");

        loaded_ = true;
        cout << "[HandPose] Model loaded: " << modelPath
             << "  outputs: " << outNames_.size() << "\n";
        for (auto& n : outNames_) cout << "  - " << n << "\n";
    } catch (const exception& e) {
        cerr << "[HandPose] Failed to load model: " << e.what() << "\n";
    }
}

HandPoseResult HandPoseEstimator::estimate(const Mat& handCrop,
                                           const Rect& roi,
                                           const Size& /*frameSize*/) {
    HandPoseResult result;
    if (!loaded_ || handCrop.empty()) return result;

    // ── Preprocess: BGR → RGB, resize to 224×224, normalize to [0,1] ─────────
    Mat resized;
    resize(handCrop, resized, Size(MODEL_W, MODEL_H));
    cvtColor(resized, resized, COLOR_BGR2RGB);

    // Convert to float and build NHWC blob (model requires NHWC, not NCHW)
    // We allocate owned data to guarantee lifetime through forward()
    Mat imgF;
    resized.convertTo(imgF, CV_32F, 1.0 / 255.0);

    // Clone ensures this Mat owns its data (imgF might share with resized on some paths)
    Mat nhwcBlob = imgF.clone();
    nhwcBlob = nhwcBlob.reshape(1, {1, MODEL_H, MODEL_W, 3});

    net_.setInput(nhwcBlob);

    // ── Forward pass ──────────────────────────────────────────────────────────
    vector<Mat> outs;
    try {
        net_.forward(outs, outNames_);
    } catch (const Exception& e) {
        cerr << "[HandPose] Inference error: " << e.what() << "\n";
        return result;
    }

    if (outs.empty()) return result;

    // ── Parse outputs ─────────────────────────────────────────────────────────
    // outs[0] Identity  : (1, 63) → 21 landmarks × (x, y, z), x/y in [0,224]
    // outs[1] Identity_1: (1,  1) → hand-presence score (sigmoid 0–1)
    // outs[2] Identity_2: (1,  1) → handedness (unused)
    // outs[3] Identity_3: (1, 63) → world landmarks (unused)

    Mat landmarkMat = outs[0].reshape(1, 1);
    if (landmarkMat.total() < static_cast<size_t>(MP::COUNT * 3)) return result;

    const float* lmPtr = landmarkMat.ptr<float>(0);

    float confidence = (outs.size() >= 2) ? outs[1].ptr<float>(0)[0] : 1.0f;
    // Apply sigmoid if the score is outside [0,1] (raw logit)
    if (confidence < 0.0f || confidence > 1.0f)
        confidence = 1.0f / (1.0f + std::exp(-confidence));

    result.confidence = confidence;
    if (confidence < confThreshold_) return result;

    // ── Map landmarks from model space [0..224] → frame space ─────────────────
    float scaleX = static_cast<float>(roi.width)  / MODEL_W;
    float scaleY = static_cast<float>(roi.height) / MODEL_H;

    result.landmarks.resize(MP::COUNT);
    result.zCoords.resize(MP::COUNT);

    for (int i = 0; i < MP::COUNT; ++i) {
        float mx = lmPtr[i * 3 + 0];
        float my = lmPtr[i * 3 + 1];
        float mz = lmPtr[i * 3 + 2];

        result.landmarks[i] = Point2f(roi.x + mx * scaleX,
                                      roi.y + my * scaleY);
        result.zCoords[i] = mz;
    }

    result.valid = true;
    return result;
}
