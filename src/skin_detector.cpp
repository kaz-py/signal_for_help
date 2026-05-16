#include "skin_detector.hpp"
#include <algorithm>

SkinDetector::SkinDetector()
    : kernel5_(cv::getStructuringElement(cv::MORPH_ELLIPSE, {5, 5})),
      kernel3_(cv::getStructuringElement(cv::MORPH_ELLIPSE, {3, 3}))
{}

std::vector<HandRegion> SkinDetector::detect(const cv::Mat& bgr) {
    cv::Mat mask = buildSkinMask(bgr);
    lastMask_ = mask.clone();

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    std::vector<HandRegion> regions;
    for (auto& c : contours) {
        double area = cv::contourArea(c);
        if (area < minArea_ || area > maxArea_) continue;
        if (!couldBeHand(c, area)) continue;

        HandRegion r;
        r.contour = c;
        r.area    = area;
        r.bbox    = cv::boundingRect(c);
        regions.push_back(std::move(r));
    }

    // Sort largest first so we process the most prominent hand first
    std::sort(regions.begin(), regions.end(),
              [](const HandRegion& a, const HandRegion& b){ return a.area > b.area; });

    // Limit to 2 candidates to keep inference fast
    if (regions.size() > 2) regions.resize(2);
    return regions;
}

cv::Mat SkinDetector::buildSkinMask(const cv::Mat& bgr) const {
    cv::Mat ycrcb;
    cv::cvtColor(bgr, ycrcb, cv::COLOR_BGR2YCrCb);

    cv::Mat mask;
    cv::inRange(ycrcb, loSkin_, hiSkin_, mask);

    // Remove noise, then close small gaps
    cv::morphologyEx(mask, mask, cv::MORPH_OPEN,  kernel3_);
    cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, kernel5_);
    cv::morphologyEx(mask, mask, cv::MORPH_DILATE, kernel5_);
    return mask;
}

bool SkinDetector::couldBeHand(const std::vector<cv::Point>& contour, double area) const {
    cv::Rect   bbox   = cv::boundingRect(contour);
    double     aspect = static_cast<double>(bbox.width) / bbox.height;

    // Hands are taller than wide most of the time; reject very elongated shapes
    if (aspect < 0.2 || aspect > 3.5) return false;

    // Convexity check: a hand with extended fingers has moderate solidity
    std::vector<cv::Point> hull;
    cv::convexHull(contour, hull);
    double hullArea = cv::contourArea(hull);
    if (hullArea < 1.0) return false;
    double solidity = area / hullArea;

    // A fully open hand ~0.55–0.85; a fist ~0.85–0.98; random blobs ~<0.35
    if (solidity < 0.30 || solidity > 1.0) return false;

    return true;
}
