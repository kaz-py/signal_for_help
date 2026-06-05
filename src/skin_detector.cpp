#include "skin_detector.hpp"
#include <algorithm>

using namespace cv;

SkinDetector::SkinDetector()
    : kernel5_(getStructuringElement(MORPH_ELLIPSE, {5, 5})),
      kernel3_(getStructuringElement(MORPH_ELLIPSE, {3, 3}))
{}

std::vector<HandRegion> SkinDetector::detect(const Mat& bgr) {
    Mat mask = buildSkinMask(bgr);
    lastMask_ = mask.clone();

    std::vector<std::vector<Point>> contours;
    findContours(mask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    std::vector<HandRegion> regions;
    for (auto& c : contours) {
        double area = contourArea(c);
        if (area < minArea_ || area > maxArea_) continue;
        if (!couldBeHand(c, area)) continue;

        HandRegion r;
        r.contour = c;
        r.area    = area;
        r.bbox    = boundingRect(c);
        regions.push_back(std::move(r));
    }

    std::sort(regions.begin(), regions.end(),
              [](const HandRegion& a, const HandRegion& b){ return a.area > b.area; });

    if (regions.size() > 2) regions.resize(2);
    return regions;
}

Mat SkinDetector::buildSkinMask(const Mat& bgr) const {
    // ── YCrCb channel ────────────────────────────────────────────────────────
    Mat ycrcb;
    cvtColor(bgr, ycrcb, COLOR_BGR2YCrCb);
    Mat maskYCC;
    inRange(ycrcb, loYCC_, hiYCC_, maskYCC);

    // ── HSV channel — two arcs covering red-orange skin hues ─────────────────
    Mat hsv;
    cvtColor(bgr, hsv, COLOR_BGR2HSV);
    Mat m1, m2, maskHSV;
    inRange(hsv, loHSV1_, hiHSV1_, m1);
    inRange(hsv, loHSV2_, hiHSV2_, m2);
    bitwise_or(m1, m2, maskHSV);

    // ── Combine both channels with OR for maximum recall ──────────────────────
    Mat mask;
    bitwise_or(maskYCC, maskHSV, mask);

    // ── Exclude top portion of frame (typically the face region) ─────────────
    int faceRows = static_cast<int>(bgr.rows * faceExclFrac_);
    if (faceRows > 0 && faceRows < mask.rows)
        mask(Rect(0, 0, mask.cols, faceRows)).setTo(0);

    // ── Morphological cleanup ─────────────────────────────────────────────────
    morphologyEx(mask, mask, MORPH_OPEN,   kernel3_);  // remove specks
    morphologyEx(mask, mask, MORPH_CLOSE,  kernel5_);  // close small holes
    morphologyEx(mask, mask, MORPH_DILATE, kernel3_);  // gentle join of nearby blobs
    return mask;
}

bool SkinDetector::couldBeHand(const std::vector<Point>& contour, double area) const {
    Rect   bbox   = boundingRect(contour);
    double aspect = static_cast<double>(bbox.width) / std::max(1, bbox.height);

    if (aspect < 0.20 || aspect > 3.5) return false;

    std::vector<Point> hull;
    convexHull(contour, hull);
    double hullArea = contourArea(hull);
    if (hullArea < 1.0) return false;

    double solidity = area / hullArea;
    // Open hand: ~0.55–0.85 | fist: ~0.85–0.98 | noise: < 0.30
    if (solidity < 0.28 || solidity > 1.0) return false;

    return true;
}
