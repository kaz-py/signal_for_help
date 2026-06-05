#pragma once
#include <opencv2/opencv.hpp>
#include <vector>

struct HandRegion {
    cv::Rect               bbox;
    double                 area;
    std::vector<cv::Point> contour;
};

// Detects candidate hand regions using dual YCrCb + HSV skin-color segmentation.
// Filters by area, aspect ratio, and convexity to reject non-hand skin regions.
// Optionally excludes the top fraction of the frame (face area).
class SkinDetector {
public:
    SkinDetector();

    std::vector<HandRegion> detect(const cv::Mat& bgr);

    cv::Mat getLastMask() const { return lastMask_; }

private:
    cv::Mat buildSkinMask(const cv::Mat& bgr) const;
    bool    couldBeHand(const std::vector<cv::Point>& contour, double area) const;

    // YCrCb skin ranges (Y, Cr, Cb) — classic Kovac 2003, widened for more tones
    cv::Scalar loYCC_{0,   125, 68};
    cv::Scalar hiYCC_{255, 188, 140};

    // HSV skin ranges — two arcs for red-orange hues (0-22° and 158-180°)
    cv::Scalar loHSV1_{0,   25, 45};
    cv::Scalar hiHSV1_{22, 255, 255};
    cv::Scalar loHSV2_{158, 25, 45};
    cv::Scalar hiHSV2_{180, 255, 255};

    // Fraction of frame height to blank (face exclusion, measured from top)
    float faceExclFrac_{0.22f};

    // Area bounds in pixels² (tuned for 640×480)
    double minArea_{1800.0};
    double maxArea_{200000.0};

    cv::Mat kernel5_;
    cv::Mat kernel3_;
    cv::Mat lastMask_;
};
