#pragma once
#include <opencv2/opencv.hpp>
#include <vector>

struct HandRegion {
    cv::Rect               bbox;
    double                 area;
    std::vector<cv::Point> contour;
};

// Detects candidate hand regions using YCrCb skin-color segmentation.
// Filters by area, aspect ratio, and convexity to reject non-hand skin regions.
class SkinDetector {
public:
    SkinDetector();

    std::vector<HandRegion> detect(const cv::Mat& bgr);

    // Returns the raw binary skin mask from the last call to detect()
    cv::Mat getLastMask() const { return lastMask_; }

private:
    cv::Mat buildSkinMask(const cv::Mat& bgr) const;
    bool    couldBeHand(const std::vector<cv::Point>& contour, double area) const;

    // YCrCb skin ranges (OpenCV channel order: Y, Cr, Cb).
    // Wider than the classic Kovac 2003 range so more skin tones and lighting
    // conditions are accepted without false-positives from backgrounds.
    cv::Scalar loSkin_{0,   128, 70};
    cv::Scalar hiSkin_{255, 185, 138};

    // Area bounds in pixels² (tuned for 640×480; scales with resolution)
    double minArea_{2000.0};
    double maxArea_{220000.0};

    cv::Mat kernel5_;   // 5×5 ellipse for closing small gaps
    cv::Mat kernel3_;   // 3×3 ellipse for opening noise
    cv::Mat lastMask_;
};
