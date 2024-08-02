#ifndef HSL_H
#define HSL_H

#include <opencv2/opencv.hpp>
#include <string>

struct HSL {
    double h, s, l;
};

HSL rgb_to_hsl(double r, double g, double b);
cv::Vec3b hsl_to_rgb(double h, double s, double l);
cv::Mat adjust_yellow(const cv::Mat& img, double hue, double saturation, double lightness, const std::string& output_path);
cv::Mat adjust_green(const cv::Mat& img, double hue, double saturation, double lightness, const std::string& output_path);
cv::Mat adjust_hsl_yellow_frame(const cv::Mat& frame, double hue, double saturation, double lightness);
cv::Mat adjust_hsl_green_frame(const cv::Mat& frame, double hue, double saturation, double lightness);

#endif // HSL_H