#include "hsl.hpp"
#include <iostream>
#include <cmath>

using namespace cv;

HSL rgb_to_hsl(double r, double g, double b) {
    r /= 255.0;
    g /= 255.0;
    b /= 255.0;
    double cmax = std::max({r, g, b});
    double cmin = std::min({r, g, b});
    double diff = cmax - cmin;

    HSL result;
    result.l = (cmax + cmin) / 2;

    if (cmax == cmin) {
        result.h = result.s = 0;
    } else {
        result.s = result.l <= 0.5 ? diff / (cmax + cmin) : diff / (2.0 - cmax - cmin);
        
        if (cmax == r) {
            result.h = (g - b) / diff + (g < b ? 6 : 0);
        } else if (cmax == g) {
            result.h = (b - r) / diff + 2;
        } else {
            result.h = (r - g) / diff + 4;
        }
        result.h *= 60;
    }

    result.s *= 100;
    result.l *= 100;
    return result;
}

cv::Vec3b hsl_to_rgb(double h, double s, double l) {
    s /= 100;
    l /= 100;
    
    auto hue_to_rgb = [](double p, double q, double t) {
        if (t < 0) t += 1;
        if (t > 1) t -= 1;
        if (t < 1.0/6) return p + (q - p) * 6 * t;
        if (t < 1.0/2) return q;
        if (t < 2.0/3) return p + (q - p) * (2.0/3 - t) * 6;
        return p;
    };

    if (s == 0) {
        return cv::Vec3b(l * 255, l * 255, l * 255);
    } else {
        double q = l < 0.5 ? l * (1 + s) : l + s - l * s;
        double p = 2 * l - q;
        double r = hue_to_rgb(p, q, h / 360.0 + 1.0/3);
        double g = hue_to_rgb(p, q, h / 360.0);
        double b = hue_to_rgb(p, q, h / 360.0 - 1.0/3);
        return cv::Vec3b(b * 255, g * 255, r * 255);
    }
}

cv::Mat adjust_yellow(const cv::Mat& img, double hue, double saturation, double lightness, const std::string& output_path) {
    if (img.empty()) {
        std::cerr << "Error: Could not read the image." << std::endl;
        return cv::Mat();
    }

    cv::Mat result = img.clone();
    for (int y = 0; y < img.rows; y++) {
        for (int x = 0; x < img.cols; x++) {
            cv::Vec3b pixel = img.at<cv::Vec3b>(y, x);
            HSL hsl = rgb_to_hsl(pixel[2], pixel[1], pixel[0]);  // OpenCV uses BGR

            hsl.h = std::fmod(hsl.h + hue, 360.0);
            hsl.s = std::clamp(hsl.s * (1 + saturation / 100), 0.0, 100.0);
            hsl.l = std::clamp(hsl.l * (1 + lightness / 100), 0.0, 100.0);

            result.at<cv::Vec3b>(y, x) = hsl_to_rgb(hsl.h, hsl.s, hsl.l);
        }
    }

    cv::imwrite(output_path, result);
    std::cout << "Adjusted Yellow image saved at: " << output_path << std::endl;
    return result;

}
cv::Mat adjust_green(const cv::Mat& img, double hue, double saturation, double lightness, const std::string& output_path) {
    if (img.empty()) {
        std::cerr << "Error: Could not read the image." << std::endl;
        return cv::Mat();
    }

    cv::Mat result = img.clone();
    for (int y = 0; y < img.rows; y++) {
        for (int x = 0; x < img.cols; x++) {
            cv::Vec3b pixel = img.at<cv::Vec3b>(y, x);
            HSL hsl = rgb_to_hsl(pixel[2], pixel[1], pixel[0]);  // OpenCV uses BGR

            // Điều chỉnh màu xanh lá cây (khoảng 60-180 độ trong hệ HSL)
            if (hsl.h >= 60 && hsl.h <= 180) {
                hsl.h = std::clamp(hsl.h + hue, 60.0, 180.0);
                hsl.s = std::clamp(hsl.s * (1 + saturation / 100.0), 0.0, 100.0);
                hsl.l = std::clamp(hsl.l * (1 + lightness / 100.0), 0.0, 100.0);
            }

            result.at<cv::Vec3b>(y, x) = hsl_to_rgb(hsl.h, hsl.s, hsl.l);
        }
    }

    cv::imwrite(output_path, result);
    std::cout << "Adjusted Green image saved at: " << output_path << std::endl;
    return result;
}

cv::Mat adjust_hsl_yellow_frame(const cv::Mat& frame, double hue, double saturation, double lightness) {
    cv::Mat result = frame.clone();
    for (int y = 0; y < frame.rows; y++) {
        for (int x = 0; x < frame.cols; x++) {
            cv::Vec3b pixel = frame.at<cv::Vec3b>(y, x);
            HSL hsl = rgb_to_hsl(pixel[2], pixel[1], pixel[0]);  // OpenCV uses BGR

            hsl.h = std::fmod(hsl.h + hue, 360.0);
            hsl.s = std::clamp(hsl.s * (1 + saturation / 100), 0.0, 100.0);
            hsl.l = std::clamp(hsl.l * (1 + lightness / 100), 0.0, 100.0);

            result.at<cv::Vec3b>(y, x) = hsl_to_rgb(hsl.h, hsl.s, hsl.l);
        }
    }
    return result;
}

cv::Mat adjust_hsl_green_frame(const cv::Mat& frame, double hue, double saturation, double lightness) {
    cv::Mat result = frame.clone();
    for (int y = 0; y < frame.rows; y++) {
        for (int x = 0; x < frame.cols; x++) {
            cv::Vec3b pixel = frame.at<cv::Vec3b>(y, x);
            HSL hsl = rgb_to_hsl(pixel[2], pixel[1], pixel[0]);  // OpenCV uses BGR

            // Điều chỉnh màu xanh lá cây (khoảng 60-180 độ trong hệ HSL)
            if (hsl.h >= 60 && hsl.h <= 180) {
                hsl.h = std::clamp(hsl.h + hue, 60.0, 180.0);
                hsl.s = std::clamp(hsl.s * (1 + saturation / 100.0), 0.0, 100.0);
                hsl.l = std::clamp(hsl.l * (1 + lightness / 100.0), 0.0, 100.0);
            }

            result.at<cv::Vec3b>(y, x) = hsl_to_rgb(hsl.h, hsl.s, hsl.l);
        }
    }
    return result;
}

// int main() {
//     cv::Mat image = cv::imread("data/da1fade93250900ec941.jpg");
//     // Adjust yellow colors
//     cv::Mat yellow_adjusted = adjust_yellow(image, 0, -70, 40, "result_hsl/output_yellow_adjusted.jpg");
//     // Adjust green colors on the yellow-adjusted image
//     cv::Mat final_result = adjust_green(yellow_adjusted, 20, 40, -5, "result_hsl/output_final.jpg");
//     return 0;
// }