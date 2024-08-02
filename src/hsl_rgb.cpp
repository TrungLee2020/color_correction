#include <opencv2/opencv.hpp>
#include <iostream>
#include <cmath>
#include <chrono>
#include <filesystem>
#include <string>
struct HSL {
    double h, s, l;
};

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

void adjust_hsl(const std::string& input_path, const std::string& output_path, double hue, double saturation, double lightness) {
    cv::Mat img = cv::imread(input_path);
    if (img.empty()) {
        std::cerr << "Error: Could not read the image." << std::endl;
        return;
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
    std::cout << "Adjusted image saved at: " << output_path << std::endl;
    // return result;

}

int main() {
    auto start = std::chrono::high_resolution_clock::now();

    std::string input_folder = "data";
    std::string output_folder = "result_hsl";

    // Đảm bảo thư mục đầu ra tồn tại
    std::filesystem::create_directories(output_folder);

    // Duyệt qua tất cả các tệp trong thư mục đầu vào
    for (const auto & entry : std::filesystem::directory_iterator(input_folder)) {
        if (entry.is_regular_file()) {
            std::string input_path = entry.path().string();
            std::string file_name = entry.path().filename().string();
            std::string output_path = output_folder + "/" + file_name;

            std::cout << "Processing: " << file_name << std::endl;

            // Áp dụng điều chỉnh HSL cho mỗi ảnh
            adjust_hsl(input_path, output_path, 0, -70, 30);
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Processing time: " << duration.count() << " milliseconds" << std::endl;
    return 0;
}