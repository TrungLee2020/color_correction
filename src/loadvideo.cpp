#include <opencv2/opencv.hpp>
#include <iostream>
#include <cmath>
#include <chrono>


using namespace cv;
struct HSL {
    double h, s, l;
};

HSL rgb_to_hsl(double r, double g, double b) {
    r /= 255.0;
    g /= 255.0;
    b /= 255.0;

    double mx = std::max({r, g, b});
    double mn = std::min({r, g, b});
    double h = (mx + mn) / 2.0;
    double s = h;
    double l = h;

    double c = mx - mn;
    if (c == 0) {
        h = s = 0;
    } else {
        if (mx == r) {
            h = 60 * ((g - b) / c);
            if (g < b) h += 360;
        } else if (mx == g) {
            h = 60 * ((b - r) / c + 2);
        } else if (mx == b) {
            h = 60 * ((r - g) / c + 4);
        }
        s = (l == 0 || l == 1) ? 0 : (mx - l) / std::min(l, 1 - l);
    }

    return { std::fmod(h, 360.0), s * 100, l * 100 };
}

cv::Vec3b hsl_to_rgb(double h, double s, double l) {
    s /= 100.0;
    l /= 100.0;

    double c = (1 - std::abs(2 * l - 1)) * s;
    double x = c * (1 - std::abs(std::fmod(h / 60.0, 2) - 1));
    double m = l - c / 2.0;

    double r = 0, g = 0, b = 0;
    if (0 <= h && h < 60) {
        r = c; g = x; b = 0;
    } else if (60 <= h && h < 120) {
        r = x; g = c; b = 0;
    } else if (120 <= h && h < 180) {
        r = 0; g = c; b = x;
    } else if (180 <= h && h < 240) {
        r = 0; g = x; b = c;
    } else if (240 <= h && h < 300) {
        r = x; g = 0; b = c;
    } else if (300 <= h && h < 360) {
        r = c; g = 0; b = x;
    }

    return cv::Vec3b(static_cast<uchar>((r + m) * 255),
                     static_cast<uchar>((g + m) * 255),
                     static_cast<uchar>((b + m) * 255));
}

cv::Mat adjust_hsl_frame(const cv::Mat& frame, double hue, double saturation, double lightness) {
    cv::Mat result = frame.clone();
    for (int y = 0; y < frame.rows; y++) {
        for (int x = 0; x < frame.cols; x++) {
            cv::Vec3b pixel = frame.at<cv::Vec3b>(y, x);
            HSL hsl = rgb_to_hsl(pixel[2], pixel[1], pixel[0]);  // OpenCV uses BGR

            hsl.h = std::fmod(hsl.h + hue, 360.0);
            hsl.s = std::min(std::max(hsl.s * (1 + saturation / 100), 0.0), 100.0);
            hsl.l = std::min(std::max(hsl.l * (1 + lightness / 100), 0.0), 100.0);

            cv::Vec3b rgb = hsl_to_rgb(hsl.h, hsl.s, hsl.l);
            result.at<cv::Vec3b>(y, x) = rgb;
        }
    }
    return result;
}

void process_video(const std::string& input_path, const std::string& output_path, double hue, double saturation, double lightness) {
    cv::VideoCapture cap(input_path);
    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open the video file: " << input_path << std::endl;
        return;
    }

    std::cout << "Successfully opened video file: " << input_path << std::endl;

    int frame_width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int frame_height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    double fps = cap.get(cv::CAP_PROP_FPS);

    cv::VideoWriter writer(output_path, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), fps, cv::Size(frame_width, frame_height));

    cv::Mat frame;
    while (cap.read(frame)) {
        cv::Mat adjusted_frame = adjust_hsl_frame(frame, hue, saturation, lightness);
        writer.write(adjusted_frame);
    }
    cap.release();
    writer.release();
    std::cout << "Processed video saved at: " << output_path << std::endl;
}

int main() {
    auto start = std::chrono::high_resolution_clock::now();

    std::string input_path = "/home/trunglx/Downloads/github/Linear_Color_Correction_Matrix/original_videos/2.mp4";  // Đường dẫn tới video đầu vào
    std::string output_path = "result_hsl_video/result_hsl_video.mp4";  // Đường dẫn tới video đầu ra

    std::cout << "Processing video: " << input_path << std::endl;

    process_video(input_path, output_path, 0, -40, 30);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Processing time: " << duration.count() << " milliseconds" << std::endl;

    return 0;
}
