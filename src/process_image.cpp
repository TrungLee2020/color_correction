#include <iostream>
#include <opencv4/opencv2/opencv.hpp>
#include <fstream>
#include <vector>
#include <chrono>
#include <filesystem>
#include <string>

using namespace std;
namespace fs = std::filesystem;

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
cv::Mat applyColorCorrection(const cv::Mat& img, const cv::Mat& ColorMatrix) {
    cv::Mat Dst = img.clone();
    int channels = img.channels();
    int ImgHeight = img.rows, ImgWidth = img.cols;

    const float *CMC_1 = ColorMatrix.ptr<float>(0);
    const float *CMC_2 = ColorMatrix.ptr<float>(1);
    const float *CMC_3 = ColorMatrix.ptr<float>(2);
    
    for (int i = 0; i < ImgHeight; ++i) {
        const uchar *SP = img.ptr<uchar>(i);
        uchar *DP = Dst.ptr<uchar>(i);
        for (int j = 0; j < ImgWidth*channels; j += 3) {
            DP[j] = cv::saturate_cast<uchar>(SP[j] * CMC_1[0] + SP[j + 1] * CMC_2[0] + SP[j+2] * CMC_3[0]);
            DP[j+1] = cv::saturate_cast<uchar>(SP[j] * CMC_1[1] + SP[j + 1] * CMC_2[1] + SP[j+2] * CMC_3[1]);
            DP[j+2] = cv::saturate_cast<uchar>(SP[j] * CMC_1[2] + SP[j + 1] * CMC_2[2] + SP[j+2] * CMC_3[2]);
        }
    }
    double alpha = 0.95; // Điều chỉnh giá trị này để thay đổi độ sáng (< 1.0 để giảm, > 1.0 để tăng)
    Dst.convertTo(Dst, -1, alpha, 0);
    return Dst;
}

cv::Mat readColorCorrectionMatrix(const std::string& filename) {
    std::fstream CMC(filename, std::ios::in);
    
    if (!CMC) {
        std::cerr << "Error opening the file: " << filename << std::endl;
        exit(EXIT_FAILURE);
    }
    
    cv::Mat ColorMatrix(3, 3, CV_32FC1, cv::Scalar(0));
    
    std::string textline;
    int i = 0;
    while (getline(CMC, textline)) {
        std::string::size_type pos = 0, prev_pos = 0;
        float *CMCPtr = ColorMatrix.ptr<float>(i);
        int j = 0;
        while ((pos = textline.find_first_of(',', pos)) != std::string::npos) {
            CMCPtr[j++] = std::stof(textline.substr(prev_pos, pos - prev_pos));
            prev_pos = ++pos;
        }
        i++;
    }
    CMC.close();
    
    return ColorMatrix;
}

cv::Mat unsharpMask(const cv::Mat& input, float amount) {
    cv::Mat blurred;
    cv::GaussianBlur(input, blurred, cv::Size(0, 0), 3);
    cv::Mat sharpened = input * (1 + amount) + blurred * (-amount);
    return sharpened;
}
cv::Mat gammaCorrection(const cv::Mat& input, float gamma) {
    cv::Mat lookUpTable(1, 256, CV_8U);
    uchar* p = lookUpTable.ptr();
    for(int i = 0; i < 256; ++i)
        p[i] = cv::saturate_cast<uchar>(pow(i / 255.0, gamma) * 255.0);
    
    cv::Mat output;
    cv::LUT(input, lookUpTable, output);
    return output;
}
cv::Mat adjustWhiteBalance(const cv::Mat &img) {
    cv::Mat result;
    cv::Mat lab;
    cv::cvtColor(img, lab, cv::COLOR_BGR2Lab);

    std::vector<cv::Mat> lab_channels(3);
    cv::split(lab, lab_channels);

    double mean_l = cv::mean(lab_channels[0])[0];
    double mean_a = cv::mean(lab_channels[1])[0];
    double mean_b = cv::mean(lab_channels[2])[0];

    lab_channels[1] -= (mean_a - 129);
    lab_channels[2] -= (mean_b - 129);

    cv::merge(lab_channels, lab);
    cv::cvtColor(lab, result, cv::COLOR_Lab2BGR);
    return result;
}
cv::Mat bilateralFilter(const cv::Mat& input, int d, double sigmaColor, double sigmaSpace) {
    cv::Mat output;
    cv::bilateralFilter(input, output, d, sigmaColor, sigmaSpace);
    return output;
}
void processImages(const std::string& inputDir, const std::string& outputDir, const std::string& cmcFile) {
    cv::Mat ColorMatrix = readColorCorrectionMatrix(cmcFile);

    for (const auto & entry : fs::directory_iterator(inputDir)) {
        if (entry.path().extension() == ".jpg" || entry.path().extension() == ".png") {
            std::cout << "Processing: " << entry.path() << std::endl;
            
            cv::Mat img = cv::imread(entry.path().string());
            if (img.empty()) {
                std::cerr << "Cannot read image: " << entry.path() << std::endl;
                continue;
            }

            cv::Mat corrected = applyColorCorrection(img, ColorMatrix);
            
            // You can add more processing steps here if needed
            // For example:
            // corrected = unsharpMask(corrected, 0.5);
            corrected = gammaCorrection(corrected, 1.2);
            corrected = adjustWhiteBalance(corrected);
            // corrected = bilateralFilter(corrected, 9, 75, 75);

            std::string outputPath = outputDir + "/" + entry.path().filename().string();
            cv::imwrite(outputPath, corrected);
            std::cout << "Saved: " << outputPath << std::endl;
        }
    }
}


int main() {
    auto start = std::chrono::high_resolution_clock::now();

    // std::string inputDir = "result_hsl";
    std::string outputDir = "results";
    std::string cmcFile = "ref/LCC_CMC.csv";

    std::string input_folder_hsl = "data";
    std::string output_folder_hsl = "result_hsl";

    fs::create_directories(output_folder_hsl);

        // Duyệt qua tất cả các tệp trong thư mục đầu vào
    for (const auto & entry : fs::directory_iterator(input_folder_hsl)) {
        if (entry.is_regular_file()) {
            std::string input_path = entry.path().string();
            std::string file_name = entry.path().filename().string();
            std::string output_path = output_folder_hsl + "/" + file_name;

            std::cout << "Processing: " << file_name << std::endl;

            // Áp dụng điều chỉnh HSL cho mỗi ảnh
            adjust_hsl(input_path, output_path, 0, -40, 30);
            // Áp dụng điều chỉnh HSL cho anh vach ke duong
            // adjust_hsl(input_path, output_path, 0, -70, 30);

        }
    }

    processImages(output_folder_hsl, outputDir, cmcFile);

    std::cout << "All images processed." << std::endl;

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Processing time: " << duration.count() << " milliseconds" << std::endl;
    return 0;
}