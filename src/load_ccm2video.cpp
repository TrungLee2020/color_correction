#include <iostream>
#include <opencv2/opencv.hpp>
#include <fstream>
#include <vector>
#include <cmath>

using namespace std;

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

    lab_channels[1] -= (mean_a - 128);
    lab_channels[2] -= (mean_b - 128);

    cv::merge(lab_channels, lab);
    cv::cvtColor(lab, result, cv::COLOR_Lab2BGR);
    return result;
}

cv::Mat bilateralFilter(const cv::Mat& input, int d, double sigmaColor, double sigmaSpace) {
    cv::Mat output;
    cv::bilateralFilter(input, output, d, sigmaColor, sigmaSpace);
    return output;
}

int main() {
    // Mở video đầu vào
    cv::VideoCapture cap("result_hsl_video/vach_ke_duong/95.mp4");
    if (!cap.isOpened()) {
        std::cerr << "Cannot open video file" << std::endl;
        return -1;
    }

    // Lấy thông tin video
    int frame_width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int frame_height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    double fps = cap.get(cv::CAP_PROP_FPS);
    
    // Tạo đối tượng VideoWriter để ghi video đầu ra
    cv::VideoWriter writer("result_ccm_video/vach_ke_duong/95.mp4", cv::VideoWriter::fourcc('m','p','4','v'), fps, cv::Size(frame_width, frame_height));
    if (!writer.isOpened()) {
        std::cerr << "Cannot open video writer" << std::endl;
        return -1;
    }

    int i = 0, j = 0;
    cv::Mat ColorMatrix(3, 3, CV_32FC1, cv::Scalar(0));
    std::fstream CMC("ref/LCC_CMC.csv", std::ios::in);
    
    if (!CMC) {
        std::cerr << "Open the file error" << std::endl;
        return -1;
    }
    
    std::string textline;
    while (getline(CMC, textline)) {
        std::string::size_type pos = 0, prev_pos = 0;
        float *CMCPtr = ColorMatrix.ptr<float>(i);
        j = 0;
        while ((pos = textline.find_first_of(',', pos)) != std::string::npos) {
            CMCPtr[j++] = std::stof(textline.substr(prev_pos, pos - prev_pos));
            prev_pos = ++pos;
        }
        i++;
    }
    CMC.close();
    
    cv::Mat frame;
    while (cap.read(frame)) {
        if (frame.empty()) break;
        
        cv::Mat Dst = frame.clone();
        
        float *CMC_1 = ColorMatrix.ptr<float>(0);
        float *CMC_2 = ColorMatrix.ptr<float>(1);
        float *CMC_3 = ColorMatrix.ptr<float>(2);
        
        for (i = 0; i < frame.rows; ++i) {
            uchar *SP = frame.ptr<uchar>(i);
            uchar *DP = Dst.ptr<uchar>(i);
            for (j = 0; j < frame.cols * frame.channels(); j += 3) {
                DP[j] = cv::saturate_cast<uchar>(SP[j] * CMC_1[0] + SP[j + 1] * CMC_2[0] + SP[j + 2] * CMC_3[0]);
                DP[j + 1] = cv::saturate_cast<uchar>(SP[j] * CMC_1[1] + SP[j + 1] * CMC_2[1] + SP[j + 2] * CMC_3[1]);
                DP[j + 2] = cv::saturate_cast<uchar>(SP[j] * CMC_1[2] + SP[j + 1] * CMC_2[2] + SP[j + 2] * CMC_3[2]);
            }
        }
        
        // Áp dụng White Balance
        // Dst = adjustWhiteBalance(Dst);
        
        // Tăng độ sắc nét
        // float sharpAmount = 0.5; // Điều chỉnh giá trị này để thay đổi mức độ sắc nét
        // Dst = unsharpMask(Dst, sharpAmount);
        
        // Giảm độ sáng
        double alpha = 1; // Điều chỉnh giá trị này để thay đổi độ sáng (< 1.0 để giảm, > 1.0 để tăng)
        Dst.convertTo(Dst, -1, alpha, 0);
        
        // Ghi khung hình vào video đầu ra
        writer.write(Dst);
    }
    
    cap.release();
    writer.release();
    std::cout << "Video processing completed and saved to result_ccm_video" << std::endl;
    
    return 0;
}
