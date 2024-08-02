#include <iostream>
#include <opencv2/opencv.hpp>
#include <fstream>
#include <vector>
#include <chrono>
#include <filesystem>
#include <string>

using namespace std;
namespace fs = std::filesystem;

cv::Mat readColorCorrectionMatrix(const std::string& filename) {
    std::ifstream CMC(filename, std::ios::in);
    
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

cv::Mat applyColorCorrection(const cv::Mat& img, const cv::Mat& ColorMatrix, double zoom_factor) {
    cv::Mat Dst = img.clone();
    int channels = img.channels();
    int ImgHeight = img.rows, ImgWidth = img.cols;

    const float *CMC_1 = ColorMatrix.ptr<float>(0);
    const float *CMC_2 = ColorMatrix.ptr<float>(1);
    const float *CMC_3 = ColorMatrix.ptr<float>(2);
    
    // Điều chỉnh hiệu ứng CCM dựa trên zoom_factor
    double enhancement = std::min(zoom_factor - 1.0, 1.0);  // Giới hạn tăng cường
    
    for (int i = 0; i < ImgHeight; ++i) {
        const uchar *SP = img.ptr<uchar>(i);
        uchar *DP = Dst.ptr<uchar>(i);
        for (int j = 0; j < ImgWidth*channels; j += 3) {
            // Áp dụng ma trận CCM
            double b = SP[j] * CMC_1[0] + SP[j + 1] * CMC_2[0] + SP[j+2] * CMC_3[0];
            double g = SP[j] * CMC_1[1] + SP[j + 1] * CMC_2[1] + SP[j+2] * CMC_3[1];
            double r = SP[j] * CMC_1[2] + SP[j + 1] * CMC_2[2] + SP[j+2] * CMC_3[2];
            
            // Áp dụng hiệu ứng tăng cường dựa trên zoom_factor
            DP[j] = cv::saturate_cast<uchar>(SP[j] * (1.0 - enhancement) + b * enhancement);
            DP[j+1] = cv::saturate_cast<uchar>(SP[j+1] * (1.0 - enhancement) + g * enhancement);
            DP[j+2] = cv::saturate_cast<uchar>(SP[j+2] * (1.0 - enhancement) + r * enhancement);
        }
    }
    // double alpha = 0.95; // Điều chỉnh giá trị này để thay đổi độ sáng (< 1.0 để giảm, > 1.0 để tăng)
    // Dst.convertTo(Dst, -1, alpha, 0);
    return Dst;
}

void processVideo(const std::string& inputVideo, const std::string& outputVideo, const std::string& cmcFile) {
    cv::VideoCapture cap(inputVideo);
    if (!cap.isOpened()) {
        std::cerr << "Error opening video file" << std::endl;
        return;
    }

    cv::Mat ColorMatrix = readColorCorrectionMatrix(cmcFile);

    int frame_width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
    int frame_height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    int fps = cap.get(cv::CAP_PROP_FPS);

    cv::VideoWriter video(outputVideo, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), fps, cv::Size(frame_width, frame_height));

    cv::Mat frame;
    double base_width = frame_width;
    while (cap.read(frame)) {
        // Tính toán zoom factor
        double zoom_factor = static_cast<double>(frame.cols) / base_width;
        
        cv::Mat corrected = applyColorCorrection(frame, ColorMatrix, zoom_factor);
        video.write(corrected);
    }

    cap.release();
    video.release();
}
int main() {
    auto start = std::chrono::high_resolution_clock::now();
    std::cout << "Start processing apply CCM to video." << std::endl;
    std::string inputVideo = "original_videos/vach_ke_duong/41_.mp4";
    std::string outputVideo = "result_ccm_video/vach_ke_duong/41_.mp4";
    std::string cmcFile = "ref/LCC_CMC.csv";

    processVideo(inputVideo, outputVideo, cmcFile);

    std::cout << "Video processing apply CCM to video completed." << std::endl;

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Processing time: " << duration.count() << " milliseconds" << std::endl;
    return 0;
}