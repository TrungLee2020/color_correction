#include <opencv4/opencv2/opencv.hpp>
#include <iostream>
#include <cmath>
#include <chrono>
#include <filesystem>
#include <string>

#include "mylib/hsl.hpp"

using namespace cv;

void process_video(const std::string& input_path, const std::string& output_path, double hue, double saturation, double lightness) {
    cv::VideoCapture cap(input_path);
    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open the video file." << std::endl;
        return;
    }

    int frame_width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int frame_height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    double fps = cap.get(cv::CAP_PROP_FPS);

    cv::VideoWriter writer(output_path, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), fps, cv::Size(frame_width, frame_height));

    cv::Mat frame;
    while (cap.read(frame)) {
        cv::Mat adjusted_yello2frame = adjust_hsl_yellow_frame(frame, hue, saturation, lightness);
        cv::Mat adjusted_frame = adjust_hsl_green_frame(adjusted_yello2frame, hue, saturation, lightness);
        writer.write(adjusted_frame);
    }

    cap.release();
    writer.release();

    std::cout << "Processed video saved at: " << output_path << std::endl;
}


int main() {
    std::string input_path = "original_videos/am_vang/28.mp4";
    std::string output_path = "result_hsl_video/am_vang/28.mp4";

    auto start = std::chrono::high_resolution_clock::now();
    std::cout << "Processing video: " << input_path << std::endl;
    cv::VideoCapture cap(input_path);
    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open the video file." << std::endl;
        return -1;
    }

    int frame_width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int frame_height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    double fps = cap.get(cv::CAP_PROP_FPS);

    cv::VideoWriter writer(output_path, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), fps, cv::Size(frame_width, frame_height));

    cv::Mat frame;
    while (cap.read(frame)) {
        cv::Mat adjusted_yello2frame = adjust_hsl_yellow_frame(frame, 0, -40, 30);
        cv::Mat adjusted_frame = adjust_hsl_green_frame(adjusted_yello2frame, 20, 40, -5);
        writer.write(adjusted_frame);
    }

    cap.release();
    writer.release();

    std::cout << "Processed video saved at: " << output_path << std::endl;


    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Processing time: " << duration.count() << " milliseconds" << std::endl;

    return 0;
}

// int main() {
//     auto start = std::chrono::high_resolution_clock::now();

//     std::string input_folder = "original_videos";
//     std::string output_folder = "result_hsl_video";

//     std::filesystem::create_directories(output_folder);

//     for (const auto & entry : std::filesystem::directory_iterator(input_folder)) {
//         if (entry.is_regular_file() && entry.path().extension() == ".mp4") {
//             std::string input_path = entry.path().string();
//             std::string file_name = entry.path().filename().string();
//             std::string output_path = output_folder + "/" + file_name;

//             std::cout << "Processing video: " << file_name << std::endl;

//             process_video(input_path, output_path, 0, -40, 30);
//         }
//     }

//     auto end = std::chrono::high_resolution_clock::now();
//     auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
//     std::cout << "Processing time: " << duration.count() << " milliseconds" << std::endl;

//     return 0;
// }