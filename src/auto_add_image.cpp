#include <opencv4/opencv2/opencv.hpp>
#include <iostream>
#include <cmath>
#include <vector>
#include "hsl.hpp"
#include <string>


using namespace cv;
using namespace std;

// ... (Giữ nguyên các hàm rgb_to_hsl và hsl_to_rgb như bạn đã cung cấp)

// Hàm mới để tính toán thống kê HSL của ảnh
void calculateHSLStats(const Mat& img, vector<double>& meanHSL, vector<double>& stdDevHSL) {
    vector<double> hValues, sValues, lValues;
    for (int y = 0; y < img.rows; y++) {
        for (int x = 0; x < img.cols; x++) {
            Vec3b pixel = img.at<Vec3b>(y, x);
            HSL hsl = rgb_to_hsl(pixel[2], pixel[1], pixel[0]);  // OpenCV uses BGR
            hValues.push_back(hsl.h);
            sValues.push_back(hsl.s);
            lValues.push_back(hsl.l);
        }
    }

    auto calculateMeanStdDev = [](const vector<double>& values) {
        double sum = accumulate(values.begin(), values.end(), 0.0);
        double mean = sum / values.size();
        double sqSum = inner_product(values.begin(), values.end(), values.begin(), 0.0);
        double stdDev = sqrt(sqSum / values.size() - mean * mean);
        return make_pair(mean, stdDev);
    };

    auto [hMean, hStdDev] = calculateMeanStdDev(hValues);
    auto [sMean, sStdDev] = calculateMeanStdDev(sValues);
    auto [lMean, lStdDev] = calculateMeanStdDev(lValues);

    meanHSL = {hMean, sMean, lMean};
    stdDevHSL = {hStdDev, sStdDev, lStdDev};
}

// Hàm mới để xác định các điều chỉnh HSL dựa trên thống kê
void determineHSLAdjustments(const vector<double>& meanHSL, const vector<double>& stdDevHSL, 
                             int& hAdjust, int& sAdjust, int& lAdjust) {
    // Hue: Thường không điều chỉnh tự động
    hAdjust = 0;

    // Saturation: Điều chỉnh để đưa về mức trung bình nếu quá cao hoặc quá thấp
    if (meanHSL[1] < 30) {
        sAdjust = min(50, int((30 - meanHSL[1]) / 2));
    } else if (meanHSL[1] > 70) {
        sAdjust = max(-50, int((70 - meanHSL[1]) / 2));
    } else {
        sAdjust = 0;
    }

    // Lightness: Điều chỉnh để đưa về mức trung bình nếu quá tối hoặc quá sáng
    if (meanHSL[2] < 40) {
        lAdjust = min(30, int((40 - meanHSL[2]) / 2));
    } else if (meanHSL[2] > 60) {
        lAdjust = max(-30, int((60 - meanHSL[2]) / 2));
    } else {
        lAdjust = 0;
    }
}

Mat autoAdjustHSL(const Mat& img, const string& output_path) {
    vector<double> meanHSL, stdDevHSL;
    calculateHSLStats(img, meanHSL, stdDevHSL);

    int hAdjust, sAdjust, lAdjust;
    determineHSLAdjustments(meanHSL, stdDevHSL, hAdjust, sAdjust, lAdjust);

    cout << "Automatic adjustments: H: " << hAdjust 
         << ", S: " << sAdjust << ", L: " << lAdjust << endl;

    return adjust_hsl(img, hAdjust, sAdjust, lAdjust, output_path);
}

int main() {

    string input_path = "data/1df7752f386e9d30c47f.jpg";
    string output_path = "result_auto/output_image.jpg";

    cv::Mat img = imread(input_path);
    if (img.empty()) {
        cerr << "Error: Could not read the image." << endl;
        return -1;
    }

    cv::Mat result = autoAdjustHSL(img, output_path);

    if (!result.empty()) {
        cout << "Image processed successfully." << endl;
    } else {
        cerr << "Error occurred during processing." << endl;
    }

    return 0;
}