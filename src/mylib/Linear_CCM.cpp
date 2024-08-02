#include <iostream>
#include "Linear_CCM.hpp"
#include <opencv4/opencv2/opencv.hpp>
#include <opencv4/opencv2/core.hpp>
#include <opencv4/opencv2/highgui.hpp>
#include <opencv4/opencv2/imgproc.hpp>

using namespace cv;
// using namespace std;


void ROImouseEvent(int event, int x, int y, int flags, void *params)
{
    cv::Point *Pptr = (cv::Point*)params;
    
    if (event == EVENT_LBUTTONDOWN && Pptr[0].x == -1 && Pptr[0].y == -1)
    {
        Pptr[0].x = x;
        Pptr[0].y = y;
    }
    
    if (flags == EVENT_FLAG_LBUTTON)
    {
        
        Pptr[1].x = x;
        Pptr[1].y = y;
    }
    
    if (event == EVENT_LBUTTONUP && Pptr[2].x == -1 && Pptr[2].y == -1)
    {
        Pptr[2].x = x;
        Pptr[2].y = y;
    }
}

void ROISelection(cv::Mat &img, cv::Mat &OriginalColor)
{
    cv::Point *corners = new cv::Point[3];
    corners[0].x = corners[0].y = -1;
    corners[1].x = corners[1].y = -1;
    corners[2].x = corners[2].y = -1;
    
    bool downFlag = false, upFlag = false;
    int ROICount = 0;
    cv::namedWindow("ROI select", cv::WINDOW_NORMAL);
    cv::imshow("ROI select", img);
    
    while (cv::waitKey(1) != 27 && ROICount <24)
    {
        cv::setMouseCallback("ROI select", ROImouseEvent, corners);
        
        if (corners[0].x != -1 && corners[0].y != -1) { downFlag  = true; }
        if (corners[2].x != -1 && corners[2].y != -1) { upFlag  = true; }
        
        if (downFlag && !upFlag && corners[1].x != -1)
        {
            cv::Mat LocalImg = img.clone();
            cv::rectangle(LocalImg, corners[0], corners[1], cv::Scalar(0, 0, 0), 2);
            cv::imshow("ROI select", LocalImg);
        }
        
        if (downFlag && upFlag)
        {
            cv::Rect ROI;
            
            ROI.width = abs(corners[0].x - corners[2].x);
            ROI.height = abs(corners[0].y - corners[2].y);
            
            if(ROI.width < 5 && ROI.height <5)
            {
                std::cerr << "ROI size too small, please re-crop the ROI" << std::endl;
            }
            else
            {
                ROI.x = corners[0].x < corners[2].x ? corners[0].x : corners[2].x;
                ROI.y = corners[0].y < corners[2].y ? corners[0].y : corners[2].y;
                cv::Mat crop(img, ROI);
                
                cv::namedWindow("Current ROI", cv::WINDOW_NORMAL);
                cv::imshow("Current ROI", crop);
                cv::waitKey(500);  // Hiển thị ROI trong 500ms
                cv::destroyWindow("Current ROI");
                
                int ROISize = crop.cols * crop.rows;
                cv::Scalar CropSum = cv::sum(crop)/ROISize ;
                
                float *OPtr = OriginalColor.ptr<float>(ROICount);
                OPtr[0] = CropSum[0];
                OPtr[1] = CropSum[1];
                OPtr[2] = CropSum[2];
                OPtr[3] = CropSum[3];
                
                ROICount++;
            }
            
            corners[0].x = corners[0].y = -1;
            corners[1].x = corners[1].y = -1;
            corners[2].x = corners[2].y = -1;
            // corners[3].x = corners[3].y = -1;
            downFlag = upFlag  = false;
        }
    }
    cv::destroyWindow("ROI select");
    delete[] corners;
}
cv::Mat convertToLabSpace(const cv::Mat& image) {
    cv::Mat labImage;
    cv::cvtColor(image, labImage, cv::COLOR_BGR2Lab);
    return labImage;
}
cv::Mat convertToRGBSpace(const cv::Mat& labImage) {
    cv::Mat rgbImage;
    cv::cvtColor(labImage, rgbImage, cv::COLOR_Lab2BGR);
    return rgbImage;
}

// Hàm để tăng giá trị của một màu cụ thể
cv::Mat enhanceColor(const cv::Mat& image, const cv::Vec3b& color, float factor) {
    cv::Mat result = image.clone();
    for (int y = 0; y < result.rows; y++) {
        for (int x = 0; x < result.cols; x++) {
            cv::Vec3b& pixel = result.at<cv::Vec3b>(y, x);
            for (int c = 0; c < 3; c++) {
                if (std::abs(pixel[c] - color[c]) < 30) {  // Ngưỡng cho sự tương đồng màu
                    pixel[c] = cv::saturate_cast<uchar>(pixel[c] * factor);
                }
            }
        }
    }
    return result;
}
void LCC_CMC(cv::Mat &img)
{
    
    int i = 0, j = 0;
    
    cv::Mat ReferenceColor(24, 3, CV_32FC1, cv::Scalar(0));
    
    std::fstream infile;
    // gia tri bang mau tham chieu tu file ReferenceColor
    infile.open("ref/ReferenceColor.csv", std::ios::in);
    
    if (!infile)
    {
        std::cerr << "Open the reference color file error" << std::endl;
        exit(EXIT_FAILURE);
    }
    
    std::string textline;
    while (getline(infile, textline))
    {
        float *RefP = ReferenceColor.ptr<float>(i);
        j = 0;
        std::string::size_type pos = 0, prev_pos = 0;
        while ((pos = textline.find_first_of(',', pos)) != std::string::npos)
        {
            
            RefP[j++] = std::stof(textline.substr(prev_pos, pos - prev_pos));
            prev_pos = ++pos;
        }
        // Luu cac gia tri tu bang mau tham chieu vao RefP su dung func: ReferenceColor
        RefP[j++] = std::stof(textline.substr(prev_pos, pos - prev_pos));
        i++;
    }
    
    cv::Mat OriginalColor(24, 3, CV_32FC1, cv::Scalar(0));
    
    ROISelection(img, OriginalColor);
    // tinh toan ma tran hieu chinh mau
    // Sử dụng công thức: CCM = (O^T * O)^-1 * O^T * R Trong đó O là ma trận màu gốc, R là ma trận màu tham chiếu
    cv::Mat O_T = OriginalColor.t(); // chuyen vi ma tran O
    cv::Mat temp = O_T*OriginalColor; // O^T * O
    cv::Mat ColorMatrix(3, 3, CV_32FC1, cv::Scalar(0));
    ColorMatrix = temp.inv() * O_T * ReferenceColor; // nghich dao (O^T*O)  * ReferenceColor
    
    std::cout << "CCM da tinh xong.!" << std::endl;
    // Lưu CCM vào file
    std::ofstream outfile("./ref/LCC_CMC.csv");
    if (outfile)
    {
        for (i = 0; i < 3; ++i)
        {
            for (j = 0; j < 3; ++j)
            {
                outfile << ColorMatrix.at<float>(i, j) << ",";
            }
            outfile << "\n";
        }
        std::cout << "CCM đã được lưu vào file ./ref/LCC_CMC.csv" << std::endl;
    }
    else
    {
        std::cerr << "Không thể mở file để lưu CCM" << std::endl;
    }
    
}

// AP dung ma tran chinh mau
void LCC(cv::Mat &img,cv::Mat &Dst)
{
    int i = 0, j = 0;

    int channels = img.channels();
    int ImgHeight = img.rows, ImgWidth = img.cols;

    // printf("%d %d %d\n",channels, ImgWidth, ImgWidth*channels);

    std::fstream CMC("ref/LCC_CMC.csv", std::ios::in);
    
    if (!CMC)
    {
        std::cerr << "Open the file error" << std::endl;
        exit(EXIT_FAILURE);
    }
    
    cv::Mat ColorMatrix(3, 3, CV_32FC1, cv::Scalar(0));
    
    std::string textline;
    while (getline(CMC, textline))
    {
        std::string::size_type pos = 0, prev_pos = 0;
        float *CMCPtr = ColorMatrix.ptr<float>(i);
        j = 0;
        while ((pos = textline.find_first_of(',', pos)) != std::string::npos)
        {
            CMCPtr[j++] = std::stof(textline.substr(prev_pos, pos - prev_pos));
            prev_pos = ++pos;
        }
        i++;
    }
    CMC.close();
    
    
    Dst = img.clone();
    
    float *CMC_1 = ColorMatrix.ptr<float>(0);
    float *CMC_2 = ColorMatrix.ptr<float>(1);
    float *CMC_3 = ColorMatrix.ptr<float>(2);
    
    for (i = 0; i < ImgHeight; ++i)
    {
        uchar *SP = img.ptr<uchar>(i); // gia trij pixel tai anh goc
        uchar *DP = Dst.ptr<uchar>(i);
        for (j = 0; j < ImgWidth*channels; j += 3) // channels = 3 kenh mau
        {
            // saturate_cast: dam bao gia tri pixel nam trong [0;255]
            DP[j] = cv::saturate_cast<uchar>(SP[j] * CMC_1[0] + SP[j + 1] * CMC_2[0] + SP[j+2] * CMC_3[0]);
            DP[j+1] = cv::saturate_cast<uchar>(SP[j] * CMC_1[1] + SP[j + 1] * CMC_2[1] + SP[j+2] * CMC_3[1]);
            DP[j+2] = cv::saturate_cast<uchar>(SP[j] * CMC_1[2] + SP[j + 1] * CMC_2[2] + SP[j+2] * CMC_3[2]);
        }
    }
    
}
