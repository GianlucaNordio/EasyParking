#include <iostream>
#include <opencv2/highgui.hpp>
#include <filesystem>

int main() {
    std::cout << "Hello, World!" <<std::filesystem::current_path()  << std::endl;
    
    cv::Mat img = cv::imread("../src/test.jpg");
    cv::namedWindow("Test image");
    cv::imshow("Test image", img);
    cv::waitKey(0);
    return 0;
}