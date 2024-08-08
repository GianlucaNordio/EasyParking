#include "segmentation.hpp"

#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

void test(std::vector<cv::Mat> &bck, cv::Mat img){
    //cv::Ptr<cv::BackgroundSubtractor> pBackSub = cv::createBackgroundSubtractorKNN();
    cv::Ptr<cv::BackgroundSubtractor> pBackSub = cv::createBackgroundSubtractorMOG2();
    cv::Mat mask;
    for(int i = 0; i < bck.size(); i++) {
        pBackSub->apply(bck[i], mask);
    }
    pBackSub->apply(img, mask);

    cv::imshow("Mask", mask);
    cv::imshow("Original", img);
    cv::waitKey();
}