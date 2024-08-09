#include "segmentation.hpp"

#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

Segmentation::Segmentation(const std::vector<cv::Mat> &backgroundImages) {
    // Build the background model
     pBackSub = cv::createBackgroundSubtractorKNN();
    //pBackSub = cv::createBackgroundSubtractorMOG2();
    cv::Mat mask;
    for(int i = 0; i < backgroundImages.size(); i++) {
        cv::Mat equalized;
        cv::cvtColor(backgroundImages[i], equalized, cv::COLOR_BGR2GRAY);
        cv::equalizeHist(equalized, equalized);
        pBackSub -> apply(equalized, mask);
    }
}

void Segmentation::segmentImage(const cv::Mat &image, cv::Mat &outputMask) {
    cv::Mat equalized;
    cv::cvtColor(image, equalized, cv::COLOR_BGR2GRAY);
    cv::equalizeHist(equalized, equalized);
    pBackSub -> apply(image, outputMask, BACKGROUND_NOT_UPDATED);
}

void Segmentation::segmentVectorImages(const std::vector<cv::Mat> &images, std::vector<cv::Mat> &outputMasks) {
    // Apply segmentImage to each element of the vector separately
    for(int i = 0; i < images.size(); i++) {
        outputMasks.push_back(cv::Mat());
        segmentImage(images[i], outputMasks[i]);
    }
}