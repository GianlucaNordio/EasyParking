#include "segmentation.hpp"

#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

Segmentation::Segmentation(const std::vector<cv::Mat> &backgroundImages) {
    // Build the background model
    // pBackSub = cv::createBackgroundSubtractorKNN();
    pBackSub = cv::createBackgroundSubtractorMOG2();
    cv::Mat mask;
    for(int i = 0; i < backgroundImages.size(); i++) {
        pBackSub -> apply(backgroundImages[i], mask);
    }
}

void Segmentation::segmentImage(const cv::Mat &image, cv::Mat &outputMask) {
    pBackSub -> apply(image, outputMask, BACKGROUND_NOT_UPDATED);
    //cv::blur(outputMask, outputMask, cv::Size(15, 15), cv::Point(-1, -1));
    // Remove the shadow parts and the noise
    //cv::threshold(outputMask, outputMask, 128, 255, cv::THRESH_BINARY);
}

void Segmentation::segmentVectorImages(const std::vector<cv::Mat> &images, std::vector<cv::Mat> &outputMasks) {
    // Apply segmentImage to each element of the vector separately
    for(int i = 0; i < images.size(); i++) {
        outputMasks.push_back(cv::Mat());
        segmentImage(images[i], outputMasks[i]);
    }
}