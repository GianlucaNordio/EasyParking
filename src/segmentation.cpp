#include "segmentation.hpp"

#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>


Segmentation::Segmentation(const std::vector<cv::Mat> &backgroundImages) {
    // Build the background model
    //pBackSub = cv::createBackgroundSubtractorKNN();

    //pBackSub = new BackgroundSubtractorMOG();
    const int HISTORY_DEFAULT_VALUE = 500;
    const bool SHADES_DETECTION = true; 
    const int VAR_THRESHOLD = 80;
    pBackSub = cv::createBackgroundSubtractorMOG2(HISTORY_DEFAULT_VALUE, VAR_THRESHOLD, SHADES_DETECTION);
    cv::Mat mask;
    for(int i = 0; i < backgroundImages.size(); i++) {
        pBackSub -> apply(backgroundImages[i], mask);
    }
}

void Segmentation::segmentImage(const cv::Mat &image, cv::Mat &outputMask) {
    const int PIXEL_SIZE_THRESHOLD = 700;   // Threshold for the minumum size of the connected component to be kept
    const int CONNECTIVITY_8 = 8;   // 8-connectivity for connectedComponentsWithStats
    const int MORPH_RECT = cv::MORPH_RECT;  // Rectangular structuring element for morphologyEx
    const int MORPH_SIZE = 2;   // Size of structuring element for closing

    pBackSub -> apply(image, outputMask, BACKGROUND_NOT_UPDATED);
    
    // Remove the shadow parts and the noise 
    //(tried using the noShade option in the constructor but still finds shades, simply doesn't differentiate them)
    cv::threshold(outputMask, outputMask, 128, 255, cv::THRESH_BINARY);

    // Compute connected components and their statistics
    cv::Mat stats, centroids, labelImage;
    int numLabels = cv::connectedComponentsWithStats(outputMask, labelImage, stats, centroids, CONNECTIVITY_8, CV_32S);

    // Create a mask for large connected components
    cv::Mat mask = cv::Mat::zeros(labelImage.size(), CV_8UC1);

    // Filter components based on area size threshold
    for (int i = 1; i < numLabels; ++i) {
        int area = stats.at<int>(i, cv::CC_STAT_AREA);
        if (area > PIXEL_SIZE_THRESHOLD) {
            mask = mask | (labelImage == i);
        }
    }

    // Perform morphological closing to refine the mask (remove white lines)
    cv::Mat element = cv::getStructuringElement(MORPH_RECT, 
        cv::Size(2 * MORPH_SIZE + 1, 2 * MORPH_SIZE + 1), 
        cv::Point(MORPH_SIZE, MORPH_SIZE));

    //cv::Mat closedMask;
    //cv::morphologyEx(mask, closedMask, cv::MORPH_OPEN, element);

    // Apply the refined mask to the input image
    image.copyTo(outputMask, mask); //TODO use closedMask instead of mask

}

void Segmentation::segmentVectorImages(const std::vector<cv::Mat> &images, std::vector<cv::Mat> &outputMasks) {
    // Apply segmentImage to each element of the vector separately
    for(int i = 0; i < images.size(); i++) {
        outputMasks.push_back(cv::Mat());
        segmentImage(images[i], outputMasks[i]);
    }
}