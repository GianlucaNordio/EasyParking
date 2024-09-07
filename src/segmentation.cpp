#include "segmentation.hpp"

#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

const int PIXEL_SIZE_THRESHOLD = 1000;

Segmentation::Segmentation(const std::vector<cv::Mat> &backgroundImages) {
    // Build the background model
    // pBackSub = cv::createBackgroundSubtractorKNN();
    pBackSub = cv::createBackgroundSubtractorMOG2(59, 700, true);
    cv::Mat mask;
    for(int i = 0; i < backgroundImages.size(); i++) {
        // Convert the image from BGR to HSV color space
        cv::Mat hsvImage;
        cv::cvtColor(backgroundImages[i], hsvImage, cv::COLOR_BGR2HSV);

        // Split the HSV channels into three separate images
        std::vector<cv::Mat> hsvChannels(3);
        cv::split(hsvImage, hsvChannels);

        // Perform histogram equalization on each channel separately
        for (int i = 0; i <= 2; ++i) {
            cv::equalizeHist(hsvChannels[i], hsvChannels[i]);
        }

        // Merge the equalized channels back together
        cv::Mat equalizedHSV;
        cv::merge(hsvChannels, equalizedHSV);

        // Convert the equalized HSV image back to BGR
        cv::Mat resultImage;
        cv::cvtColor(equalizedHSV, resultImage, cv::COLOR_HSV2BGR);

        // Display the original and processed images
        cv::imshow("Original Image", backgroundImages[i]);
        cv::imshow("Equalized Image", resultImage);

        // Wait for a key press indefinitely
        cv::waitKey(0);
        pBackSub -> apply(resultImage, mask);
    }
}

void Segmentation::segmentImage(const cv::Mat &image, cv::Mat &outputMask) {
    // Convert the image from BGR to HSV color space
    cv::Mat hsvImage;
    cv::cvtColor(image, hsvImage, cv::COLOR_BGR2HSV);

    // Split the HSV channels into three separate images
    std::vector<cv::Mat> hsvChannels(3);
    cv::split(hsvImage, hsvChannels);

    // Perform histogram equalization on each channel separately
    for (int i = 0; i <= 2; ++i) {
        //cv::imshow("channels", hsvChannels[i]);
        //cv::waitKey();
        cv::equalizeHist(hsvChannels[i], hsvChannels[i]);
    }

    // Merge the equalized channels back together
    cv::Mat equalizedHSV;
    cv::merge(hsvChannels, equalizedHSV);

    // Convert the equalized HSV image back to BGR
    cv::Mat resultImage;
    cv::cvtColor(equalizedHSV, resultImage, cv::COLOR_HSV2BGR);

    // Display the original and processed images
    pBackSub -> apply(resultImage, outputMask, BACKGROUND_NOT_UPDATED);
    
    // Remove the shadow parts and the noise
    // Tried using the noShade option in the constructor but still finds shades
    cv::threshold(outputMask, outputMask, 128, 255, cv::THRESH_BINARY);

    // Currently testing with multiple values for the thresholds
    //for(int k = 20; k < 3000; k+=200) {

    // Keeping only connected that have a dimension bigger than 600 pixels
    // Improved code for extracting large connected components

    // Define constants for better readability
    const int CONNECTIVITY_8 = 8;              // 8-connectivity for connectedComponentsWithStats
    const int MORPH_RECT = cv::MORPH_CROSS;     // Rectangular structuring element for morphologyEx
    const int MORPH_SIZE = 1;                // Size of structuring element for closing

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

    // Perform morphological closing to refine the mask
    cv::Mat element = cv::getStructuringElement(MORPH_RECT, 
        cv::Size(2 * MORPH_SIZE + 1, 2 * MORPH_SIZE + 1), 
        cv::Point(MORPH_SIZE, MORPH_SIZE));

    cv::Mat closedMask;
    cv::morphologyEx(mask, closedMask, cv::MORPH_CLOSE, element);

    // Apply the refined mask to the input image
    image.copyTo(outputMask, closedMask);

}

void Segmentation::segmentVectorImages(const std::vector<cv::Mat> &images, std::vector<cv::Mat> &outputMasks) {
    // Apply segmentImage to each element of the vector separately
    for(int i = 0; i < images.size(); i++) {
        outputMasks.push_back(cv::Mat());
        segmentImage(images[i], outputMasks[i]);
    }
}