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
    // Tried using the noShade option in the constructor but still finds shades
    cv::threshold(outputMask, outputMask, 128, 255, cv::THRESH_BINARY);

    // Currently testing with multiple values for the thresholds
    for(int k = 20; k < 3000; k+=200) {
        // Keeping only connected that have a dimension bigger than 2000 pixels
        cv::Mat stats, centroids, labelImage;
        int numLabels = cv::connectedComponentsWithStats(outputMask, labelImage, stats, centroids, 8, CV_32S);
        cv::Mat mask(labelImage.size(), CV_8UC1, cv::Scalar(0));
        cv::Mat surfSup=stats.col(4) > k;
        for (int i = 1; i < numLabels; i++) {
            if (surfSup.at<uchar>(i, 0)) {
                mask = mask | (labelImage==i);
            }
        }
        cv::Mat r(outputMask.size(), CV_8UC1, cv::Scalar(0));
        image.copyTo(r,mask);
        imshow("Result", r);
        cv::waitKey(500);
        std::cout<<"Area " << k<<std::endl;
    }
}

void Segmentation::segmentVectorImages(const std::vector<cv::Mat> &images, std::vector<cv::Mat> &outputMasks) {
    // Apply segmentImage to each element of the vector separately
    for(int i = 0; i < images.size(); i++) {
        outputMasks.push_back(cv::Mat());
        segmentImage(images[i], outputMasks[i]);
    }
}