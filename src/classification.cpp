#include "classification.hpp"
#include "parkingSpot.hpp"

#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>


void classifySequence(std::vector<ParkingSpot> spaces, std::vector<cv::Mat> segmentationMasks, std::vector<cv::Mat>& output) {
    for (int i = 0; i < segmentationMasks.size(); i++) {
        cv::Mat classifiedMask;
        classifyImage(spaces, segmentationMasks[i], classifiedMask);
        output.push_back(classifiedMask);
    }

}

void classifyImage(std::vector<ParkingSpot> spaces, cv::Mat segmentationMasks, cv::Mat& output) {
    // Extract connected components only once
    cv::Mat labels, stats, centroids;
    int numComponents = cv::connectedComponentsWithStats(segmentationMasks, labels, stats, centroids);
    segmentationMasks.copyTo(output);  // Initialize output mask
    
    // Loop through all parking spaces
    for (int i = 0; i < spaces.size(); i++) {
        // Loop through all connected components, skipping the background (label 0)
        for (int j = 1; j < numComponents; j++) {  // Start from 1 to skip the background
            calculateComponentInsideRotatedRect(labels, stats, output, spaces[i], j); 
        }
    }
}

void calculateComponentInsideRotatedRect(const cv::Mat& labels, const cv::Mat& stats, cv::Mat& output, ParkingSpot& parkingSpot, int componentLabel) {
    // Ensure the labels are in the right format
    CV_Assert(labels.type() == CV_32SC1);

    // Get stats for the component: [left, top, width, height, area]
    int x = stats.at<int>(componentLabel, cv::CC_STAT_LEFT);
    int y = stats.at<int>(componentLabel, cv::CC_STAT_TOP);
    int w = stats.at<int>(componentLabel, cv::CC_STAT_WIDTH);
    int h = stats.at<int>(componentLabel, cv::CC_STAT_HEIGHT);
    int componentArea = stats.at<int>(componentLabel, cv::CC_STAT_AREA);

    // Create a mask for the rotatedRect
    cv::Mat rotatedRectMask = cv::Mat::zeros(labels.size(), CV_8UC1);

    cv::RotatedRect rotatedRect = parkingSpot.rect;

    // Convert rotatedRect to a set of 4 points
    cv::Point2f vertices[4];
    rotatedRect.points(vertices);
    
    // Create a filled polygon representing the rotated rectangle in the mask
    std::vector<cv::Point> contour(vertices, vertices + 4);
    cv::fillConvexPoly(rotatedRectMask, contour, cv::Scalar(255));

    // Create a mask for the component
    cv::Mat componentMask = (labels == componentLabel);

    // Calculate how many pixels of the component are inside the rotatedRect
    cv::Mat intersectionMask;
    cv::bitwise_and(rotatedRectMask, componentMask, intersectionMask);

    int insidePixels = cv::countNonZero(intersectionMask);

    // Calculate the percentage of the component inside the rotated rectangle
    float percentageInside = (static_cast<float>(insidePixels) / componentArea) * 100.0f;
    float percentageOutside = 1.0f - percentageInside;

    // Set component label based on percentage outside
    if (percentageOutside > PERCENTAGE_OUTSIDE_THRESHOLD) {
        changeComponentValue(labels, output, componentLabel, labelId::carOutsideParkingSpot);
    } else {
        changeComponentValue(labels, output, componentLabel, labelId::carInsideParkingSpot);

        // Set the parking spot as occupied
        parkingSpot.occupied = true;
    }
}

void changeComponentValue(const cv::Mat& labels, cv::Mat& mask, int componentLabel, labelId labelId) {
    // Ensure the labels are in the right format
    CV_Assert(labels.type() == CV_32SC1);

    // Change the value of all pixels in the desired component
    for (int i = 0; i < labels.rows; ++i) {
        for (int j = 0; j < labels.cols; ++j) {
            if (labels.at<int>(i, j) == componentLabel) {
                mask.at<uchar>(i, j) = uchar(labelId);
            }
        }
    }
}
