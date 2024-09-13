#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "classification.hpp"
#include "parkingSpot.hpp"

/**
 * Classifies a sequence of images by processing segmentation masks and stores the classified results in the output vector.
 *
 * @param parkingSpot            A vector of ParkingSpot objects representing the parking spaces.
 * @param segmentationMasks      A vector of cv::Mat representing the segmentation masks for each image.
 * @param classifiedMasks        A reference to a vector of cv::Mat where the classified results will be stored.
 */
void classifySequence(std::vector<ParkingSpot> parkingSpot, std::vector<cv::Mat> segmentationMasks, std::vector<cv::Mat>& classifiedMasks) {
    for (int i = 0; i < segmentationMasks.size(); i++) {
        cv::Mat classifiedMask;
        classifyImage(parkingSpot, segmentationMasks[i], classifiedMask);
        classifiedMasks.push_back(classifiedMask);
    }
}

/**
 * Classifies a single image by processing the segmentation mask and marking parking spots as occupied or not.
 *
 * @param parkingSpot            A vector of ParkingSpot objects representing the parking spaces.
 * @param segmentationMask       A cv::Mat representing the segmentation mask for the image.
 * @param classifiedMask         A reference to a cv::Mat where the classified output will be stored.
 */
void classifyImage(std::vector<ParkingSpot> parkingSpot, cv::Mat segmentationMask, cv::Mat& classifiedMask) {
    // Extract connected components only once
    cv::Mat labels, stats, centroids;
    int numComponents = cv::connectedComponentsWithStats(segmentationMask, labels, stats, centroids);
    
    // Initialize the output mask with the segmentation mask
    segmentationMask.copyTo(classifiedMask);  
    
    // Loop through all parking spaces
    for (int i = 0; i < parkingSpot.size(); i++) {
        // Loop through all connected components, skipping the background(label 0) starting from 1 
        for (int j = 1; j < numComponents; j++) {  
            calculateComponentInsideRotatedRect(labels, stats, classifiedMask, parkingSpot[i], j); 
        }
    }
}

/**
 * Calculates if a connected component lies within a parking spot's rotated rectangle.
 * Updates the output mask based on whether the component is inside or outside the spot.
 *
 * @param labels               A cv::Mat representing the connected component labels.
 * @param stats                A cv::Mat containing the statistics of the connected components.
 * @param classifiedMask       A cv::Mat where the updated classification is stored.
 * @param parkingSpot          A reference to a ParkingSpot object representing the parking spot.
 * @param componentLabel       An integer representing the label of the connected component.
 */
void calculateComponentInsideRotatedRect(const cv::Mat& labels, const cv::Mat& stats, cv::Mat& classifiedMask, ParkingSpot& parkingSpot, int componentLabel) {
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
        changeComponentValue(labels, classifiedMask, componentLabel, labelId::carOutsideParkingSpot);
    } 
    else {
        changeComponentValue(labels, classifiedMask, componentLabel, labelId::carInsideParkingSpot);

        // Set the parking spot as occupied
        parkingSpot.occupied = true;
    }
}

/**
 * Changes the value of all pixels in a specific connected component in the output mask.
 *
 * @param labels                 A cv::Mat representing the connected component labels.
 * @param classifiedMask         A cv::Mat where the component's new value will be written.
 * @param componentLabel         An integer representing the label of the connected component.
 * @param labelId                The label ID to assign to the component in the mask
 * .
 */
void changeComponentValue(const cv::Mat& labels, cv::Mat& classifiedMask, int componentLabel, labelId labelId) {
    // Ensure the labels are in the right format
    CV_Assert(labels.type() == CV_32SC1);

    // Change the value of all pixels in the desired component
    for (int i = 0; i < labels.rows; ++i) {
        for (int j = 0; j < labels.cols; ++j) {
            if (labels.at<int>(i, j) == componentLabel) {
                classifiedMask.at<uchar>(i, j) = uchar(labelId);
            }
        }
    }
}
