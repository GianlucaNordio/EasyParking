#include "classification.hpp"
#include "parkingSpot.hpp"

#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>


cv::Mat classifyCars(std::vector<ParkingSpot> spaces, cv::Mat segmentationMasks) {
    // Extract connected components
    cv::Mat labels, stats, centroids;
    int numComponents = cv::connectedComponentsWithStats(segmentationMasks, labels, stats, centroids);
    cv::Mat output;
    segmentationMasks.copyTo(output);  // Initialize output mask
    
    for (int i = 0; i < spaces.size(); i++) {
        for (int j = 0; j < numComponents; j++) {
            calculateComponentInsideRotatedRect(segmentationMasks, output, spaces[i].rect, j); 
        }
    }

    return output;
}

float calculateComponentInsideRotatedRect(const cv::Mat& mask, cv::Mat& output, const cv::RotatedRect& rotatedRect, int componentLabel) {
    // Ensure the mask is a binary image
    CV_Assert(mask.type() == CV_8UC1);

    cv::Mat labels, stats, centroids;
    int numComponents = cv::connectedComponentsWithStats(mask, labels, stats, centroids);

    if (componentLabel >= numComponents) {
        std::cerr << "Component label is out of range!" << std::endl;
        return -1.0f;
    }

    // Get stats for the component: [left, top, width, height, area]
    int x = stats.at<int>(componentLabel, cv::CC_STAT_LEFT);
    int y = stats.at<int>(componentLabel, cv::CC_STAT_TOP);
    int w = stats.at<int>(componentLabel, cv::CC_STAT_WIDTH);
    int h = stats.at<int>(componentLabel, cv::CC_STAT_HEIGHT);
    int componentArea = stats.at<int>(componentLabel, cv::CC_STAT_AREA);

    // Create a mask for the rotatedRect
    cv::Mat rotatedRectMask = cv::Mat::zeros(mask.size(), CV_8UC1);

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
        changeComponentValue(output, componentLabel, ID_CAR_OUTSIDE_PARKING_LOT);
    } else {
        changeComponentValue(output, componentLabel, ID_CAR_INSIDE_PARKING_LOT);
    }
    
    return percentageInside;
}

void changeComponentValue(cv::Mat& mask, int componentLabel, uchar newValue) {
    // Ensure the mask is a single-channel binary image
    CV_Assert(mask.type() == CV_8UC1);

    // Extract connected components
    cv::Mat labels, stats, centroids;
    int numComponents = cv::connectedComponentsWithStats(mask, labels, stats, centroids);

    if (componentLabel >= numComponents) {
        std::cerr << "Component label is out of range!" << std::endl;
        return;
    }

    // Change the value of all pixels in the desired component
    for (int i = 0; i < labels.rows; ++i) {
        for (int j = 0; j < labels.cols; ++j) {
            if (labels.at<int>(i, j) == componentLabel) {
                mask.at<uchar>(i, j) = newValue;
            }
        }
    }
}
