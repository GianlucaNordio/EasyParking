#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "classification.hpp"
#include "parkingSpot.hpp"

/**
 * Classifies a sequence of images by processing segmentation masks and stores the classified results in the output vector.
 *
 * @param parkingSpot            A vector of vector of ParkingSpot objects representing the parking spaces.
 * @param segmentationMasks      A vector of cv::Mat representing the segmentation masks for each image.
 * @param classifiedMasks        A reference to a vector of cv::Mat where the classified results will be stored.
 */
void classifySequence(std::vector<std::vector<ParkingSpot>> parkingSpot, std::vector<cv::Mat> segmentationMasks, std::vector<cv::Mat>& classifiedMasks) {
    for (int i = 0; i < segmentationMasks.size(); i++) {
        cv::Mat classifiedMask;
        classifyImage(parkingSpot[i], segmentationMasks[i], classifiedMask);
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
    cv::Mat labels, stats, centroids;
    int numComponents = cv::connectedComponentsWithStats(segmentationMask, labels, stats, centroids);

    classifiedMask = cv::Mat::zeros(segmentationMask.size(), CV_8UC1); // Initialize with zeros

    // Create a mask for each parking spot in parallel
    for (auto& spot : parkingSpot) {
        cv::Mat spotMask = cv::Mat::zeros(segmentationMask.size(), CV_8UC1);
        cv::Point2f vertices[4];
        spot.rect.points(vertices);
        std::vector<cv::Point> contour(vertices, vertices + 4);
        cv::fillConvexPoly(spotMask, contour, cv::Scalar(255));

        // Process each component
        for (int j = 1; j < numComponents; j++) {
            cv::Mat componentMask = (labels == j);
            cv::Mat intersectionMask;
            cv::bitwise_and(spotMask, componentMask, intersectionMask);

            int componentArea = stats.at<int>(j, cv::CC_STAT_AREA);
            int insidePixels = cv::countNonZero(intersectionMask);
            float percentageInside = (static_cast<float>(insidePixels) / componentArea) * 100.0f;

            std::cout << "Component " << j << " is " << percentageInside << "% inside the parking spot." << std::endl;

            // Update classifiedMask based on percentage inside
            if ((1 - percentageInside) > PERCENTAGE_OUTSIDE_THRESHOLD) {
                classifiedMask.setTo(labelId::carOutsideParkingSpot, componentMask);
            } else {
                classifiedMask.setTo(labelId::carInsideParkingSpot, componentMask);
                spot.occupied = true;
            }
        }

        cv::Mat mostra = classifiedMask.clone();
        std::vector<cv::Mat> channels;
        channels.push_back(mostra);
        cv::Mat result = cv::Mat::zeros(segmentationMask.size(), CV_8UC3);
        std::vector<cv::Mat> channel;
        channel.push_back(result);
        convertGreyMaskToBGR(channels, channel);
        cv::imshow("Classified Mask", channel[0]);
        cv::waitKey(0);
    }
}