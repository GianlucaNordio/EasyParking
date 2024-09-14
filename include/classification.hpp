#ifndef CLASSIFICATION_HPP
#define CLASSIFICATION_HPP

#include <opencv2/opencv.hpp>

#include "parkingSpot.hpp"
#include "utils.hpp"

/**
 * Enum representing different labels for components in an image.
 * These labels are used to classify regions in the segmentation masks.
 */
enum labelId
{
    /** Label for background regions. */
    background,

    /** Label for a car that is fully or mostly inside a parking spot. */
    carInsideParkingSpot,

    /** Label for a car that is fully or mostly outside a parking spot. */
    carOutsideParkingSpot
};

/**
 * A constant threshold used to determine if car is outside a parking spot.
 * If more than 50% of the component is outside the parking spot, it is classified as 'outside'.
 */
const float PERCENTAGE_OUTSIDE_THRESHOLD = 0.5;

/**
 * Classifies a sequence of parking spots by processing segmentation masks and 
 * stores the classified results in the output vector.
 *
 * @param parkingSpot            A vector of vector of ParkingSpot objects representing the parking spaces.
 * @param segmentationMasks      A vector of cv::Mat representing the segmentation masks for each image.
 * @param classifiedMasks        A reference to a vector of cv::Mat where the classified results will be stored.
 */
void classifySequence(std::vector<std::vector<ParkingSpot>> parkingSpot, std::vector<cv::Mat> segmentationMasks, std::vector<cv::Mat>& classifiedMasks);

/**
 * Classifies a single image by processing the segmentation mask and marking parking spots as occupied or not.
 *
 * @param parkingSpot            A vector of ParkingSpot objects representing the parking spaces.
 * @param segmentationMask       A cv::Mat representing the segmentation mask for the image.
 * @param classifiedMask         A reference to a cv::Mat where the classified output will be stored.
 */
void classifyImage(std::vector<ParkingSpot> parkingSpot, cv::Mat segmentationMask, cv::Mat& classifiedMask);

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
void calculateComponentInsideRotatedRect(const cv::Mat& labels, const cv::Mat& stats, cv::Mat& classifiedMask, ParkingSpot& rotatedRect, int componentLabel);

/**
 * Changes the value of all pixels in a specific connected component in the output mask.
 *
 * @param labels                 A cv::Mat representing the connected component labels.
 * @param classifiedMask         A cv::Mat where the component's new value will be written.
 * @param componentLabel         An integer representing the label of the connected component.
 * @param labelId                The label ID to assign to the component in the mask
 * .
 */
void changeComponentValue(const cv::Mat& labels, cv::Mat& classifiedMask, int componentLabel, labelId labelId);

#endif // CLASSIFICATION_HPP