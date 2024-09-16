#ifndef CLASSIFICATION_HPP
#define CLASSIFICATION_HPP

#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "parkingSpot.hpp"
#include "constants.hpp"

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
 * Classifies a sequence of parking spots by processing segmentation masks and 
 * stores the classified results in the output vector.
 *
 * @param parkingSpot            A vector of vector of ParkingSpot objects representing the parking spaces.
 * @param segmentationMasks      A vector of cv::Mat representing the segmentation masks for each image.
 * @param classifiedMasks        A reference to a vector of cv::Mat where the classified results will be stored.
 */
void classifySequence(std::vector<std::vector<ParkingSpot>>& parkingSpot, std::vector<cv::Mat> segmentationMasks, std::vector<cv::Mat>& classifiedMasks);

/**
 * @brief Classifies parking spots in an image based on segmentation mask.
 *
 * This function processes an input segmentation mask to classify parking spots 
 * as either occupied or free. It utilizes connected components analysis to identify
 * distinct components in the segmentation mask and then checks each parking spot 
 * against these components. The classification is based on the percentage of each 
 * component that overlaps with the parking spot.
 *
 * @param[in] parkingSpot        A vector of `ParkingSpot` objects, each representing a 
 *                                 parking spot with its associated rectangle.
 * @param[in] segmentationMask   A binary mask where connected components are 
 *                                 identified and labeled.
 * @param[out] classifiedMask    An output mask of the same size as the 
 *                                 segmentationMask, where each pixel value indicates 
 *                                 the classification of the component (0: background, 
 *                                 1: car inside, 2: car outside).
 *
 * @note The `PERCENTAGE_INSIDE_THRESHOLD` is used to determine if a parking spot 
 *       is considered occupied based on the percentage of the component's area 
 *       covered by the parking spot.
 */
void classifyImage(std::vector<ParkingSpot>& parkingSpot, cv::Mat segmentationMask, cv::Mat& classifiedMask);

#endif // CLASSIFICATION_HPP