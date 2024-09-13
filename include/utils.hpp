#ifndef UTILS_HPP
#define UTILS_HPP

#include <opencv2/highgui.hpp>

#include "parkingSpot.hpp"
#include "parser.hpp"

/**
 * @brief Produce a single image starting from a vector containing one or more images
 * @param images vector of images (has to be at least of size greater then 1) that have all same witdth and height
 * @param imagesPerLine number of images required for each line 
 */
cv::Mat produceSingleImage(const std::vector<cv::Mat>& images, int imagesPerLine);

/**
 * @brief Loads the images from the sequence used to detect the path
 * @param datasetPath path of the dataset (for example ../dataset)
 * @param images vector that will contain the obtained images
 */
void loadBaseSequenceFrames(const std::string& datasetPath, std::vector<cv::Mat> &images);


/**
 * @brief Loads the images from the test sequences
 * @param datasetPath path of the dataset (for example ../dataset)
 * @param images vector containing each sequence as a vector of images
 */
void loadSequencesFrames(const std::string& datasetPath, int numSequences, std::vector<std::vector<cv::Mat>> &images);

/**
 * @brief Loads the images from the given path
 * @param path path of the images to load
 * @param images vector that will contain the found images
 */
void loadImages(std::string path, std::vector<cv::Mat> &images);

/**
 * @brief Loads the masks used as grouund truth for segmentation from the given path
 * @param path path of the masks to load
 * @param numSequences number of sequences present in the data
 */
void loadSequencesSegMasks(const std::string& datasetPath, int numSequences, std::vector<std::vector<cv::Mat>> &segMasks);

void loadGroundTruth(const std::string path, std::vector<ParkingSpot> &groundTruth);

void loadBaseSequenceGroundTruth(const std::string& datasetPath, std::vector<ParkingSpot> &groundTruth);

void loadSequencesGroundTruth(const std::string& datasetPath, int numSequences, std::vector<std::vector<ParkingSpot>> &groundTruth);

/**
 * Allows to convert the greyscale masks provided by the dataset to BGR.
 * Performs the following mapping:
 * 0: (128,128,128)
 * 1: (255,0,0)
 * 2: (0,255,0
 * 
 * @param srcImages vector of greyscale images containing the mask (values are only 0, 1, 2)
 * @param dstImages vector of BGR images produced by perorming the mapping on the input masks
 */
void convertGreyMaskToBGR(const std::vector<std::vector<cv::Mat>> &srcImages, std::vector<std::vector<cv::Mat>> &dstImages);


#endif // UTILS_HPP