#ifndef UTILS_HPP
#define UTILS_HPP

#include <opencv2/highgui.hpp>

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

#endif // UTILS_HPP