#ifndef UTILS_HPP
#define UTILS_HPP

#include <opencv2/highgui.hpp>

/**
 * @brief Produce a single image starting from a vector containing one or more images
 * @param images vector of images (has to be at least of size greater then 1) that have all same witdth and height
 * @param imagesPerLine number of images required for each line 
 */
cv::Mat produceSingleImage(const std::vector<cv::Mat>& images, int imagesPerLine);

#endif // UTILS_HPP