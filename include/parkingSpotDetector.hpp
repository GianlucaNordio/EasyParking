#ifndef PARKINGSPOTDETECTOR_HPP
#define PARKINGSPOTDETECTOR_HPP

#include "parkingSpot.hpp"
#include <opencv2/opencv.hpp>
#include <opencv2/dnn/dnn.hpp>
#include <vector>
#include <algorithm>
#include <math.h>

void detectParkingSpots(const std::vector<cv::Mat>& images, std::vector<ParkingSpot>& parkingSpots);

std::vector<ParkingSpot> detectParkingSpotInImage(const cv::Mat& image);

cv::Mat applyGammaTransform(const cv::Mat& src, double gamma);
cv::Mat contrastStretchTransform(const cv::Mat& src);


#endif // PARKINGSPOTDETECTOR_HPP