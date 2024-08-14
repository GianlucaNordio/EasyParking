#ifndef PARKINGSPOTDETECTOR_HPP
#define PARKINGSPOTDETECTOR_HPP

#include "parkingSpot.hpp"
#include <opencv2/opencv.hpp>
#include <vector>
#include <algorithm>

void detectParkingSpot(const std::vector<cv::Mat>& images, std::vector<ParkingSpot>& parkingSpots);

bool isOverlapping(const cv::RotatedRect& spot1, const cv::RotatedRect& spot2, cv::Size imageSize);

std::vector<ParkingSpot> detectParkingSpotInImage(const cv::Mat& image);

std::vector<ParkingSpot> nonMaximaSuppression(const std::vector<std::vector<ParkingSpot>>& parkingSpots, cv::Size imageSize);

std::vector<cv::Point> convertToIntPoints(const std::vector<cv::Point2f>& floatPoints);

cv::Mat applyGammaTransform(const cv::Mat& src, double gamma);


#endif // PARKINGSPOTDETECTOR_HPP