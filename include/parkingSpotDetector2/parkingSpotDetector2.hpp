#ifndef PARKINGSPOTDETECTOR2_HPP
#define PARKINGSPOTDETECTOR2_HPP

#include "../parkingSpot/parkingSpot.hpp"
#include <opencv2/opencv.hpp>
#include <vector>
#include <algorithm>

void detectParkingSpots2(const std::vector<cv::Mat>& images, std::vector<ParkingSpot>& parkingSpots);

std::vector<ParkingSpot> detectParkingSpotInImage2(const cv::Mat& image);

cv::Mat applyGammaTransform(const cv::Mat& src, double gamma);

std::vector<ParkingSpot> nonMaximaSuppression2(const std::vector<std::vector<ParkingSpot>>& parkingSpots, cv::Size imageSize);

std::vector<cv::Point> convertToIntPoints2(const std::vector<cv::Point2f>& floatPoints);

bool isOverlapping2(const cv::RotatedRect& spot1, const cv::RotatedRect& spot2, cv::Size imageSize);

#endif // PARKINGSPOTDETECTOR2_HPP