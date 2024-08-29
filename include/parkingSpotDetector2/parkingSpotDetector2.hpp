#ifndef PARKINGSPOTDETECTOR2_HPP
#define PARKINGSPOTDETECTOR2_HPP

#include "../parkingSpot/parkingSpot.hpp"
#include <opencv2/opencv.hpp>
#include <vector>
#include <algorithm>
#include <iostream>

void detectParkingSpots2(const std::vector<cv::Mat>& images, std::vector<ParkingSpot>& parkingSpots);

std::vector<ParkingSpot> detectParkingSpotInImage2(const cv::Mat& image);

cv::Mat applyGammaTransform(const cv::Mat& src, double gamma);

#endif // PARKINGSPOTDETECTOR2_HPP