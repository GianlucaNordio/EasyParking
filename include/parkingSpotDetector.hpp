#ifndef PARKINGSPOTDETECTOR_HPP
#define PARKINGSPOTDETECTOR_HPP

#include <opencv2/opencv.hpp>
#include <vector>

#include "parkingSpot.hpp"
#include "rectUtils.hpp"
#include "lineUtils.hpp"
#include "templateMatching.hpp"
#include "constants.hpp"

void detectParkingSpots(const std::vector<cv::Mat>& images, std::vector<ParkingSpot>& parkingSpots);

std::vector<cv::RotatedRect> detectParkingSpotInImage(const cv::Mat& image);

cv::RotatedRect buildRotateRectFromPerpendicular(const cv::Vec4f& segment, const std::vector<cv::Vec4f>& segments);

std::vector<cv::RotatedRect> buildRotateRectsFromSegments(const std::vector<cv::Vec4f>& segments);

#endif // PARKINGSPOTDETECTOR_HPP