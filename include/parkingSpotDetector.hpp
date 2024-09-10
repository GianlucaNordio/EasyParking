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

cv::Mat preprocess(const cv::Mat& src);
float calculateDistance(const cv::Point2f& p1, const cv::Point2f& p2);
std::vector<cv::Point2f> removeClosePoints(const std::vector<cv::Point2f>& points, float distanceThreshold);
double computeIntersectionArea(const cv::RotatedRect& rect1, const cv::RotatedRect& rect2);
std::vector<std::pair<cv::RotatedRect, double>>::const_iterator elementIterator(
    const std::vector<std::pair<cv::RotatedRect, double>>& vec,
    const std::pair<cv::RotatedRect, double>& elem);
bool isDarkerThanThreshold(const cv::Mat& image, const cv::RotatedRect& box, double threshold);
void filterBoundingBoxes(cv::Mat& image, std::vector<std::pair<cv::RotatedRect, double>>& boxes);
cv::Mat applyGammaTransform(const cv::Mat& src, double gamma);
#endif // PARKINGSPOTDETECTOR_HPP