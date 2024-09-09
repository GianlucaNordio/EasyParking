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
cv::Mat preprocess(const cv::Mat& src);
float calculateDistance(const cv::Point2f& p1, const cv::Point2f& p2);
std::vector<cv::Point2f> removeClosePoints(const std::vector<cv::Point2f>& points, float distanceThreshold);
void addSaltPepperNoise(cv::Mat& src, cv::Mat& dst, double noise_amount);
bool isMoreThanHalfBlack(const cv::Mat& image, const cv::RotatedRect& box);
void filterBoundingBoxes(cv::Mat& image, std::vector<cv::RotatedRect>& boxes);
#endif // PARKINGSPOTDETECTOR_HPP