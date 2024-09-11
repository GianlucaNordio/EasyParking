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
cv::Mat preprocess(const cv::Mat& src);
cv::Vec2f get_direction(cv::Vec4f segm,bool blueStart);
cv::Vec2f get_segm_params(cv::Vec4f segm);
float get_segment_angular_coefficient(const cv::Vec4f& segment);
float get_segment_length(const cv::Vec4f& segment);

double computeIntersectionArea(const cv::RotatedRect& rect1, const cv::RotatedRect& rect2);
void nms(std::vector<cv::RotatedRect>& vec, std::vector<cv::RotatedRect>& elementsToRemove);
std::vector<cv::RotatedRect>::const_iterator elementIterator(const std::vector<cv::RotatedRect>& vec, const cv::RotatedRect& elem);
#endif // PARKINGSPOTDETECTOR_HPP