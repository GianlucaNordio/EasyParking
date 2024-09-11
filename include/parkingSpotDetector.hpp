#ifndef PARKINGSPOTDETECTOR_HPP
#define PARKINGSPOTDETECTOR_HPP

#include "parkingSpot.hpp"
#include <opencv2/opencv.hpp>
#include <opencv2/dnn/dnn.hpp>
#include <vector>
#include <algorithm>
#include <numeric>
#include <math.h>

void detectParkingSpots(const std::vector<cv::Mat>& images, std::vector<ParkingSpot>& parkingSpots);

std::vector<ParkingSpot> detectParkingSpotInImage(const cv::Mat& image);

cv::Mat preprocess_find_white_lines(const cv::Mat& src);
cv::Mat preprocess_find_parking_lines(const cv::Mat& src);

cv::Vec2f get_direction(cv::Vec4f segm,bool blueStart);
cv::Vec2f get_segm_params(cv::Vec4f segm);

double compute_median(std::vector<double>& data);
float get_segment_angular_coefficient(const cv::Vec4f& segment);
float get_segment_length(const cv::Vec4f& segment);
std::vector<cv::Mat> generate_template(double width, double height, double angle, bool flipped);
std::vector<cv::RotatedRect> merge_overlapping_rects(std::vector<cv::RotatedRect>& rects);
bool are_rects_aligned(const cv::RotatedRect& rect1, const cv::RotatedRect& rect2, float angle_tolerance);
bool are_rects_overlapping(const cv::RotatedRect& rect1, const cv::RotatedRect& rect2);
cv::Point2f compute_longest_segment(const cv::RotatedRect& rect);
cv::RotatedRect build_rotated_rect_from_longest_segments(const cv::RotatedRect& rect1, const cv::RotatedRect& rect2);
cv::Vec4f convert_rect_to_line(const cv::RotatedRect& rect);

double computeIntersectionArea(const cv::RotatedRect& rect1, const cv::RotatedRect& rect2);
void nms(std::vector<cv::RotatedRect>& vec, std::vector<cv::RotatedRect>& elementsToRemove);
std::vector<cv::RotatedRect>::const_iterator elementIterator(const std::vector<cv::RotatedRect>& vec, const cv::RotatedRect& elem);

cv::Mat applyGammaTransform(const cv::Mat& src, double gamma);
#endif // PARKINGSPOTDETECTOR_HPP