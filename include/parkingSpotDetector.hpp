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

double compute_avg(std::vector<double>& data);
float get_segment_angular_coefficient(const cv::Vec4f& segment);
float get_segment_length(const cv::Vec4f& segment);
std::vector<cv::Mat> generate_template(double width, double height, double angle, bool flipped);
std::vector<cv::Vec4f> split_segment(const cv::Vec4f& segment);
cv::Vec4f convert_rect_to_line(const cv::RotatedRect& rect);
cv::Point2f compute_perpendicular_direction(const cv::Vec4f& segment);
cv::Point2f compute_midpoint(const cv::Vec4f& segment);
std::vector<cv::Vec4f> filter_close_segments(const std::vector<cv::Vec4f>& segments, float distance_threshold);
std::vector<cv::Vec4f> merge_parallel_segments(std::vector<cv::Vec4f>& segments, float angle_threshold, float distance_threshold, cv::Mat image);
double computeIntersectionArea(const cv::RotatedRect& rect1, const cv::RotatedRect& rect2);
void nms(std::vector<cv::RotatedRect>& vec, std::vector<cv::RotatedRect>& elementsToRemove);
std::vector<cv::RotatedRect>::const_iterator elementIterator(const std::vector<cv::RotatedRect>& vec, const cv::RotatedRect& elem);
cv::Vec4f merge_segments(const cv::Vec4f& seg1, const cv::Vec4f& seg2);
std::vector<cv::Vec4f> filter_segments_near_top_right(const std::vector<cv::Vec4f>& segments, const cv::Size& image_size, double distance_threshold);
bool are_rects_overlapping(const cv::RotatedRect& rect1, const cv::RotatedRect& rect2);
std::vector<cv::RotatedRect> merge_overlapping_rects(std::vector<cv::RotatedRect>& rects);
// Helper function to check if two rotated rectangles are aligned (same angle within a tolerance)
bool are_rects_aligned(const cv::RotatedRect& rect1, const cv::RotatedRect& rect2, float angle_tolerance);
cv::Point2f get_rightmost_endpoint(const cv::Vec4f& segment);
bool segments_intersect(const cv::Vec4f& seg1, const cv::Vec4f& seg2, cv::Point2f& intersection);
cv::RotatedRect build_rotatedrect_from_movement(const cv::Vec4f& segment, const std::vector<cv::Vec4f>& segments, cv::Mat image);
std::vector<cv::RotatedRect> process_segments(const std::vector<cv::Vec4f>& segments, cv::Mat image);

cv::Mat applyGammaTransform(const cv::Mat& src, double gamma);
#endif // PARKINGSPOTDETECTOR_HPP