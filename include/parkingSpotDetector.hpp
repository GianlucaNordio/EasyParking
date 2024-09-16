#ifndef PARKINGSPOTDETECTOR_HPP
#define PARKINGSPOTDETECTOR_HPP

#include <opencv2/opencv.hpp>
#include <opencv2/dnn/dnn.hpp>
#include <vector>
#include <algorithm>
#include <numeric>
#include <math.h>

#include "parkingSpot.hpp"
#include "rectUtils.hpp"
#include "lineUtils.hpp"
#include "templateMatching.hpp"
#include "constants.hpp"

void detectParkingSpots(const std::vector<cv::Mat>& images, std::vector<ParkingSpot>& parkingSpots);

std::vector<cv::RotatedRect> detectParkingSpotInImage(const cv::Mat& image);

double compute_avg(std::vector<double>& data);

std::vector<cv::Vec4f> merge_parallel_segments(std::vector<cv::Vec4f>& segments, float angle_threshold, float distance_threshold, cv::Mat image);
cv::Vec4f merge_segments(const cv::Vec4f& seg1, const cv::Vec4f& seg2);

cv::RotatedRect build_rotatedrect_from_movement(const cv::Vec4f& segment, const std::vector<cv::Vec4f>& segments, cv::Mat image);

std::vector<cv::RotatedRect> process_segments(const std::vector<cv::Vec4f>& segments, cv::Mat image);

cv::Vec4f extend_segment(const cv::Vec4f& seg, float extension_ratio);

cv::RotatedRect shrink_rotated_rect(const cv::RotatedRect& rect, float shorten_percentage);

std::vector<cv::RotatedRect> filter_by_surrounding(const std::vector<cv::RotatedRect>& rects1, const std::vector<cv::RotatedRect>& rects2,cv::Mat image);

double compute_median(std::vector<double>& data);

void align_points(std::vector<cv::Point2f>& points, float threshold);

#endif // PARKINGSPOTDETECTOR_HPP