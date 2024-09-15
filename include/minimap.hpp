#include <opencv2/opencv.hpp>
#include <vector>

#include "parkingSpot.hpp"
#include "parkingSpotDetector.hpp"

void buildSequenceMinimap(std::vector<std::vector<ParkingSpot>> parkingSpots, std::vector<cv::Mat>& miniMaps);

void buildMinimap(std::vector<ParkingSpot> parkingSpot, cv::Mat& miniMap);

std::vector<cv::Point2f> find_corners(const std::vector<cv::Point2f>& points);

void align_rects(std::vector<cv::RotatedRect>& rects, float threshold);

void addMinimap(std::vector<cv::Mat>& minimap, const std::vector<cv::Mat>& sequence);