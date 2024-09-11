#ifndef CLASSIFICATION_HPP
#define CLASSIFICATION_HPP

#include <opencv2/opencv.hpp>
#include "parkingSpot.hpp"

enum labelId
{
    background,
    carInsideParkingSpot,
    carOutsideParkingSpot
};

const int ID_CAR_INSIDE_PARKING_LOT = 128; // should be 1
const int ID_CAR_OUTSIDE_PARKING_LOT = 255; // should be 2
const float PERCENTAGE_OUTSIDE_THRESHOLD = 0.5;

cv::Mat classifyCars(std::vector<ParkingSpot> spaces, cv::Mat segmentationMasks);

float calculateComponentInsideRotatedRect(const cv::Mat& labels, const cv::Mat& stats, cv::Mat& output, const cv::RotatedRect& rotatedRect, int componentLabel);

void changeComponentValue(const cv::Mat& labels, cv::Mat& mask, int componentLabel, uchar newValue);

#endif // CLASSIFICATION_HPP