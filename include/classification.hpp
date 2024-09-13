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

const float PERCENTAGE_OUTSIDE_THRESHOLD = 0.5;

cv::Mat classifyCars(std::vector<ParkingSpot> spaces, cv::Mat segmentationMasks);

void calculateComponentInsideRotatedRect(const cv::Mat& labels, const cv::Mat& stats, cv::Mat& output, ParkingSpot& rotatedRect, int componentLabel);

void changeComponentValue(const cv::Mat& labels, cv::Mat& mask, int componentLabel, labelId labelId);

#endif // CLASSIFICATION_HPP