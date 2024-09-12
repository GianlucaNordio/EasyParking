#ifndef PERFORMANCE_MEASUREMENT_H
#define PERFORMANCE_MEASUREMENT_H

#include <iostream>
#include <vector>
#include "parkingSpot.hpp"
#include "classification.hpp"

const float IOU_THRESHOLD = 0.5;

double calculateMeanAveragePrecision(const std::vector<ParkingSpot>& groundTruths, const std::vector<ParkingSpot>& detections);

double calculateMeanIntersectionOverUnion(const cv::Mat &foundMask, const cv::Mat &groundTruthMask);

double calculateIoU(const ParkingSpot& rect1, const ParkingSpot& rect2);

std::vector<std::pair<double, double>> calculatePrecisionRecallCurve(const std::vector<ParkingSpot>& groundTruths, const std::vector<ParkingSpot>& detections);

double calculateAveragePrecision(const std::vector<std::pair<double, double>>& precisionRecallPoints);

double classIoU(const cv::Mat &foundMask, const cv::Mat &groundTruthMask, labelId id);

#endif