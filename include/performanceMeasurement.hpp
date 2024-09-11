#ifndef PERFORMANCE_MEASUREMENT_H
#define PERFORMANCE_MEASUREMENT_H

#include <iostream>
#include <vector>
#include "parkingSpot.hpp"

const float IOU_THRESHOLD = 0.5;
double meanAveragePrecision(std::vector<ParkingSpot>& groundTruth, std::vector<ParkingSpot>& detections);
double meanIntersectionOverUnion();

#endif