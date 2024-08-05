#ifndef PARKINGSPOTDETECTOR_HPP
#define PARKINGSPOTDETECTOR_HPP

#include "../parkingSpot/parkingSpot.hpp"
#include <vector>

void detectParkingSpot(const std::vector<cv::Mat>& images, std::vector<ParkingSpot>& parkingSpots);

#endif // PARKINGSPOTDETECTOR_HPP