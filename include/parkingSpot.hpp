#ifndef PARKINGSPOT_HPP
#define PARKINGSPOT_HPP

#include <opencv2/opencv.hpp>

class ParkingSpot {
    public:
        int id;
        double confidence;
        bool occupied;
        cv::RotatedRect rect;

        ParkingSpot(int spotId, double spotConfidence, bool isOccupied, const cv::RotatedRect& spotRect)
        : id(spotId), confidence(spotConfidence), occupied(isOccupied), rect(spotRect) {}
        ParkingSpot() 
        : id(0), confidence(0.0), occupied(false), rect(cv::RotatedRect()) {}
        void park();
        void leave();
        void displayStatus() const;
};

#endif // PARKINGSPOT_HPP