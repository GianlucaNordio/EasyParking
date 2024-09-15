#ifndef PARKINGSPOT_HPP
#define PARKINGSPOT_HPP

#include <opencv2/opencv.hpp>

class ParkingSpot {
    public:
        int id;
        bool occupied;
        cv::RotatedRect rect;

        ParkingSpot();
        ParkingSpot(cv::RotatedRect bounding_box);
        void park();
        void leave();
        void displayStatus() const;
};

#endif // PARKINGSPOT_HPP