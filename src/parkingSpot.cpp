#include "parkingSpot.hpp"

ParkingSpot::ParkingSpot(cv::RotatedRect bounding_box) {
    rect = bounding_box;
    id = 0;
    occupied = false;
}

ParkingSpot::ParkingSpot(){
    cv::RotatedRect rect;
    id = 0;
    occupied = false;
}

// Function to set as occupied the parking spot
void ParkingSpot::park() {
    occupied = true;
}
// Function to set as free the parking spot
void ParkingSpot::leave() {
    occupied = false;
}

// Function to display the status of the parking spot
void ParkingSpot::displayStatus() const {
    std::cout << "Parking spot " << id << " is ";
    if(occupied)
        std::cout << "occupied" << std::endl;
    else
        std::cout << "free" << std::endl;
}
