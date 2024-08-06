#include "parkingSpot.hpp"

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
