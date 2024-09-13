#include "parkingSpot.hpp"

// Function to display the status of the parking spot
void ParkingSpot::displayStatus() const {
    std::cout << "Parking spot " << id << " is ";
    if(occupied)
        std::cout << "occupied" << std::endl;
    else
        std::cout << "free" << std::endl;
}
