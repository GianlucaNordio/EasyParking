#include "parkingSpot.hpp"

/**
 * Displays the status of the parking spot, including its ID, occupancy status, and confidence score.
 * This function is intended to provide a quick overview of the parking spot's state.
 */
void ParkingSpot::displayStatus() const {
    std::cout << "Parking spot " << id << " is ";
    if(occupied)
        std::cout << "occupied" << std::endl;
    else
        std::cout << "free" << std::endl;
}