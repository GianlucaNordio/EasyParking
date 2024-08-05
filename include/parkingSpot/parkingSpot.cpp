#include "parkingSpot.hpp"

void ParkingSpot::park() {
    occupied = true;
}

void ParkingSpot::leave() {
    occupied = false;
}

void ParkingSpot::displayStatus() const {
    std::cout << "Parking spot " << id << " is ";
    if(occupied)
        std::cout << "occupied" << std::endl;
    else
        std::cout << "free" << std::endl;
}
