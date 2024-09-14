#ifndef PARKINGSPOT_HPP
#define PARKINGSPOT_HPP

#include <opencv2/opencv.hpp>

/**
 * Class representing a parking spot in an image.
 * Each parking spot has an ID, confidence score, occupancy status, and a rotated rectangle that defines its bounds.
 */
class ParkingSpot {
    public:
        /** The ID of the parking spot. */
        int id;

        /** The confidence score for the detection of the parking spot. */
        double confidence;

        /** A flag indicating whether the parking spot is occupied. */
        bool occupied;

        /** The bounding box of the parking spot, represented as a rotated rectangle. */
        cv::RotatedRect rect;

        /**
         * Constructor to initialize a parking spot with given parameters.
         *
         * @param spotId           The ID of the parking spot.
         * @param spotConfidence   The confidence score for detecting the parking spot.
         * @param isOccupied       Boolean indicating whether the parking spot is occupied.
         * @param spotRect         A rotated rectangle defining the bounds of the parking spot.
         */
        ParkingSpot(int spotId, double spotConfidence, bool isOccupied, const cv::RotatedRect& spotRect)
        : id(spotId), confidence(spotConfidence), occupied(isOccupied), rect(spotRect) {}

        /**
         * Default constructor that initializes a parking spot with default values.
         * Sets ID to 0, confidence to 0.0, occupancy to false, and uses an empty rotated rectangle.
         */
        ParkingSpot() 
        : id(0), confidence(0.0), occupied(false), rect(cv::RotatedRect()) {}

        /**
         * Displays the status of the parking spot, including its ID, occupancy status, and confidence score.
         * This function is intended to provide a quick overview of the parking spot's state.
         */
        void displayStatus() const;
};


#endif // PARKINGSPOT_HPP