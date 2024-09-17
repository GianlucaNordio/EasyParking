#ifndef PARKINGSPOTDETECTOR_HPP
#define PARKINGSPOTDETECTOR_HPP

#include <opencv2/opencv.hpp>
#include <vector>

#include "parkingSpot.hpp"
#include "rectUtils.hpp"
#include "lineUtils.hpp"
#include "templateMatching.hpp"
#include "constants.hpp"

void detectParkingSpots(const std::vector<cv::Mat>& images, std::vector<ParkingSpot>& bestParkingSpots, std::vector<std::vector<ParkingSpot>>& baseSequenceParkingSpots);

void detectParkingSpotInImage(const cv::Mat& image, std::vector<ParkingSpot>& parkingSpots);

/**
 * @brief Constructs rotated rectangles from a vector of segments.
 * 
 * This function processes a vector of segments to build rotated rectangles. For each segment in the input vector:
 * 1. It calls the `buildRotatedRectFromPerpendicular` function to create a rotated rectangle based on the segment.
 * 2. It checks if the resulting rotated rectangle has a size area greater than a predefined minimum area.
 * 3. Only rectangles meeting the area criterion are added to the result vector.
 * 
 * @param segments A vector of `cv::Vec4f` representing the segments, where each `cv::Vec4f` contains coordinates of a segment (x1, y1, x2, y2).
 * @return A vector of `cv::RotatedRect` representing the constructed rotated rectangles that meet the size area requirement.
 */
std::vector<cv::RotatedRect> buildRotatedRectsFromSegments(const std::vector<cv::Vec4f>& segments);

/**
 * @brief Builds a rotated rectangle based on a perpendicular segment and other segments.
 * 
 * This function constructs a rotated rectangle from a given segment by extending the segment and finding
 * intersections with other segments. The process involves:
 * 1. Calculating the perpendicular direction to the original segment.
 * 2. Extending the segment to a certain length.
 * 3. Checking for intersections between the extended perpendicular segment and other segments.
 * 4. Using the closest intersection to define the rotated rectangle.
 * 
 * The function also adjusts the center of the resulting rotated rectangle by a predefined shift.
 * 
 * @param segment A `cv::Vec4f` representing the original segment, with coordinates (x1, y1, x2, y2).
 * @param segments A vector of `cv::Vec4f` representing other segments to check for intersections.
 * @return A `cv::RotatedRect` representing the constructed rotated rectangle, or an empty `cv::RotatedRect` if no valid intersection is found.
 */
cv::RotatedRect buildRotatedRectFromPerpendicular(const cv::Vec4f& segment, const std::vector<cv::Vec4f>& segments);

#endif // PARKINGSPOTDETECTOR_HPP