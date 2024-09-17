// Davide Molinaroli

#ifndef PARKINGSPOTDETECTOR_HPP
#define PARKINGSPOTDETECTOR_HPP

#include <opencv2/opencv.hpp>
#include <vector>

#include "parkingSpot.hpp"
#include "rectUtils.hpp"
#include "lineUtils.hpp"
#include "templateMatching.hpp"
#include "constants.hpp"

/**
 * @brief Detects parking spots in a sequence of images, applies Non-Maximum Suppression (NMS) 
 * to remove overlapping or redundant detections, and stores the filtered spots.
 *
 * This function processes each image in the provided vector of images to detect potential parking spots.
 * For each image, it identifies the parking spots using the `detectParkingSpotInImage` function and 
 * stores the spots in the `baseSequenceParkingSpots`. The coordinates of all the detected spots 
 * are accumulated into a vector and then passed to the Non-Maximum Suppression (NMS) function to 
 * filter out redundant or overlapping spots. Finally, the best filtered parking spots are stored in 
 * the `bestParkingSpots` vector.
 *
 * @param images A vector of OpenCV Mat objects, where each Mat represents an image.
 * @param bestParkingSpots A reference to a vector where the best parking spots (after NMS) will be stored.
 * @param baseSequenceParkingSpots A reference to a vector of vectors, where parking spots for each image are stored.
 */
void detectParkingSpots(const std::vector<cv::Mat>& images, std::vector<ParkingSpot>& bestParkingSpots, std::vector<std::vector<ParkingSpot>>& baseSequenceParkingSpots);

/**
 * @brief Detects parking spots in a single image by analyzing line segments and applying template matching and 
 * Non-Maximum Suppression (NMS) to identify valid parking spot regions.
 *
 * This function performs a multi-step approach to detect parking spots within the input image. First, it preprocesses 
 * the image to find white lines. After detecting line segments, it filters out short segments and segments near 
 * the top-right of the image. The function calculates the angles and lengths of remaining segments, performs 
 * multi-scale template matching using these average values, and merges overlapping bounding boxes. Non-Maximum 
 * Suppression (NMS) is applied to further refine the detected regions. Finally, filtered bounding boxes are converted 
 * into `ParkingSpot` objects, which are returned via the `parkingSpots` vector.
 *
 * @param image The input image as an OpenCV Mat object in which parking spots will be detected.
 * @param parkingSpots A reference to a vector where detected parking spots will be stored.
 */
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