#ifndef MINIMAP_HPP
#define MINIMAP_HPP

#include <opencv2/opencv.hpp>
#include <vector>

#include "parkingSpot.hpp"
#include "parkingSpotDetector.hpp"
#include "constants.hpp"

/**
 * @brief Builds a series of minimaps for a sequence of parking spot data.
 * 
 * This function iterates over multiple sets of parking spot data and generates a minimap for each set.
 * It calls `buildMinimap` for each set of parking spots to draw the corresponding minimap on the provided
 * images in the `miniMaps` vector.
 * 
 * @param parkingSpots A vector of vectors, where each inner vector contains `ParkingSpot` objects representing
 * the parking spots for a specific frame or time step.
 * @param miniMaps A vector of `cv::Mat` objects where each element will be updated with the minimap corresponding
 * to the parking spots in the `parkingSpots` vector. Each `cv::Mat` should be pre-allocated with appropriate size
 * and type.
 * 
 * @note The size of the `miniMaps` vector must match the size of the `parkingSpots` vector, as each minimap 
 * corresponds to one set of parking spots.
 * 
 * @throws std::out_of_range If the `miniMaps` vector does not have enough elements to match the number of 
 * `parkingSpots` vectors.
 */
void buildSequenceMinimap(std::vector<std::vector<ParkingSpot>> parkingSpots, std::vector<cv::Mat>& miniMaps);

/**
 * @brief Builds a minimap representing parking spots and their occupancy status on a convex hull.
 * 
 * This function creates a minimap image based on the parking spot data provided. It first computes
 * the convex hull of the parking spots' bounding boxes and highlights the four longest edges. The function
 * then calculates the intersection points of the lines to determine the corners, applies a perspective
 * transformation, and draws the transformed parking spots on the minimap.
 * 
 * Each parking spot is displayed with a color representing its occupancy status (red for occupied, blue 
 * for free), and the bounding box of the parking spot is drawn at a transformed location in the minimap.
 * 
 * @param parkingSpot A vector of `ParkingSpot` objects representing the parking spots to be displayed 
 * on the minimap. Each `ParkingSpot` contains a `cv::RotatedRect` for its bounding box and an occupancy 
 * flag.
 * @param miniMap A reference to a `cv::Mat` object where the minimap will be drawn. This matrix is 
 * expected to have the appropriate size and type for rendering the minimap.
 * 
 * @note The function assumes that all parking spots in the input have valid, non-zero area bounding boxes. 
 * Parking spots with an area smaller than 1 are ignored. It is also assumed that the `miniMap` has been 
 * initialized to a blank image of appropriate size.
 * 
 * @throws std::invalid_argument If the `parkingSpot` vector is empty.
 */
void buildMinimap(std::vector<ParkingSpot> parkingSpot, cv::Mat& miniMap);

/**
 * @brief Finds and arranges the corners of a quadrilateral from a set of 4 points.
 * 
 * This function takes a vector of 4 points and arranges them into a specific order to represent the
 * corners of a quadrilateral. The function sorts the points by their y-coordinate to determine which
 * points are the top and bottom ones. Then, it further sorts the top and bottom points by their x-coordinate
 * to determine the left and right corners.
 * 
 * @param points A vector of cv::Point2f objects containing exactly 4 points. These points are expected to
 * form the corners of a quadrilateral.
 * 
 * @return A vector of cv::Point2f objects representing the corners of the quadrilateral in the following
 * order: top-left, top-right, bottom-left, and bottom-right.
 * 
 * @throws std::invalid_argument If the input vector does not contain exactly 4 points.
 * 
 * @note The input points are assumed to form a convex quadrilateral. The function will not handle cases where
 * the points are not in such an arrangement.
 */
std::vector<cv::Point2f> findCorners(const std::vector<cv::Point2f>& points);

/**
 * @brief Aligns rectangles along their y-coordinates based on a specified threshold.
 * 
 * This function adjusts the y-coordinates of a set of rotated rectangles so that rectangles within
 * a given threshold of each other are aligned to the same y-coordinate. The rectangles are first
 * sorted by their y-coordinate, and then rectangles with y-coordinates close to each other are aligned
 * to the base y-coordinate of the first rectangle in their group.
 * 
 * @param rects A vector of cv::RotatedRect objects representing the rectangles to be aligned.
 * @param threshold A double value representing the maximum allowable distance between y-coordinates 
 * for rectangles to be considered for alignment. Rectangles whose y-coordinates differ by this amount 
 * or less will be aligned to the same y-coordinate.
 * 
 * @note The function modifies the y-coordinates of the rectangles in the input vector in-place.
 */
void alignRects(std::vector<ParkingSpot>& rects, double threshold);

/**
 * @brief Adds a minimap to a sequence of images by overlaying it in a defined region.
 * 
 * This function overlays minimaps onto a sequence of images. The minimaps are placed in the bottom-left 
 * corner of each corresponding image in the sequence. The function ensures that the minimap dimensions 
 * fit within the respective images before copying.
 * 
 * @param miniMap A vector of cv::Mat objects representing the minimaps to be added. Each minimap 
 * corresponds to an image in the sequence.
 * @param sequence A vector of cv::Mat objects representing the sequence of images onto which the minimaps 
 * will be overlaid. The minimap is copied into the bottom-left region of each image.
 * 
 * @note It is assumed that the size of the miniMap vector is equal to the size of the sequence vector. 
 * If a minimap is larger than its corresponding image in either dimension, it will not be added.
 */

void addMinimap(const std::vector<cv::Mat>& miniMap, std::vector<cv::Mat>& sequence);

#endif // MINIMAP_HPP