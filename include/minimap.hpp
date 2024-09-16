#ifndef MINIMAP_HPP
#define MINIMAP_HPP

#include <opencv2/opencv.hpp>
#include <vector>

#include "parkingSpot.hpp"
#include "parkingSpotDetector.hpp"
#include "constants.hpp"

void buildSequenceMinimap(std::vector<std::vector<ParkingSpot>> parkingSpots, std::vector<cv::Mat>& miniMaps);

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
void alignRects(std::vector<cv::RotatedRect>& rects, double threshold);

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