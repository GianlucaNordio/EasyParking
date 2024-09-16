#ifndef RECTUTILS_HPP
#define RECTUTILS_HPP

#include <opencv2/opencv.hpp>

#include "constants.hpp"

/**
 * @brief Performs Non-Maximum Suppression (NMS) on a set of rotated rectangles based on their overlap and size.
 * 
 * This function processes a list of rotated rectangles and removes redundant overlapping rectangles 
 * based on a specified intersection area threshold. The decision on which rectangle to remove is 
 * made based on their area size, and can be configured to keep either the smallest or largest rectangle.
 *
 * @param rects A vector of cv::RotatedRect objects representing the bounding boxes to process.
 * @param elementsToRemove A vector of cv::RotatedRect objects that will store the rectangles marked for removal.
 * @param threshold A double representing the normalized intersection area threshold above which two rectangles 
 * are considered overlapping.
 * @param keepSmallest A boolean flag that indicates whether to keep the smallest rectangle (true) 
 * or the largest rectangle (false) when two rectangles overlap.
 *
 * @note The function assumes that the rects vector contains unique rectangles. 
 * Rectangles whose centers are exactly the same will not be compared.
 */
void nonMaximumSuppression(std::vector<cv::RotatedRect> &rects, std::vector<cv::RotatedRect> &elementsToRemove, double threshold, bool keepSmallest);

/**
 * @brief Computes the normalized intersection area between two rotated rectangles.
 *
 * This function calculates the area of intersection between two rotated rectangles (`rect1` and `rect2`) and
 * normalizes it by dividing by the smaller of the two rectangle areas. The normalization ensures that the 
 * intersection area is represented as a ratio, where 1 means complete overlap and 0 means no overlap.
 *
 * The function first extracts the corner points of both rectangles, then uses OpenCV's 
 * `cv::intersectConvexConvex` to find the intersection polygon. The area of this intersection is divided by the 
 * smaller of the two original rectangle areas.
 *
 * @param rect1 The first rotated rectangle.
 * @param rect2 The second rotated rectangle.
 * @return The normalized intersection area as a ratio in the range [0, 1], where 1 indicates complete overlap
 *         and 0 indicates no intersection.
 */
double computeIntersectionAreaNormalized(const cv::RotatedRect& rect1, const cv::RotatedRect& rect2);

/**
 * @brief Finds the iterator to an element in a vector of rotated rectangles based on the center coordinates.
 *
 * This function searches through a vector of `cv::RotatedRect` and returns an iterator to the first element
 * that has the same center coordinates (`center.x` and `center.y`) as the specified `elem`. If no such element
 * is found, the function returns `rects.cend()`, indicating the end of the vector.
 *
 * @param rects A constant reference to a vector of rotated rectangles (`cv::RotatedRect`).
 * @param elem The rotated rectangle whose center coordinates will be used for comparison.
 * @return A constant iterator to the found element in the vector, or `rects.cend()` if the element is not found.
 */
std::vector<cv::RotatedRect>::const_iterator elementIterator(const std::vector<cv::RotatedRect>& rects, const cv::RotatedRect& elem);

/**
 * @brief Converts a cv::RotatedRect into a line segment represented by its two midpoints.
 *
 * This function takes a rotated rectangle (cv::RotatedRect) and calculates the midpoints
 * of the two longest opposite edges. It then returns a line segment that connects these
 * midpoints as a vector of four floating point values (x1, y1, x2, y2), where (x1, y1)
 * is the midpoint of one edge and (x2, y2) is the midpoint of the opposite edge.
 *
 * The function identifies the two longest opposite edges of the rectangle and calculates
 * their midpoints, which are used to define the line segment.
 *
 * @param rect The rotated rectangle (cv::RotatedRect) to convert into a line.
 * 
 * @return cv::Vec4f A vector of four floats representing the line segment: (x1, y1, x2, y2).
 */
cv::Vec4f convertRectToLine(const cv::RotatedRect& rect);

/**
 * @brief Splits a rotated rectangle into two smaller rectangles and shifts them apart.
 * 
 * This function takes a `cv::RotatedRect` and splits it into two smaller rectangles, each 
 * having half the height of the original rectangle. The splitting occurs along the longest side 
 * of the original rectangle (between vertices 0 and 1). The midpoints of these sides are calculated, 
 * and the resulting smaller rectangles are shifted apart along the x and y axes, based on the angle of the original rectangle.
 * This is done because if both parts (obtained by splitting a rectangle), when slightly shifted and rotated, overlap with other rectangles, 
 * then we are between two rows of parking spots, which is not possible
 * 
 * The shift direction is determined by the rectangle's angle, adjusted by an optional offset 
 * defined by `SPLIT_DIRECTION_OFFSET`, and the shift amount is proportional to the original rectangle's width.
 * 
 * @param rect The input rotated rectangle to be split and shifted.
 * @return A pair of `cv::RotatedRect` objects, representing the two split and shifted parts of the original rectangle.
 */
std::pair<cv::RotatedRect, cv::RotatedRect> splitAndShiftRotatedRect(const cv::RotatedRect& rect);

/**
 * @brief Resolves overlaps between two sets of rotated rectangles by shifting rectangles in the second set.
 * 
 * This function iterates through two vectors of rotated rectangles and checks for overlaps between rectangles 
 * in `vector1` and `vector2`. If an overlap is detected (based on a normalized intersection area), the function 
 * shifts the overlapping rectangle from `vector2` along its longest axis by a specified shift amount until the overlap 
 * is resolved.
 * 
 * The overlap is resolved when the normalized intersection area between the two rectangles becomes less than the 
 * `RESOLVE_OVERLAP_THRESHOLD`. The `shiftAlongLongestAxis` function is used to move the rectangle.
 * 
 * @param vector1 A vector of `cv::RotatedRect` objects representing the first set of rectangles.
 * @param vector2 A vector of `cv::RotatedRect` objects representing the second set of rectangles to be shifted if overlaps occur.
 * @param shiftAmount The distance by which overlapping rectangles in `vector2` are shifted along their longest axis.
 */
void resolve_overlaps(std::vector<cv::RotatedRect>& vector1, std::vector<cv::RotatedRect>& vector2, double shiftAmount);

/**
 * @brief Shifts a rotated rectangle along its longest axis by a specified amount.
 * 
 * This function calculates the direction of the longest axis of a `cv::RotatedRect` and shifts 
 * the rectangle's center along this axis by the specified `shiftAmount`. The direction of the shift can 
 * be inverted if `invertDirection` is set to `true`. The shifted rectangle retains its original size 
 * and rotation angle.
 * 
 * The longest axis is determined by comparing the distances between the rectangle's vertices. 
 * The shift is applied by normalizing the direction vector along the longest axis and moving the 
 * rectangle's center by the specified amount in that direction.
 * 
 * @param rect The input rotated rectangle to be shifted.
 * @param shiftAmount The amount by which to shift the rectangle along its longest axis.
 * @param invertDirection A boolean flag indicating whether to shift in the opposite direction of the axis.
 * @return A new `cv::RotatedRect` with the shifted center, keeping the original size and rotation angle.
 */
cv::RotatedRect shiftAlongLongestAxis(const cv::RotatedRect& rect, float shift_amount, bool invert_direction);

/**
 * @brief Checks if a rotated rectangle is isolated, i.e., it does not overlap with any other rectangles in a given list.
 * 
 * This function determines if a `cv::RotatedRect` is isolated by first scaling the rectangle using 
 * the `scaleRotatedRect` function with a predefined `SCALE_FACTOR`. It then checks for overlaps 
 * between the scaled rectangle and each rectangle in a provided vector, `rects`. If any rectangle in 
 * the vector overlaps with the scaled rectangle (excluding overlap with the rectangle itself), the function 
 * returns `false`, indicating the rectangle is not isolated. Otherwise, it returns `true`.
 * 
 * @param rect The rotated rectangle to be checked for isolation.
 * @param rects A vector of `cv::RotatedRect` objects representing other rectangles to compare against.
 * @return `true` if the rectangle is isolated (i.e., does not overlap with any other rectangles), 
 *         `false` if there is any overlap with other rectangles in the list.
 */
bool isAlone(cv::RotatedRect rect, std::vector<cv::RotatedRect> rects);

/**
 * @brief Scales a rotated rectangle by a specified factor.
 * 
 * This function scales the size (width and height) of a `cv::RotatedRect` by a given `scaleFactor`. 
 * The function creates and returns a new `cv::RotatedRect` with the updated size while keeping 
 * the original center and rotation angle unchanged.
 * 
 * @param rect The input rotated rectangle to be scaled.
 * @param scaleFactor The factor by which to scale the rectangle's size. A value greater than 1 
 *                    increases the size, while a value less than 1 decreases it.
 * @return A new `cv::RotatedRect` with the scaled size, same center, and same rotation angle as the original.
 */
cv::RotatedRect scaleRotatedRect(const cv::RotatedRect& rect, double scaleFactor);

/**
 * @brief Determines whether two rotated rectangles overlap.
 * 
 * This function checks if two `cv::RotatedRect` objects overlap by using the OpenCV function 
 * `cv::rotatedRectangleIntersection`, which computes the intersection points between the rectangles. 
 * The function returns `true` if the rectangles fully or partially overlap, and `false` otherwise.
 * 
 * The possible intersection statuses are:
 * - `cv::INTERSECT_FULL`: The rectangles fully overlap.
 * - `cv::INTERSECT_PARTIAL`: The rectangles partially overlap.
 * 
 * @param rect1 The first rotated rectangle.
 * @param rect2 The second rotated rectangle.
 * @return `true` if the rectangles fully or partially overlap, `false` otherwise.
 */
bool areRectsOverlapping(const cv::RotatedRect& rect1, const cv::RotatedRect& rect2);

#endif // RECTUTILS.HPP