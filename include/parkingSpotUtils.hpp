#ifndef PARKINGSPOTUTILS_HPP
#define PARKINGSPOTUTILS_HPP

#include <opencv2/opencv.hpp>

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
 * @brief Preprocesses an image to enhance parking lines for detection.
 *
 * This function applies several image processing techniques to the input image in order to
 * highlight parking lines. It uses a bilateral filter for noise reduction while preserving
 * edges, converts the image to grayscale, and computes the gradients along the x and y axes
 * using the Sobel operator. The gradients are then combined, and adaptive thresholding is
 * applied to create a binary image. The result is further refined using morphological operations
 * (dilation and erosion) to enhance the parking lines.
 *
 * Steps:
 * 1. Apply bilateral filtering to reduce noise while preserving edges.
 * 2. Convert the image to grayscale.
 * 3. Compute x and y gradients using the Sobel operator.
 * 4. Combine the gradients to obtain the gradient magnitude.
 * 5. Apply adaptive thresholding to create a binary image highlighting the lines.
 * 6. Use morphological operations (dilation and erosion) to further enhance the detected lines.
 *
 * @param image The input image (cv::Mat) in which to find parking lines.
 * 
 * @return cv::Mat A binary image (cv::Mat) where the parking lines are enhanced for detection.
 */
cv::Mat preprocessFindParkingLines(const cv::Mat& image);


cv::Mat preprocessFindWhiteLines(const cv::Mat& image);

#endif // PARKINGSPOTUTILS_HPP