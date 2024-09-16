#ifndef LINEUTILS_HPP
#define LINEUTILS_HPP

#include <opencv2/opencv.hpp>

#include "constants.hpp"

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

/**
 * @brief Preprocesses an image to highlight white lines by applying a series of filters and morphological operations.
 * 
 * This function performs several image processing steps to enhance white lines in the input image. The process includes:
 * - Bilateral filtering to reduce noise while preserving edges.
 * - Conversion to grayscale for further thresholding.
 * - Adaptive thresholding to create a binary image based on local means.
 * - Sobel operator to compute the gradient in both X and Y directions.
 * - Addition of the gradients to obtain the magnitude of edges.
 * - Morphological operations (dilation followed by erosion) to enhance the structure of detected lines.
 * 
 * @param image The input image in BGR format (typically loaded from a file or camera).
 * @return A binary image where white lines are enhanced and emphasized, ready for further processing.
 */
cv::Mat preprocessFindWhiteLines(const cv::Mat& image);

/**
 * @brief Filters out line segments that are too close to each other based on a specified distance threshold.
 * 
 * This function iterates through a list of 2D line segments (represented by cv::Vec4f, with each vector containing 
 * the coordinates of the two endpoints of a line: (x1, y1, x2, y2)). It compares each segment with the others 
 * and discards those that are within a specified distance threshold. The remaining segments are returned in a new vector.
 * 
 * The function avoids rechecking discarded segments for efficiency by maintaining a boolean vector. 
 * 
 * @param segments A vector of line segments, where each segment is represented by a cv::Vec4f (x1, y1, x2, y2).
 * @param distanceThreshold The threshold distance used to determine whether two segments are considered too close.
 * @return A vector of filtered line segments that are not within the specified distance of each other.
 */
std::vector<cv::Vec4f> filterCloseSegments(const std::vector<cv::Vec4f>& segments, double distance_threshold);

/**
 * @brief Computes the Euclidean distance between the midpoints of two line segments.
 * 
 * This function calculates the distance between the midpoints of two given 2D line segments. 
 * The midpoints of the segments are computed using the `computeMidpoint` function, and the 
 * Euclidean distance between these midpoints is calculated using `cv::norm`.
 * 
 * @param segment1 The first line segment, represented as a cv::Vec4f containing the endpoints (x1, y1, x2, y2).
 * @param segment2 The second line segment, represented as a cv::Vec4f containing the endpoints (x1, y1, x2, y2).
 * @return The Euclidean distance between the midpoints of the two segments.
 */
double computeDistanceBetweenSegments(const cv::Vec4f& segment1, const cv::Vec4f& segment2);

/**
 * @brief Computes the midpoint of a 2D line segment.
 * 
 * This function calculates the midpoint of a line segment defined by two endpoints (x1, y1) and (x2, y2). 
 * The midpoint is computed as the average of the x and y coordinates of the endpoints.
 * 
 * @param segment A line segment represented as a cv::Vec4f, where the vector contains the endpoints (x1, y1, x2, y2).
 * @return A cv::Point2f representing the midpoint of the segment.
 */
cv::Point2f computeMidpoint(const cv::Vec4f& segment);

/**
 * @brief Filters out line segments that are near the top-right corner of the image.
 * 
 * This function removes line segments whose endpoints or midpoints fall near the top-right corner 
 * of the image. A convex hull is defined around the top-right corner, and the function checks 
 * whether each segment is inside this region. Segments that are entirely outside the convex hull 
 * are added to the filtered list.
 * 
 * The hull is defined using three points: the top-right corner of the image, and two additional points 
 * to form a triangular region. The function uses `cv::pointPolygonTest` to check whether the 
 * segment's endpoints and midpoint lie inside the convex hull.
 * 
 * @param segments A vector of line segments, where each segment is represented as cv::Vec4f (x1, y1, x2, y2).
 * @param imageSize The size of the image (width and height), used to determine the top-right corner.
 * @return A vector of filtered segments that are not near the top-right corner of the image.

 */
std::vector<cv::Vec4f> filterSegmentsNearTopRight(const std::vector<cv::Vec4f>& segments, const cv::Size& imageSize);

/**
 * @brief Calculates the angular coefficient (angle) of a line segment in degrees.
 * 
 * This function computes the angle of a line segment defined by two endpoints, expressed in degrees. 
 * The angle is calculated using the arctangent of the ratio of the vertical difference to the horizontal 
 * difference between the endpoints. The result is converted from radians to degrees.
 * 
 * @param segment A `cv::Vec4f` representing the line segment, with coordinates (x1, y1) and (x2, y2).
 * @return The angle of the line segment in degrees, relative to the x-axis.
 */
double getSegmentAngle(const cv::Vec4f& segment);

/**
 * @brief Calculates the length of a line segment.
 * 
 * This function computes the length of a line segment defined by two endpoints. The length is 
 * determined using the Euclidean distance formula, which calculates the distance between the two 
 * endpoints of the segment.
 * 
 * @param segment A `cv::Vec4f` representing the line segment, with coordinates (x1, y1) and (x2, y2).
 * @return The length of the line segment.
 */
double getSegmentLength(const cv::Vec4f& segment);

#endif // LINEUTILS_HPP