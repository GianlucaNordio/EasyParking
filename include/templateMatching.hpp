#ifndef TEMPLATEMATCHING_HPP
#define TEMPLATEMATCHING_HPP


#include <opencv2/opencv.hpp>
#include <vector>

#include "rectUtils.hpp"
#include "constants.hpp"

/**
 * @brief Generates a rotated rectangular template and mask based on the provided dimensions and rotation angle.
 *
 * This function creates a rectangular template and its corresponding mask with specified widtd and rotation
 * angle. The template can be flipped if needed. The resulting template and mask are rotated by the given angle and returned 
 * as a vector of two `cv::Mat` objects.
 *
 * The rotation is performed around a center point, and the template is adjusted accordingly. If the `flipped` flag is set 
 * to `true`, the width and height of the template are swapped, and the rotation angle is adjusted by -90 degrees.
 *
 * @param width The width of the template before rotation.
 * @param angle The angle in degrees to rotate the template.
 * @param flipped A boolean flag indicating whether the template should be flipped (width and height swapped).
 * @return A vector of two `cv::Mat` objects: the first is the rotated template, and the second is the rotated mask.
 */
std::vector<cv::Mat> generateTemplate(double width, double angle, bool flipped);

/**
 * @brief Performs multi-rotation template matching on an input image to detect objects at various angles for a given scale.
 *
 * This function applies template matching with multiple rotated templates on the input image. The templates are
 * generated based on the provided average width, angle, and scaling factor. For each rotation, local minima 
 * (best match locations) are found, and corresponding rotated rectangles (detections) are created. 
 * The function can handle overlapping detections by only keeping the one with the better match score based on its last parameter.
 *
 * @param image The input image in which to perform template matching.
 * @param avgWidth The average width of the object to match, used as a base to calculate template size.
 * @param avgAngle The average angle of the object, used as a base to generate rotated templates.
 * @param templateHeight The height of the template used for matching.
 * @param scaleTemplate Scaling factor for the template size.
 * @param scaleRect Scaling factor for the resulting bounding box (rotated rectangle) size.
 * @param threshold Threshold to filter out poor matches (local minima with a match score above this value are ignored).
 * @param angleOffsets Vector of offsets to apply to the average angle for generating rotated templates.
 * @param rects A reference to a vector of cv::RotatedRect objects, where the detected bounding boxes are stored.
 * @param isAnglePositive A boolean indicating whether to rotate the templates in the positive or negative direction.
 *
 * The function does the following:
 * - Generates rotated templates and corresponding masks based on input parameters.
 * - Performs template matching using cv::matchTemplate with squared difference as the metric.
 * - Normalizes the template matching result and identifies local minima (best match points).
 * - For each local minimum, a rotated bounding box (rectangle) is created.
 * - If the angle is positive, the bounding box does not overlap with any existing detections with positive angle, it is added to the output.
 * - If the angle is positive and the bounding box overlaps with existing ones, only the detection with the better score (smaller squared difference) is kept.
 */
void multiRotationTemplateMatching(const cv::Mat& image, double avgWidth, double avgAngle, double templateHeight, double scaleTemplate, double scaleRect, double threshold, std::vector<int> angleOffsets, std::vector<cv::RotatedRect>& rects, bool isAnglePositive);

#endif // TEMPLATEMATCHING_HPP