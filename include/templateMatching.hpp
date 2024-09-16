#ifndef TEMPLATEMATCHING_HPP
#define TEMPLATEMATCHING_HPP


#include <opencv2/opencv.hpp>
#include <vector>

#include "constants.hpp"

/**
 * @brief Generates a rotated rectangular template and mask based on the provided dimensions and rotation angle.
 *
 * This function creates a rectangular template and its corresponding mask with specified width, height, and rotation
 * angle. The template can be flipped if needed. The resulting template and mask are rotated by the given angle and returned 
 * as a vector of two `cv::Mat` objects.
 *
 * The rotation is performed around a center point, and the template is adjusted accordingly. If the `flipped` flag is set 
 * to `true`, the width and height of the template are swapped, and the rotation angle is adjusted by -90 degrees.
 *
 * @param width The width of the template before rotation.
 * @param height The height of the template before rotation (6 is added to the base height).
 * @param angle The angle in degrees to rotate the template.
 * @param flipped A boolean flag indicating whether the template should be flipped (width and height swapped).
 * @return A vector of two `cv::Mat` objects: the first is the rotated template, and the second is the rotated mask.
 */
std::vector<cv::Mat> generateTemplate(double width, double height, double angle, bool flipped);

#endif // TEMPLATEMATCHING_HPP