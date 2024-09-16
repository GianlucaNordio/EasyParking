#include "templateMatching.hpp"

/**
 * @brief Generates a rotated rectangular template and mask based on the provided dimensions and rotation angle.
 *
 * This function creates a rectangular template and its corresponding mask with specified width and rotation
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
std::vector<cv::Mat> generateTemplate(double width, double angle, bool flipped){
 
    int templateHeight;
    int templateWidth;
    cv::Point rotationCenter;
    double rotationAngle;
    double rotatedWidth;
    double rotatedHeight;

    double height = 10;
    width +=4;

    // Rotate the template
    if(!flipped) {
        templateHeight = height;
        templateWidth = width;
        rotationAngle = angle; // negative rotationAngle for not flipped (angle is negative)
        rotatedWidth = templateWidth*cos(-rotationAngle*CV_PI/180)+templateHeight;
        rotatedHeight = templateWidth*sin(-rotationAngle*CV_PI/180)+templateHeight; 
    }

    if(flipped) {
        templateHeight = width;
        templateWidth = height;
        rotationAngle = -90-angle; // negative rotationAngle for flipped (angle is negative)
        rotatedWidth = templateHeight*cos(-angle*CV_PI/180)+templateWidth;
        rotatedHeight = templateHeight;
                                            
    }

    // Horizontal template and mask definition
    cv::Mat horizontalTemplate(templateHeight, templateWidth, IMAGE_TYPE_1_CHANNEL, cv::Scalar(0));
    cv::Mat horizontalMask(templateHeight, templateWidth, IMAGE_TYPE_1_CHANNEL);

    // Build the template and mask
    for(int i = 0; i< horizontalTemplate.rows; i++) {
        for(int j = 0; j<horizontalTemplate.cols; j++) {
            if((!flipped ? i>2 && i < height-2 : j>2&&j<height-2) && (!flipped ? j>2 && j < width-2 : j>2&&j<width-2)) {
            horizontalTemplate.at<uchar>(i,j) = TEMPLATE_LINE_VALUE;
            horizontalMask.at<uchar>(i,j) = MASK_LINE_VALUE_HIGH;
            }
            else {
                horizontalMask.at<uchar>(i,j) = MASK_LINE_VALUE_LOW;
            }
        }
    }

    rotationCenter.y = templateHeight-1;
    rotationCenter.x = 0;

    cv::Mat rotationMatrix = cv::getRotationMatrix2D(rotationCenter, rotationAngle, 1);
    cv::Mat rotatedTemplate;
    cv::Mat rotatedMask;
    
    cv::warpAffine(horizontalTemplate, rotatedTemplate, rotationMatrix, cv::Size(rotatedWidth, rotatedHeight));
    cv::warpAffine(horizontalMask, rotatedMask, rotationMatrix, cv::Size(rotatedWidth, rotatedHeight));


    return std::vector<cv::Mat>{rotatedTemplate,rotatedMask};
}