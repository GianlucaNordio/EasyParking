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
    width += TEMPLATE_BLACK_MARGIN; // We add 4 as black margin

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
    // Used to build a template with black pixels along its longest direction. 
    // This is done to not match with random patches of white pixels
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

void multiRotationTemplateMatching(const cv::Mat& image, double avgWidth, double avgAngle, double templateHeight, double scaleTemplate, double scaleRect, double threshold, std::vector<int> angleOffsets, std::vector<cv::RotatedRect>& rects, bool isAnglePositive){
    std::vector<double> rectScores(rects.size(), -1); // Initialize scores with -1 for non-existing rects

    for(int k = 0; k < angleOffsets.size(); k++) {

        int templateWidth = avgWidth * scaleTemplate;
        double angle = (isAnglePositive ? - avgAngle : avgAngle) + angleOffsets[k];

        std::vector<cv::Mat> rotatedTemplateAndMask = generateTemplate(templateWidth, angle, !isAnglePositive);
        cv::Mat rotatedTemplate = rotatedTemplateAndMask[0];
        cv::Mat rotatedMask = rotatedTemplateAndMask[1];
        cv::Mat tmResultUnnorm;
        cv::Mat tmResult;

        cv::matchTemplate(image, rotatedTemplate, tmResultUnnorm, cv::TM_SQDIFF, rotatedMask);
        cv::normalize(tmResultUnnorm, tmResult, 0, 1, cv::NORM_MINMAX, -1, cv::Mat());

        // Finding local minima
        cv::Mat eroded;
        std::vector<cv::Point> minima;
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(rotatedTemplate.cols, rotatedTemplate.rows));
        cv::erode(tmResult, eroded, kernel);
        cv::Mat localMinimaMask = (tmResult == eroded) & (tmResult <= threshold);

        // Find all non-zero points (local minima) in the mask
        cv::findNonZero(localMinimaMask, minima);

        // Iterate through each local minimum and process them
        for (const cv::Point& point : minima) {
            float score = tmResultUnnorm.at<float>(point);

            // Get center of the bbox to draw the rotated rect
            cv::Point center;
            center.x = point.x + rotatedTemplate.cols / 2;
            center.y = point.y + rotatedTemplate.rows / 2;
            cv::RotatedRect rotatedRect(center, cv::Size(templateWidth * scaleRect, templateHeight * scaleRect), (isAnglePositive?-angle : angle));
            
            bool overlaps = false;
            std::vector<size_t> overlappingIndices;

            for (size_t i = 0; i < rects.size(); ++i) {
                if (isAnglePositive && computeIntersectionAreaNormalized(rotatedRect, rects[i]) > 0) {
                    overlaps = true;
                    overlappingIndices.push_back(i);
                }
            }

            if (!overlaps) {
                // No overlap, add the rect directly
                rects.push_back(rotatedRect);
                rectScores.push_back(score);
            }
            else {
                // Handle overlap case: check if the current rect's score is higher (lower is good because we use sqdiff)
                bool addCurrentRect = true;
                for (size_t index : overlappingIndices) {
                    if (rotatedRect.size.area() < rects[index].size.area() || (rotatedRect.size.area() == rects[index].size.area() && score >= rectScores[index])) {
                        // The current rect has a higher or equal score, so don't add it
                        addCurrentRect = false;
                        break;
                    }
                }

                if (addCurrentRect) {
                    // Replace overlapping rects with the current one
                    for (size_t index : overlappingIndices) {
                        rects[index] = rotatedRect;
                        rectScores[index] = score;     
                    }
                }
            }
        }
    }
}