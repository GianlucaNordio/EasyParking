#include "parkingSpotDetector.hpp"

/*
TODO: 
1. Preprocessing (pensare a come rendere invariante alle condizioni climatiche)
2. Generare meglio i template
3. Threshold iniziale sul template match normalizzando in base al massimo dell'output di matchTemplate
*/

// Function to detect parking spots in the images
void detectParkingSpots(const std::vector<cv::Mat>& images, std::vector<ParkingSpot>& parkingSpots) {
    
    std::vector<std::vector<ParkingSpot>> parkingSpotsPerImage;
    for(const auto& image : images) {
        // Find parking spots for each image separately
        parkingSpotsPerImage.push_back(detectParkingSpotInImage(image));
    }
}

// This function detects the parking spots in a single image
std::vector<ParkingSpot> detectParkingSpotInImage(const cv::Mat& image) {
    std::vector<ParkingSpot> parkingSpots;

    cv::Mat filteredImage;
    cv::bilateralFilter(image, filteredImage, -1, 40, 10);

    cv::Mat gs;
    cv::cvtColor(image, gs, cv::COLOR_BGR2GRAY);

    cv::Mat stretched = contrastStretchTransform(gs);
 
    // Set the gamma value
    double gammaValue = 1.25; // Example gamma value

    // Apply gamma transformation
    cv::Mat gammaCorrected1 = applyGammaTransform(gs, gammaValue);
 

    gammaValue = 2;
    cv::Mat gammaCorrected2 = applyGammaTransform(gammaCorrected1, gammaValue);
 

    cv::Mat gsthold;
    cv::threshold( gammaCorrected2, gsthold, 180, 255,  cv::THRESH_BINARY);


    cv::Mat gx;
    cv::Sobel(gammaCorrected2, gx, CV_16S, 1,0);

    cv::Mat gy;
    cv::Sobel(gammaCorrected1, gy, CV_16S, 0,1);

    cv::Mat abs_grad_x;
    cv::Mat abs_grad_y;
    cv::convertScaleAbs(gx, abs_grad_x);
    cv::convertScaleAbs(gy, abs_grad_y);

    cv::Mat grad_magn;
    cv::addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad_magn);

    cv::Mat equalized;
    cv::equalizeHist(gammaCorrected2,equalized);


    cv::Mat equalized_filt;
    cv::GaussianBlur(equalized,equalized_filt, cv::Size(5,5),30);

    cv::Mat gxeq;
    cv::Sobel(stretched, gxeq, CV_16S, 1,0);

    cv::Mat gyeq;
    cv::Sobel(stretched, gyeq, CV_16S, 0,1);

    cv::Mat abs_grad_xeq;
    cv::Mat abs_grad_yeq;
    cv::convertScaleAbs(gxeq, abs_grad_xeq);
    cv::convertScaleAbs(gyeq, abs_grad_yeq);

    //cv::Mat grad_magneq;
    //cv::addWeighted(abs_grad_xeq, 0.5, abs_grad_yeq, 0.5, 0, grad_magneq);

    cv::Mat laplacian;
    cv::Laplacian(equalized, laplacian, CV_32F);  // Use CV_32F to avoid overflow

    // Compute the absolute value of the Laplacian
    cv::Mat abs_laplacian;
    cv::convertScaleAbs(laplacian, abs_laplacian); // Convert to absolute value

    // Normalize the result to range [0, 255] for visualization
    cv::Mat normalized_abs_laplacian;
    cv::normalize(abs_laplacian, normalized_abs_laplacian, 0, 255, cv::NORM_MINMAX, CV_8U);


    cv::Mat filtered_laplacian;
    cv::bilateralFilter(normalized_abs_laplacian, filtered_laplacian, -1, 10, 10);


    cv::Mat grad_magn_bilateral;
    cv::bilateralFilter(grad_magn, grad_magn_bilateral, -1, 20, 10);


    cv::Mat medianblurred;
    cv::medianBlur(grad_magn_bilateral, medianblurred, 3);
    cv::Mat bilateralblurred;
    //cv::bilateralFilter(medianblurred, bilateralblurred, -1,20,10);
 
    cv::Mat element = cv::getStructuringElement( 
                        cv::MORPH_RECT, cv::Size(3,3)); 

    cv::Mat erodeg; 
    cv::erode(medianblurred, erodeg, element, cv::Point(-1, -1), 1); 
    cv::Mat dilate; 
    cv::dilate(medianblurred, dilate, element, cv::Point(-1, -1), 4); 
    std::vector<int> angles = {-7,-8,-9, -10, -11, -12, -13,-14,-15};
    std::vector<float> scales = {0.7, 0.8, 1, 1.05, 1.1, 1.2, 1.5,1.6,1.7,1.8,2};
    std::vector<cv::RotatedRect> line_boxes;
    std::vector<cv::Point2f> verts;
    double maxmax = 0.0;
    std::vector<float> scores;

    for(int l = 0; l<scales.size(); l++) {
        for(int k = 0; k<angles.size(); k++) {
            // Template size
            int template_height = 39*scales[l];
            int template_width = 130*scales[l];

            // Horizontal template and mask definition
            cv::Mat horizontal_template(template_height,template_width,CV_8U,cv::Scalar(0));
            cv::Mat horizontal_mask(template_height,template_width,CV_8U,cv::Scalar(0));

            // Build the template and mask
            for(int i = 0; i< horizontal_template.rows; i++) {
                for(int j = 0; j<horizontal_template.cols; j++) {
                    if(i<8 || j > template_width-8 || (i>(template_height-8)&& j > 20*scales[l]*scales[l])) {
                        horizontal_template.at<uchar>(i,j) = 190;
                    }
                    horizontal_mask.at<uchar>(i,j) = 255;
                }
            }


            // Rotate the template
            cv::Mat R = cv::getRotationMatrix2D(cv::Point2f(0,template_height-1),angles[k],1);
            cv::Mat rotated_template;
            cv::Mat rotated_mask;

            float rotated_width = template_width*cos(-angles[k]*CV_PI/180)+template_height;
            float rotated_height = template_width*sin(-angles[k]*CV_PI/180)+template_height;

            cv::warpAffine(horizontal_template,rotated_template,R,cv::Size(rotated_width,rotated_height));
            cv::warpAffine(horizontal_mask,rotated_mask,R,cv::Size(rotated_width,rotated_height));

            cv::Mat tm_result;

            cv::Mat tm_result_unnorm;
            cv::matchTemplate(dilate,rotated_template,tm_result_unnorm,cv::TM_SQDIFF,rotated_mask);
            double min,max;
            cv::Point minloc(0,0), maxloc(0,0);
            cv::minMaxLoc(tm_result_unnorm,&min,&max,&minloc,&maxloc);

            if(min/max > 0.2) continue;

            if(max > maxmax) {
                maxmax = max;
            }

            cv::normalize( tm_result_unnorm, tm_result, 0, 1, cv::NORM_MINMAX, -1, cv::Mat() );
       

            // Finding local minima
            cv::Mat eroded;
            std::vector<cv::Point> minima;
            cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(rotated_width, rotated_height));
            cv::erode(tm_result, eroded, kernel);
            cv::Mat localMinimaMask = (tm_result == eroded) & (tm_result <= 0.025);


            // Find all non-zero points (local minima) in the mask
            cv::findNonZero(localMinimaMask, minima);

            // Draw bboxes of the found lines
            for (const cv::Point& pt : minima) {
                // Save score of local minima
                scores.push_back(tm_result_unnorm.at<float>(pt));

                // White circles at minima points
                // cv::circle(gs, pt, 3, cv::Scalar(255), 1);

                // Get center of the bbox to draw the rotated rect
                cv::Point center;
                center.x = pt.x+rotated_width/2;
                center.y = pt.y+rotated_height/2;

                cv::RotatedRect rotatedRect(center, cv::Size(template_width-30,template_height), -angles[k]);
                line_boxes.push_back(rotatedRect);

                // Draw the rotated rectangle using lines between its vertices
                cv::Point2f vertices[4];
                rotatedRect.points(vertices);

                for (int i = 0; i < 4; i++) {
                    verts.push_back(vertices[i]);
                }
            }

        }
    }

    for(int i = 0; i < line_boxes.size(); i++) {
        scores[i] = scores[i]/maxmax;
    }

    float scoreThreshold = 0.2f;  // Minimum score to keep
    float nmsThreshold = 0.2f;    // IoU threshold for NMS

    // Vector to store indices of bounding boxes to keep after NMS
    std::vector<int> indices;

    // Apply NMS for rotated rectangles
    cv::dnn::NMSBoxes(line_boxes, scores, scoreThreshold, nmsThreshold, indices);

    // Draw the remaining boxes after NMS
    std::vector<cv::Point2f> nms_centers;

    for (int idx : indices) {
        cv::RotatedRect& rect = line_boxes[idx];
        nms_centers.push_back(rect.center);
    }

    // Average min distance
    double sumOfMinDistances = 0.0;
    // Iterate through each point in the vector
    for (size_t i = 0; i < nms_centers.size(); ++i) {
        double minDistance = std::numeric_limits<double>::max(); // Initialize with the maximum possible value

        // Find the minimum distance to any other point
        for (size_t j = 0; j < nms_centers.size(); ++j) {
            if (i != j) { // Don't compare the point to itself
                double distance = cv::norm(nms_centers[i] - nms_centers[j]);
                if (distance < minDistance) {
                    minDistance = distance;
                }
            }
        }

        sumOfMinDistances += minDistance;
    }

    // Compute the average minimum distance
    double averageMinDistance = sumOfMinDistances / nms_centers.size();

    // Remove nms centers too close or too far from eachother
    for (int idx : indices) {
        cv::RotatedRect& rect = line_boxes[idx];
        double minDistance = std::numeric_limits<double>::max(); // Initialize with the maximum possible value
        int closest;

        // Find the minimum distance to any other point
        for (int j: indices) {
            if (idx != j) { // Don't compare the point to itself
                double distance = cv::norm(rect.center - line_boxes[j].center);
                if (distance < minDistance) {
                    minDistance = distance;
                    closest = j;
                }
            }
        }
        
        /*if(minDistance < averageMinDistance/2 || minDistance > averageMinDistance*2) {
            indices.erase(std::find(indices.begin(),indices.end(),idx));
        }*/
        if(minDistance < averageMinDistance*2) {
            // Draw the rotated rectangle
            cv::Point2f vertices[4];
            rect.points(vertices);
        }
    }

    // O uso questo come centri di k-means, o non uso NMS e quando disegno un nuovo rotatedrect controllo che il suo match sia migliore di quello precedente
    // rispetto al bbox che gli è più vicino
    // Display the image

    return parkingSpots;
}

cv::Mat applyGammaTransform(const cv::Mat& src, double gamma) {
    // Create a lookup table for faster processing
    cv::Mat lookupTable(1, 256, CV_8U);
    uchar* p = lookupTable.ptr();
    for (int i = 0; i < 256; ++i) {
        p[i] = cv::saturate_cast<uchar>(pow(i / 255.0, gamma) * 255.0);
    }

    cv::Mat dst;
    // Apply the lookup table to the source image
    cv::LUT(src, lookupTable, dst);

    return dst;
}

cv::Mat contrastStretchTransform(const cv::Mat& src) {
    // Create a lookup table for faster processing
    cv::Mat lookupTable(1, 256, CV_8U);
    uchar* p = lookupTable.ptr();
    for (int i = 0; i < 256; ++i) {
        if(i < 80) {
            p[i] = cv::saturate_cast<uchar>(i/4);
        }
        else {
            p[i] = cv::saturate_cast<uchar>(i*2);
        }
        
    }

    cv::Mat dst;
    // Apply the lookup table to the source image
    cv::LUT(src, lookupTable, dst);

    return dst;
}