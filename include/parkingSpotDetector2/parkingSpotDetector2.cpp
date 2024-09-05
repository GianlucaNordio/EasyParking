#include "parkingSpotDetector2.hpp"

/*
TODO: 
1. Preprocessing (pensare a come rendere invariante alle condizioni climatiche)
2. Generare meglio i template
3. Threshold iniziale sul template match normalizzando in base al massimo dell'output di matchTemplate
4. Chiedere nel forum quanti parametri possiamo usare
*/

// Function to detect parking spots in the images
void detectParkingSpots2(const std::vector<cv::Mat>& images, std::vector<ParkingSpot>& parkingSpots) {
    
    std::vector<std::vector<ParkingSpot>> parkingSpotsPerImage;
    for(const auto& image : images) {
        // Find parking spots for each image separately
        parkingSpotsPerImage.push_back(detectParkingSpotInImage2(image));
    }
}

// This function detects the parking spots in a single image
std::vector<ParkingSpot> detectParkingSpotInImage2(const cv::Mat& image) {
    std::vector<ParkingSpot> parkingSpots;

    cv::Mat filteredImage;
    cv::bilateralFilter(image, filteredImage, -1, 40, 10);

    cv::imshow("bilateral filtered", filteredImage);
    cv::waitKey(0);

    cv::Mat gs;
    cv::cvtColor(image, gs, cv::COLOR_BGR2GRAY);

    cv::imshow("grayscale", gs);
    cv::waitKey(0);

    cv::Mat stretched = contrastStretchTransform(gs);
    cv::imshow("stretched", stretched);
    cv::waitKey(0);

    // Set the gamma value
    double gammaValue = 1.25; // Example gamma value

    // Apply gamma transformation
    cv::Mat gammaCorrected1 = applyGammaTransform(gs, gammaValue);
    cv::imshow("gamma tf1", gammaCorrected1);
    cv::waitKey(0);

    gammaValue = 2;
    cv::Mat gammaCorrected2 = applyGammaTransform(gammaCorrected1, gammaValue);
    cv::imshow("gamma tf2", gammaCorrected2);
    cv::waitKey(0);

    cv::Mat gsthold;
    cv::threshold( gammaCorrected2, gsthold, 180, 255,  cv::THRESH_BINARY);
    // cv::imshow("gsthold", gsthold);
    // cv::waitKey(0);

    cv::Mat gx;
    cv::Sobel(gammaCorrected2, gx, CV_16S, 1,0);

    cv::Mat gy;
    cv::Sobel(gammaCorrected1, gy, CV_16S, 0,1);

    cv::Mat abs_grad_x;
    cv::Mat abs_grad_y;
    cv::convertScaleAbs(gx, abs_grad_x);
    cv::convertScaleAbs(gy, abs_grad_y);

    cv::imshow("gradient x gamma", abs_grad_x);
    cv::waitKey(0);
    cv::imshow("gradient y gamma", abs_grad_y);
    cv::waitKey(0);

    cv::Mat grad_magn;
    cv::addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad_magn);

    cv::Mat equalized;
    cv::equalizeHist(gammaCorrected2,equalized);
    cv::imshow("equalized", equalized);
    cv::waitKey(0);

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

    // cv::imshow("gradient x eq", abs_grad_xeq);
    // cv::waitKey(0);
    // cv::imshow("gradient y eq", abs_grad_yeq);
    // cv::waitKey(0);

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

    cv::imshow("Normalized Absolute Laplacian", gammaCorrected2+normalized_abs_laplacian);  // Normalized for grayscale
    cv::waitKey(0);  // Wait for a key press indefinitely

    cv::Mat filtered_laplacian;
    cv::bilateralFilter(normalized_abs_laplacian, filtered_laplacian, -1, 10, 10);
    cv::imshow("filtered laplacian", filtered_laplacian);
    cv::waitKey(0);

    cv::imshow("gradient magnitude", grad_magn);
    cv::waitKey(0);

    cv::Mat medianblurred;
    cv::medianBlur(filtered_laplacian, medianblurred, 3);
    cv::Mat bilateralblurred;
    cv::bilateralFilter(medianblurred, bilateralblurred, -1,20,10);
    cv::imshow("bilateral filtered2", bilateralblurred);
    cv::waitKey(0);

    cv::Mat element = cv::getStructuringElement( 
                        cv::MORPH_RECT, cv::Size(3,3)); 

    cv::Mat erodeg; 
    cv::erode(bilateralblurred, erodeg, element, cv::Point(-1, -1), 1); 
    cv::imshow("erodeg", erodeg);

    //cv::Mat gmagthold;
    //cv::threshold( erodeg, gmagthold, 110, 255,  cv::THRESH_BINARY);
    //cv::imshow("gmagthold", gmagthold);
    //cv::waitKey(0);

    cv::Mat dilate; 
    cv::dilate(bilateralblurred, dilate, element, cv::Point(-1, -1), 5); 
    cv::imshow("dilated", dilate);

/*
   std::vector<int> angles = {-8, -9, -10, -11, -12, -13};
    std::vector<float> scales = {0.75, 1, 1.05, 1.1, 1.2, 2};
    std::vector<cv::RotatedRect> line_boxes;

    for(int k = 0; k<angles.size(); k++) {
        // Template size
        int template_height = 5*2+20*scales[k]*scales[k];
        int template_width = 100*scales[k];

        // Horizontal template and mask definition
        cv::Mat horizontal_template(template_height,template_width,CV_8U);
        cv::Mat horizontal_mask(template_height,template_width,CV_8U);

        // Build the template and mask
        for(int i = 0; i< horizontal_template.rows; i++) {
            for(int j = 0; j<horizontal_template.cols; j++) {
                uchar val = 0;
                if(i < 5 || (i > template_height-5) || j > template_width-5) {
                    val = 255;
                }
                horizontal_mask.at<uchar>(i,j) = val;
                horizontal_template.at<uchar>(i,j) = val;
            }
        }
*/
    // Maybe with an omography the trees are not in the middle of the dick
    std::vector<int> angles = {-7,-8,-9, -10, -11, -12, -15};
    std::vector<float> scales = {0.8, 0.8, 1, 1.05, 1.1, 1.2, 1.5};
    std::vector<cv::RotatedRect> line_boxes;
    std::vector<cv::Point2f> verts;

    for(int k = 0; k<angles.size(); k++) {
        // Template size
        int template_height = 39*scales[k];
        int template_width = 120*scales[k];

        // Horizontal template and mask definition
        cv::Mat horizontal_template(template_height,template_width,CV_8U,cv::Scalar(0));
        cv::Mat horizontal_mask(template_height,template_width,CV_8U,cv::Scalar(0));

        // Build the template and mask
        for(int i = 0; i< horizontal_template.rows; i++) {
            for(int j = 0; j<horizontal_template.cols; j++) {
                if(i<8 || j > template_width-8 || (i>(template_height-8)&& j > 20*scales[k]*scales[k])) {
                    horizontal_template.at<uchar>(i,j) = 190;
                }
                horizontal_mask.at<uchar>(i,j) = 255;
            }
        }

        cv::imshow("Horizontal template", horizontal_template);
        cv::waitKey(0);

        // Rotate the template
        cv::Mat R = cv::getRotationMatrix2D(cv::Point2f(0,template_height-1),angles[k],1);
        cv::Mat rotated_template;
        cv::Mat rotated_mask;

        float rotated_width = template_width*cos(-angles[k]*CV_PI/180)+template_height;
        float rotated_height = template_width*sin(-angles[k]*CV_PI/180)+template_height;

        cv::warpAffine(horizontal_template,rotated_template,R,cv::Size(rotated_width,rotated_height));
        cv::warpAffine(horizontal_mask,rotated_mask,R,cv::Size(rotated_width,rotated_height));

        cv::imshow("Rotated template", rotated_template);
        cv::waitKey(0);

        cv::Mat tm_result;
        //cv::filter2D(dilate, test, CV_32F, rotated);
        //cv::imshow("test", test);
        //cv::waitKey(0);

        // use dilate or medianblurred or canny with 100-1000
        cv::matchTemplate(dilate,rotated_template,tm_result,cv::TM_SQDIFF,rotated_mask);
        double min,max;
        cv::Point minloc(0,0), maxloc(0,0);
        cv::minMaxLoc(tm_result,&min,&max,&minloc,&maxloc);

        std::cout << "MIN VALUE AFTER NORMALIZING: " << min/max << std::endl;
        if(min/max > 0.25) continue;

        cv::normalize( tm_result, tm_result, 0, 1, cv::NORM_MINMAX, -1, cv::Mat() );
    
        cv::imshow("TM Result", tm_result);
        cv::waitKey(0);

        // Finding local minima
        cv::Mat eroded;
        std::vector<cv::Point> minima;
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(rotated_width*1.5, rotated_height*1.5));
        cv::erode(tm_result, eroded, kernel);
        cv::Mat localMinimaMask = (tm_result == eroded) & (tm_result <= 0.15 );

        cv::imshow("TM Result, eroded", eroded);
        cv::waitKey(0);

        // Find all non-zero points (local minima) in the mask
        cv::findNonZero(localMinimaMask, minima);

        // Draw bboxes of the found lines
        for (const cv::Point& pt : minima) {
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
                cv::line(image, vertices[i], vertices[(i+1) % 4], cv::Scalar(0, 255, 0), 2);
                verts.push_back(vertices[i]);
            }
        }
        cv::imshow("with lines", image);
        cv::waitKey(0);
    }

    float scoreThreshold = 0.0f;  // Minimum score to keep
    float nmsThreshold = 0.1f;    // IoU threshold for NMS
    std::vector<float> scores(line_boxes.size(),0.1f);

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
    std::cout << averageMinDistance << std::endl;

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

        std::cout << minDistance << std::endl;

        /*if(minDistance < averageMinDistance/2 || minDistance > averageMinDistance*2) {
            indices.erase(std::find(indices.begin(),indices.end(),idx));
        }*/
        if(minDistance < averageMinDistance*2) {
            // Draw the rotated rectangle
            cv::Point2f vertices[4];
            rect.points(vertices);
            for (int j = 0; j < 4; j++) {
                cv::line(image, vertices[j], vertices[(j + 1) % 4], cv::Scalar(0, 0, 255), 2);
            }
        }
    }

    // O uso questo come centri di k-means, o non uso NMS e quando disegno un nuovo rotatedrect controllo che il suo match sia migliore di quello precedente
    // rispetto al bbox che gli è più vicino
    // Display the image
    cv::imshow("NMS Result", image);
    cv::waitKey(0);

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