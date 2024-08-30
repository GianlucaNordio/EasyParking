#include "parkingSpotDetector2.hpp"

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
    cv::cvtColor(filteredImage, gs, cv::COLOR_BGR2GRAY);

    cv::imshow("grayscale", gs);
    cv::waitKey(0);

    // Set the gamma value
    double gammaValue = 1.25; // Example gamma value

    // Apply gamma transformation
    cv::Mat gammaCorrected1 = applyGammaTransform(gs, gammaValue);
    // cv::imshow("gamma tf1", gammaCorrected1);
    // cv::waitKey(0);

    gammaValue = 2;
    cv::Mat gammaCorrected2 = applyGammaTransform(gammaCorrected1, gammaValue);
    // cv::imshow("gamma tf2", gammaCorrected2);
    // cv::waitKey(0);

    cv::Mat gsthold;
    cv::threshold( gammaCorrected2, gsthold, 180, 255,  cv::THRESH_BINARY);
    // cv::imshow("gsthold", gsthold);
    // cv::waitKey(0);

    // cv::Mat equalized;
    // cv::equalizeHist(gs,equalized);

    // cv::imshow("equalized", equalized);
    // cv::waitKey(0);

    // Apply Canny edge detection to find edges
    cv::Mat gx;
    cv::Sobel(gammaCorrected2, gx, CV_8U, 1,0);

    // cv::imshow("gradient x", gx);
    // cv::waitKey(0);

    cv::Mat gy;
    cv::Sobel(gammaCorrected2, gy, CV_8U, 0,1);

    // cv::imshow("gradient y", gy);
    // cv::waitKey(0);

    cv::Mat grad_magn = gx + gy;

    cv::imshow("gradient magnitude", grad_magn);
    cv::waitKey(0);

    cv::imwrite("grad_magn.png", grad_magn);

    cv::Mat medianblurred;
    cv::bilateralFilter(grad_magn, medianblurred, -1, 20, 10);
    //cv::imshow("median blurred", medianblurred);
    //cv::waitKey(0);
    cv::imwrite("grad_magn_filt.png", medianblurred);


    cv::Mat gmagthold;
    cv::threshold( medianblurred, gmagthold, 125, 255,  cv::THRESH_BINARY);
    //cv::imshow("gmagthold", gmagthold);
    //cv::waitKey(0);

    cv::Mat element = cv::getStructuringElement( 
                        cv::MORPH_RECT, cv::Size(3,3)); 
    cv::Mat dilate; 
    cv::dilate(gmagthold, dilate, element, cv::Point(-1, -1), 1); 
    cv::imshow("dilated", dilate);

    std::vector<int> angles = {-8, -9, -10, -11, -12, -15};
    std::vector<float> scales = {0.75, 1, 1.05, 1.1, 1.2, 2};

    for(int k = 0; k<angles.size(); k++) {
        // Template size
        int template_height = 5;
        int template_width = 100*scales[k];

        // Horizontal template and mask definition
        cv::Mat horizontal_template(template_height,template_width,CV_8U);
        cv::Mat horizontal_mask(template_height,template_width,CV_8U);

        // Build the template and mask
        for(int i = 0; i< horizontal_template.rows; i++) {
            for(int j = 0; j<horizontal_template.cols; j++) {
                horizontal_template.at<uchar>(i,j) = 200;
                horizontal_mask.at<uchar>(i,j) = 255;
            }
        }

        cv::imshow("Horizontal template", horizontal_template);
        cv::waitKey(0);

        // Rotate the template
        cv::Mat R = cv::getRotationMatrix2D(cv::Point2f(0,0),angles[k],1);
        cv::Mat rotated_template;
        cv::Mat rotated_mask;

        float rotated_width = template_width*cos(-angles[k]*CV_PI/180);
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
        cv::matchTemplate(medianblurred,rotated_template,tm_result,cv::TM_SQDIFF,rotated_mask);
        cv::normalize( tm_result, tm_result, 0, 1, cv::NORM_MINMAX, -1, cv::Mat() );
    
        cv::imshow("TM Result", tm_result);
        cv::waitKey(0);

        // Finding local minima
        cv::Mat eroded;
        std::vector<cv::Point> minima;
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(rotated_width, rotated_height));
        cv::erode(tm_result, eroded, kernel);
        cv::Mat localMinimaMask = (tm_result == eroded) & (tm_result <= 0.4 );

        cv::imshow("TM Result, eroded", eroded);
        cv::waitKey(0);

        // Find all non-zero points (local minima) in the mask
        findNonZero(localMinimaMask, minima);

        // Draw bboxes of the found lines
        for (const cv::Point& pt : minima) {
            // White circles at minima points
            // cv::circle(gs, pt, 3, cv::Scalar(255), 1);

            // Get center of the bbox to draw the rotated rect
            cv::Point center;
            center.x = pt.x+rotated_width/2;
            center.y = pt.y+rotated_height/2;
            
            cv::RotatedRect rotatedRect(center, cv::Size(template_width,template_height), -angles[k]);
            cv::Point2f vertices[4];
            rotatedRect.points(vertices);
            // Draw the rotated rectangle using lines between its vertices
            for (int i = 0; i < 4; i++) {
                cv::line(image, vertices[i], vertices[(i+1) % 4], cv::Scalar(0, 255, 0), 2);
            }
        }
        cv::imshow("with lines", image);
        cv::waitKey(0);
    }

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