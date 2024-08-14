#include "parkingSpotDetector.hpp"

// Function to detect parking spots in the images
void detectParkingSpot(const std::vector<cv::Mat>& images, std::vector<ParkingSpot>& parkingSpot) {
    
    std::vector<std::vector<ParkingSpot>> parkingSpotPerImage;
    for(const auto& image : images) {
        // Find parking spots for each image separately
        parkingSpotPerImage.push_back(detectParkingSpotInImage(image));
    }

    /* TODO DECOMENTARE!!!!!!!

    // Non maxima suppression to remove overlapping bounding boxes
    std::vector<ParkingSpot> parkingSpotNonMaxima = nonMaximaSuppression(parkingSpotPerImage, images[0].size());
    
    // Build the 2D map of parking spots


    // TODO --> show the result (test)
    cv::Mat toprint = images[0].clone();

    for (const auto& spot : parkingSpotNonMaxima) {
        cv::Point2f vertices[4];
        spot.rect.points(vertices);
        for (int i = 0; i < 4; ++i) {
            cv::line(toprint, vertices[i], vertices[(i + 1) % 4], cv::Scalar(0, 255, 0), 2);
        }
    }
    //cv::imshow("Detected Parking Spots", toprint);
    //cv::waitKey(0);
    */
}

// This function detects the parking spots in a single image
std::vector<ParkingSpot> detectParkingSpotInImage(const cv::Mat& image) {

    /*
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

    cv::Mat medianblurred;
    cv::bilateralFilter(grad_magn, medianblurred, -1, 20, 10);
    //cv::imshow("median blurred", medianblurred);
    //cv::waitKey(0);

    cv::Mat gmagthold;
    cv::threshold( medianblurred, gmagthold, 125, 255,  cv::THRESH_BINARY);
    //cv::imshow("gmagthold", gmagthold);
    //cv::waitKey(0);

    cv::Mat element = cv::getStructuringElement( 
                        cv::MORPH_RECT, cv::Size(3,3)); 
    cv::Mat dilate; 
    cv::dilate(gmagthold, dilate, element, cv::Point(-1, -1), 1); 
    //cv::imshow("dilated", dilate);
    //cv::waitKey(0);

    // TODO: choose which image may give the best information, then try to use a sliding window approach
    /*cv::Mat gmagthold;
    cv::threshold( grad_magn, gmagthold, 100, 255,  cv::THRESH_BINARY);
    cv::imshow("gmagthold", gmagthold);
    cv::waitKey(0);

    cv::Mat lap;
    cv::Laplacian(gs,lap,CV_8U);

    cv::Mat int1 = gy-gx+gammaCorrected2;
    cv::imshow("Intermediate 1", gy-gx+gammaCorrected2);
    cv::waitKey(0);

    cv::Mat gythold;
    cv::threshold( gy, gythold, 200, 255,  cv::THRESH_BINARY);
    cv::imshow("gythold", gythold);
    cv::waitKey(0);

    cv::Mat res = lap + gythold;
    cv::imshow("Laplacian + gradY", res);
    cv::waitKey(0);
    */

    // Detect lines using Hough Line Transform
    // std::vector<cv::Vec4i> lines;
    // int minLen = 55;
    // cv::HoughLinesP(edges, lines, 1, CV_PI / 180, 50, minLen, 20);

    
    // // Draw lines on the image
    // for (const auto& line : lines) {
    //     cv::line(warped_img, cv::Point(line[0], line[1]), cv::Point(line[2], line[3]), cv::Scalar(0, 255, 0), 2);
    // }

    // cv::imshow("Detected Lines", warped_img);
    // cv::waitKey(0);


    std::vector<ParkingSpot> parkingSpots;

    cv::Mat filteredImage;
    cv::bilateralFilter(image, filteredImage, -1, 40, 10);

    //cv::imshow("bilateral filtered", filteredImage);
    //cv::waitKey(0);

    cv::Mat gs;
    cv::cvtColor(filteredImage, gs, cv::COLOR_BGR2GRAY);

    //cv::imshow("grayscale", gs);
    //cv::waitKey(0);

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

    //cv::imshow("gradient magnitude", grad_magn);
    //cv::waitKey(0);

    cv::Mat medianblurred;
    cv::bilateralFilter(grad_magn, medianblurred, -1, 20, 10);
    //cv::imshow("median blurred", medianblurred);
    //cv::waitKey(0);

    cv::Mat gmagthold;
    cv::threshold( medianblurred, gmagthold, 125, 255,  cv::THRESH_BINARY);
    //cv::imshow("gmagthold", gmagthold);
    //cv::waitKey(0);

    cv::Mat element = cv::getStructuringElement( 
                        cv::MORPH_RECT, cv::Size(3,3)); 
    cv::Mat dilate; 
    cv::dilate(gmagthold, dilate, element, cv::Point(-1, -1), 1); 
    //cv::imshow("dilated", dilate);
    //cv::waitKey(0);

    // TODO: choose which image may give the best information, then try to use a sliding window approach
    /*cv::Mat gmagthold;
    cv::threshold( grad_magn, gmagthold, 100, 255,  cv::THRESH_BINARY);
    cv::imshow("gmagthold", gmagthold);
    cv::waitKey(0);

    cv::Mat lap;
    cv::Laplacian(gs,lap,CV_8U);

    cv::Mat int1 = gy-gx+gammaCorrected2;
    cv::imshow("Intermediate 1", gy-gx+gammaCorrected2);
    cv::waitKey(0);

    cv::Mat gythold;
    cv::threshold( gy, gythold, 200, 255,  cv::THRESH_BINARY);
    cv::imshow("gythold", gythold);
    cv::waitKey(0);

    cv::Mat res = lap + gythold;
    cv::imshow("Laplacian + gradY", res);
    cv::waitKey(0);
    */

    // Detect lines using Hough Line Transform
    // std::vector<cv::Vec4i> lines;
    // int minLen = 55;
    // cv::HoughLinesP(edges, lines, 1, CV_PI / 180, 50, minLen, 20);

    
    // // Draw lines on the image
    // for (const auto& line : lines) {
    //     cv::line(warped_img, cv::Point(line[0], line[1]), cv::Point(line[2], line[3]), cv::Scalar(0, 255, 0), 2);
    // }

    // cv::imshow("Detected Lines", warped_img);
    // cv::waitKey(0);
    
    // Detect lines using Hough Line Transform
    int minLen = 10;
    cv::createTrackbar("Dil", "Detected Lines", nullptr, 200, 0);
    cv::setTrackbarPos("Dil", "Detected Lines", minLen);
    cv::imshow("Dilate", gmagthold);
    cv::waitKey();
    std::vector<cv::Vec4i> lines;
    cv::HoughLinesP(gmagthold, lines, 1, CV_PI / 2, 7, minLen, 30);
    // Denominator of CV_PI /... is the number of bins we have
    
    // Draw lines on the image
    for (const auto& line : lines) {
        cv::line(image, cv::Point(line[0], line[1]), cv::Point(line[2], line[3]), cv::Scalar(0, 255, 0), 2);
    }

    cv::imshow("Detected Lines", image);
    cv::waitKey(0);


    return parkingSpots;

    /*
    // Convert the image to grayscale
    cv::Mat gray;
    cv::cvtColor(filteredImage, gray, cv::COLOR_BGR2GRAY);

    // Apply Canny edge detection to find edges
    cv::Mat edges;
    cv::Canny(gray, edges, 50, 150);

    // Detect lines using Hough Line Transform
    std::vector<cv::Vec4i> lines;
    int minLen = 55;
    cv::HoughLinesP(edges, lines, 1, CV_PI / 180, 50, minLen, 20);

    
    // Draw lines on the image
    for (const auto& line : lines) {
        cv::line(image, cv::Point(line[0], line[1]), cv::Point(line[2], line[3]), cv::Scalar(0, 255, 0), 2);
    }

    cv::imshow("Detected Lines", image);
    cv::waitKey(0);

    // Find contours of the areas bounded by the lines
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(edges, contours, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);

    // Prepare to store the detected parking spots
    int counter = 0;

    // Process each contour to detect parking spots
    for (const auto& contour : contours) {
    if (contour.size() < 5) continue; // cv::fitEllipse requires at least 5 points

    // Find the minimum area rectangle that encloses the contour
    cv::RotatedRect rect = cv::minAreaRect(contour);
    parkingSpots.push_back(ParkingSpot{counter, 0, rect});
    counter++;
    }

    // TODO --> show the result (test)
    for (const auto& spot : parkingSpots) {
        cv::Point2f vertices[4];
        spot.rect.points(vertices);
        for (int i = 0; i < 4; ++i) {
            cv::line(image, vertices[i], vertices[(i + 1) % 4], cv::Scalar(0, 255, 0), 2);
        }
    }
    //cv::imshow("Detected Parking Spots", image);
    //cv::waitKey(0);
    return parkingSpots;
    */
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


std::vector<ParkingSpot> nonMaximaSuppression(const std::vector<std::vector<ParkingSpot>>& parkingSpots, cv::Size imageSize) {
    std::vector<ParkingSpot> result;
    int id = 0;
    // Union of all parking spots
    std::vector<ParkingSpot> allSpots;
    for (const auto& spots : parkingSpots) {
        allSpots.insert(allSpots.end(), spots.begin(), spots.end());
    }

    // Vector to keep track of which spots have been considered
    std::vector<bool> considered(allSpots.size(), false);

    for (size_t i = 0; i < allSpots.size(); ++i) {
        if (considered[i]) continue;

        std::vector<cv::Point2f> centers;
        std::vector<cv::Size2f> sizes;

        centers.push_back(allSpots[i].rect.center);
        sizes.push_back(allSpots[i].rect.size);

        considered[i] = true;

        for (size_t j = i + 1; j < allSpots.size(); ++j) {
            if (considered[j]) continue;

            if (isOverlapping(allSpots[i].rect, allSpots[j].rect, imageSize)) {
                centers.push_back(allSpots[j].rect.center);
                sizes.push_back(allSpots[j].rect.size);

                considered[j] = true;
            }
        }

        // Compute the average center
        cv::Point2f avgCenter(0, 0);
        for (const auto& center : centers) {
            avgCenter += center;
        }
        avgCenter.x /= centers.size();
        avgCenter.y /= centers.size();

        // Compute the average size
        cv::Size2f avgSize(0, 0);
        for (const auto& size : sizes) {
            avgSize.width += size.width;
            avgSize.height += size.height;
        }
        avgSize.width /= sizes.size();
        avgSize.height /= sizes.size();


        result.push_back(ParkingSpot{id, 0, cv::RotatedRect(avgCenter, avgSize, 0)});
        id++;
    }

    return result;
}

std::vector<cv::Point> convertToIntPoints(const std::vector<cv::Point2f>& floatPoints) {
    std::vector<cv::Point> intPoints;
    for (const auto& point : floatPoints) {
        intPoints.emplace_back(cv::Point(cv::saturate_cast<int>(point.x), cv::saturate_cast<int>(point.y)));
    }
    return intPoints;
}

// Function to calculate if two rectangles are overlapping
bool isOverlapping(const cv::RotatedRect& rect1, const cv::RotatedRect& rect2, cv::Size imageSize) {
    // Vetices extraction
    std::vector<cv::Point2f> vertices1(4), vertices2(4);
    rect1.points(vertices1.data());
    rect2.points(vertices2.data());

    //  Conversion to integer points
    std::vector<cv::Point> intVertices1 = convertToIntPoints(vertices1);
    std::vector<cv::Point> intVertices2 = convertToIntPoints(vertices2);

    // Binary images creation
    cv::Mat img1 = cv::Mat::zeros(imageSize, CV_8UC1);
    cv::Mat img2 = cv::Mat::zeros(imageSize, CV_8UC1);

    // TODO --> show the result (test)
    cv::Mat combined(img1.rows, img1.cols * 2, img1.type());
    img1.copyTo(combined(cv::Rect(0, 0, img1.cols, img1.rows)));
    img2.copyTo(combined(cv::Rect(img1.cols, 0, img2.cols, img2.rows)));
    //cv::imshow("Immagini Combinate", combined);
    //cv::waitKey(0);

    // Draw the rectangles
    std::vector<std::vector<cv::Point>> contours1{intVertices1}, contours2{intVertices2};
    cv::drawContours(img1, contours1, 0, cv::Scalar(255), cv::FILLED);
    cv::drawContours(img2, contours2, 0, cv::Scalar(255), cv::FILLED);

    

    // Intersection of the two rectangles
    cv::Mat intersection;
    cv::bitwise_and(img1, img2, intersection);

    // Area calculation
    double area1 = cv::contourArea(intVertices1);
    double area2 = cv::contourArea(intVertices2);
    double intersectionArea = cv::countNonZero(intersection);

    // Overlapping condition
    return (intersectionArea >= 0.2 * area1) || (intersectionArea >= 0.2 * area2);
}