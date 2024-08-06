#include "parkingSpotDetector.hpp"

// Function to detect parking spots in the images
void detectParkingSpot(const std::vector<cv::Mat>& images, std::vector<ParkingSpot>& parkingSpot) {
    
    std::vector<std::vector<ParkingSpot>> parkingSpotPerImage;
    for(const auto& image : images) {
        // Find parking spots for each image separately
        parkingSpotPerImage.push_back(detectParkingSpotInImage(image));
    }

    // Non maxima suppression to remove overlapping bounding boxes
    std::vector<ParkingSpot> parkingSpotNonMaxima = nonMaximaSuppression(parkingSpotPerImage);
    
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
    cv::imshow("Detected Parking Spots", toprint);
    cv::waitKey(0);

}

// This function detects the parking spots in a single image
std::vector<ParkingSpot> detectParkingSpotInImage(const cv::Mat& image) {
    std::vector<ParkingSpot> parkingSpots;


    // Apply bilateral filter to smooth the image
    int diameter = 15; // Diameter of the filter pixel
    double sigmaColor = 75; // Standard deviation in color space
    double sigmaSpace = 75; // Standard deviation in coordinate space

    // Apply the bilateral filter
    cv::Mat filteredImage;
    cv::bilateralFilter(image, filteredImage, diameter, sigmaColor, sigmaSpace);

    // Convert the image to grayscale
    cv::Mat gray;
    cv::cvtColor(filteredImage, gray, cv::COLOR_BGR2GRAY);

    // Threshold the grayscale image to keep only values above 245, set the rest to 0
    cv::threshold(gray, gray, 245, 255, cv::THRESH_BINARY);

    // Apply Canny edge detection to find edges
    cv::Mat edges;
    cv::Canny(gray, edges, 50, 150);

    // Display the edges
    cv::imshow("Edges", edges);
    cv::waitKey(0);

    // Dilate the edges to connect gaps between lines
    cv::Mat dilatedEdges;
    cv::dilate(edges, dilatedEdges, cv::Mat(), cv::Point(-1, -1), 2);

    // Find contours of the areas bounded by the lines
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(dilatedEdges, contours, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);

    // Prepare to store the detected parking spots
   int counter = 0;

    // Process each contour to detect parking spots
    for (const auto& contour : contours) {
    if (contour.size() < 5) continue; // cv::fitEllipse requires at least 5 points

    // Find the minimum area rectangle that encloses the contour
    cv::RotatedRect rect = cv::minAreaRect(contour);

    // Filter the rectangle based on its angle to be approximately ±45°
    float angle = rect.angle;
    if (angle < -45.0) angle += 90.0;
    if (angle > 45.0) angle -= 90.0;

    // Check if the angle is approximately ±45°
    if (std::abs(angle - 45.0) < 10.0 || std::abs(angle + 45.0) < 10.0) {
    parkingSpots.push_back(ParkingSpot{counter, 0, rect});
    counter++;
    }
    }

    // Display the dilated edges
    cv::imshow("Dilated Edges", dilatedEdges);
    cv::waitKey(0);


    // TODO --> show the result (test)
    for (const auto& spot : parkingSpots) {
        cv::Point2f vertices[4];
        spot.rect.points(vertices);
        for (int i = 0; i < 4; ++i) {
            cv::line(image, vertices[i], vertices[(i + 1) % 4], cv::Scalar(0, 255, 0), 2);
        }
    }
    cv::imshow("Detected Parking Spots", image);
    cv::waitKey(0);

    return parkingSpots;
}

std::vector<ParkingSpot> nonMaximaSuppression(const std::vector<std::vector<ParkingSpot>>& parkingSpots) {
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

            if (isOverlapping(allSpots[i].rect, allSpots[j].rect)) {
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
bool isOverlapping(const cv::RotatedRect& rect1, const cv::RotatedRect& rect2) {
    // Vetices extraction
    std::vector<cv::Point2f> vertices1(4), vertices2(4);
    rect1.points(vertices1.data());
    rect2.points(vertices2.data());

    //  Conversion to integer points
    std::vector<cv::Point> intVertices1 = convertToIntPoints(vertices1);
    std::vector<cv::Point> intVertices2 = convertToIntPoints(vertices2);

    // Binary images creation
    cv::Mat img1 = cv::Mat::zeros(500, 500, CV_8UC1);
    cv::Mat img2 = cv::Mat::zeros(500, 500, CV_8UC1);

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
