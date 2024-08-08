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

    // (182, 571) (302, 515) (70, 197) (160, 160)
    cv::Point2f src_pts[4];
    src_pts[0] = cv::Point2f(182, 571);
    src_pts[1] = cv::Point2f(302, 515);
    src_pts[2] = cv::Point2f(70, 197);
    src_pts[3] = cv::Point2f(160, 160);

    cv::Point2f dst_pts[4];
    dst_pts[0] = cv::Point2f(182, 571);
    dst_pts[1] = cv::Point2f(302, 571);
    dst_pts[2] = cv::Point2f(182, 166);
    dst_pts[3] = cv::Point2f(302, 166);

    cv::Size warped_image_size = cv::Size(1820, 1280);
    cv::Mat M = cv::getPerspectiveTransform(src_pts, dst_pts);
    cv::Mat warped_img;
    cv::warpPerspective(image, warped_img, M, warped_image_size);

    cv::imshow("warped", warped_img);
    cv::waitKey(0);

    cv::cvtColor(warped_img, warped_img, cv::COLOR_BGR2GRAY);

    cv::imshow("gs", warped_img);
    cv::waitKey(0);

    return parkingSpots;
}