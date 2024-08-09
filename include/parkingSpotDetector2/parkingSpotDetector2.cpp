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

    // cv::imshow("warped", warped_img);
    // cv::waitKey(0);

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
    cv::imshow("median blurred", medianblurred);
    cv::waitKey(0);

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