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
    src_pts[0] = cv::Point2f(555, 538);
    src_pts[1] = cv::Point2f(607, 629);
    src_pts[2] = cv::Point2f(721, 664);
    src_pts[3] = cv::Point2f(638, 559);

    cv::Point2f dst_pts[4];
    dst_pts[0] = cv::Point2f(1780-200, 799-200);
    dst_pts[1] = cv::Point2f(1819-200, 799-200);
    dst_pts[2] = cv::Point2f(1819-200, 749-200);
    dst_pts[3] = cv::Point2f(1780-200, 749-200);

    cv::Size warped_image_size = cv::Size(1820, 800);
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
    std::vector<int> angles = {-9};
    std::vector<float> scales = {1};

    for(int k = 0; k<angles.size(); k++) {
        int kheight = 39*scales[k];
        int kwidth = 145*scales[k];

        cv::Mat test_kernel(kheight,kwidth,CV_8U);
        for(int i = 0; i< test_kernel.rows; i++) {
            for(int j = 0; j<test_kernel.cols; j++) {
                if(i<4 || (i>(kheight-4)&& j > 20*scales[k]*scales[k])) {
                    test_kernel.at<uchar>(i,j) = 255;
                }
                else {
                    test_kernel.at<uchar>(i,j) = 0;
                }
            }
        }

        //cv::imshow("added", test_kernel);
        //cv::waitKey(0);

        cv::Mat R = cv::getRotationMatrix2D(cv::Point2f(19,77),angles[k],1);
        cv::Mat rotated;
        cv::warpAffine(test_kernel,rotated,R,cv::Size(101*scales[k],55*scales[k]));

        cv::imshow("rotated", rotated);
        cv::waitKey(0);

        cv::Mat test;
        //cv::filter2D(dilate, test, CV_32F, rotated);
        //cv::imshow("test", test);
        //cv::waitKey(0);

        cv::matchTemplate(dilate,rotated,test,cv::TM_CCORR_NORMED);
        cv::normalize( test, test, 0, 1, cv::NORM_MINMAX, -1, cv::Mat() );
    
        std::vector<cv::Point> maxLocs;
        std::vector<cv::Point> centers;

        for(int i = 0; i<1; i++) {
            double minVal; double maxVal; cv::Point minLoc; cv::Point maxLoc;
            cv::minMaxLoc( test, &minVal, &maxVal, &minLoc, &maxLoc, cv::Mat() );
            maxLocs.push_back(maxLoc);
            cv::Point center;
            center.x = maxLoc.x + 101*scales[k] / 2;
            center.y = maxLoc.y + 55*scales[k] / 2;
            centers.push_back(center);
            test.at<float>(maxLoc) = 0.0;
        }

        for(int i = 0; i<1; i++) {
            cv::Point center = centers[i];
            cv::RotatedRect rotatedRect(center, cv::Size(101*scales[k],55*scales[k]), -angles[k]);
                    // Get the 4 vertices of the rotated rectangle
            cv::Point2f vertices[4];
            rotatedRect.points(vertices);

            // Draw the rotated rectangle using lines between its vertices
            for (int i = 0; i < 4; i++) {
                cv::line(gs, vertices[i], vertices[(i+1) % 4], cv::Scalar(0, 255, 0), 2);
            }

            cv::Point2f dst_pts[4];
            dst_pts[0] = cv::Point2f(1819-500, kheight+200);
            dst_pts[1] = cv::Point2f(1819-500-kwidth, kheight+200);
            dst_pts[2] = cv::Point2f(1819-500-kwidth, 200);
            dst_pts[3] = cv::Point2f(1819-500, 200);

            cv::Size warped_image2_size = cv::Size(1820, 1280);
            cv::Mat M = cv::getPerspectiveTransform(vertices, dst_pts);
            cv::Mat warped_img2;
            cv::warpPerspective(image, warped_img2, M, warped_image2_size);

            cv::Mat gswrpd;
            cv::cvtColor(warped_img2, gswrpd, cv::COLOR_BGR2GRAY);

            cv::imshow("warped", gswrpd);
            cv::waitKey(0);

            cv::Mat gammaCorrected1 = applyGammaTransform(gswrpd, gammaValue);
            cv::imshow("gamma tfff", gammaCorrected1);
            cv::waitKey(0);

            gammaValue = 2;
            cv::Mat gammaCorrected2 = applyGammaTransform(gammaCorrected1, gammaValue);

            cv::Mat gx;
            cv::Sobel(gammaCorrected2, gx, CV_8U, 1,0);

            cv::imshow("gradient x2", gx);
            cv::waitKey(0);

            cv::Mat gy;
            cv::Sobel(gammaCorrected2, gy, CV_8U, 0,1);

            cv::imshow("gradient y2", gy);
            cv::waitKey(0);

            cv::Mat grad_magn = gx + gy;

            cv::imshow("gradient magnitude2", grad_magn);
            cv::waitKey(0);

            cv::imwrite("test.png", gx);

            cv::imshow("Detected Lines", warped_img2);
            cv::waitKey(0);

        }
    }
        cv::imshow("gs2", gs );
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