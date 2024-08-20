#include "parkingSpotDetector2.hpp"

// Function to detect parking spots in the images
void detectParkingSpots2(const std::vector<cv::Mat>& images, std::vector<ParkingSpot>& parkingSpots) {
    
    std::vector<std::vector<ParkingSpot>> parkingSpotsPerImage;
    for(const auto& image : images) {
        // Find parking spots for each image separately
        parkingSpotsPerImage.push_back(detectParkingSpotInImage2(image));
    }

    std::vector<ParkingSpot> parkingSpotNonMaxima = nonMaximaSuppression2(parkingSpotsPerImage, images[0].size());

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

    std::vector<int> angles = {-7,-8,-9, -10, -11, -12, -15};
    std::vector<float> scales = {0.5, 0.75, 1, 1.05, 1.1, 1.2, 2};

    for(int k = 0; k<angles.size(); k++) {
        int kheight = 39*scales[k];
        int kwidth = 145*scales[k];

        cv::Mat test_kernel(kheight,kwidth,CV_8U);
        for(int i = 0; i< test_kernel.rows; i++) {
            for(int j = 0; j<test_kernel.cols; j++) {
                if((i<4 && j < kwidth-100*scales[k]*scales[k]) || (i>(kheight-4)&& j > 20*scales[k]*scales[k])) {
                    test_kernel.at<uchar>(i,j) = 255;
                }
                else {
                    test_kernel.at<uchar>(i,j) = 0;
                }
            }
        }

        cv::imshow("added", test_kernel);
        cv::waitKey(0);

        cv::Mat R = cv::getRotationMatrix2D(cv::Point2f(19,77),angles[k],1);
        cv::Mat rotated;
        cv::warpAffine(test_kernel,rotated,R,cv::Size(101*scales[k],55*scales[k]));

        cv::imshow("rotated", rotated);
        cv::waitKey(0);

        cv::Mat test;
        //cv::filter2D(dilate, test, CV_32F, rotated);
        //cv::imshow("test", test);
        //cv::waitKey(0);

        cv::matchTemplate(dilate,rotated,test,cv::TM_SQDIFF,rotated);
        cv::normalize( test, test, 0, 1, cv::NORM_MINMAX, -1, cv::Mat() );
    
        std::vector<cv::Point> maxLocs;
        std::vector<cv::Point> centers;

        cv::imshow("match template output", test);
        cv::waitKey(0);

        cv::Mat tholdtest;
        cv::threshold( test, tholdtest, 0.1, 255,  cv::THRESH_BINARY_INV);

        cv::imshow("match template thold", tholdtest);
        cv::waitKey(0);

        for(int i = 0; i<500; i++) {
            double minVal; double maxVal; cv::Point minLoc; cv::Point maxLoc;
            cv::minMaxLoc( test, &minVal, &maxVal, &minLoc, &maxLoc, cv::Mat() );
            maxLoc = minLoc;
            maxLocs.push_back(maxLoc);
            cv::Point center;
            center.x = maxLoc.x + 101*scales[k] / 2;
            center.y = maxLoc.y + 55*scales[k] / 2;
            centers.push_back(center);
            test.at<float>(maxLoc) = 255;
        }

        for(int i = 0; i<500; i++) {
            cv::Point center = centers[i];
            cv::RotatedRect rotatedRect(center, cv::Size(101*scales[k],55*scales[k]), -angles[k]);
                    // Get the 4 vertices of the rotated rectangle
            cv::Point2f vertices[4];
            rotatedRect.points(vertices);
            // Draw the rotated rectangle using lines between its vertices
            for (int i = 0; i < 4; i++) {
                cv::line(gs, vertices[i], vertices[(i+1) % 4], cv::Scalar(0, 255, 0), 2);
            }
            parkingSpots.push_back(ParkingSpot{0,0,rotatedRect});

            //cv::imshow("BBOX", gs);
            //cv::waitKey(0);
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

std::vector<ParkingSpot> nonMaximaSuppression2(const std::vector<std::vector<ParkingSpot>>& parkingSpots, cv::Size imageSize) {
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

            if (isOverlapping2(allSpots[i].rect, allSpots[j].rect, imageSize)) {
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

std::vector<cv::Point> convertToIntPoints2(const std::vector<cv::Point2f>& floatPoints) {
    std::vector<cv::Point> intPoints;
    for (const auto& point : floatPoints) {
        intPoints.emplace_back(cv::Point(cv::saturate_cast<int>(point.x), cv::saturate_cast<int>(point.y)));
    }
    return intPoints;
}

// Function to calculate if two rectangles are overlapping
bool isOverlapping2(const cv::RotatedRect& rect1, const cv::RotatedRect& rect2, cv::Size imageSize) {
    // Vetices extraction
    std::vector<cv::Point2f> vertices1(4), vertices2(4);
    rect1.points(vertices1.data());
    rect2.points(vertices2.data());

    //  Conversion to integer points
    std::vector<cv::Point> intVertices1 = convertToIntPoints2(vertices1);
    std::vector<cv::Point> intVertices2 = convertToIntPoints2(vertices2);

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