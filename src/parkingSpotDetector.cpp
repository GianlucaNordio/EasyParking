#include "parkingSpotDetector.hpp"
#include <opencv2/ximgproc.hpp>

/*
TODO: 
1. Preprocessing (pensare a come rendere invariante alle condizioni climatiche)
2. Generare meglio i template
3. Chiedere nel forum quanti parametri possiamo usare
4. Stesso size, angolo diverso: usare tm_result_unnormed come score, poi tra tutti quelli che overlappano per tipo l'80% tenere quello con score migliore
5. Dopo il punto 4, alla fine dei due cicli for, fare non maxima suppression. A quel punto usare NMS di opencv oppure prendere quello con area maggiore
6. Try again with homography
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

    cv::imshow("bilateral filtered", filteredImage);
    cv::waitKey(0);

    cv::Mat gs;
    cv::cvtColor(filteredImage, gs, cv::COLOR_BGR2GRAY);

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
    cv::Sobel(gs, gx, CV_16S, 1,0);

    cv::Mat gy;
    cv::Sobel(gs, gy, CV_16S, 0,1);

    cv::Mat abs_grad_x;
    cv::Mat abs_grad_y;
    cv::convertScaleAbs(gx, abs_grad_x);
    cv::convertScaleAbs(gy, abs_grad_y);

    // cv::imshow("gradient x gamma", abs_grad_x);
    // cv::waitKey(0);
    // cv::imshow("gradient y gamma", abs_grad_y);
    // cv::waitKey(0);

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

    //cv::imshow("Normalized Absolute Laplacian", normalized_abs_laplacian);  // Normalized for grayscale
    //cv::waitKey(0);  // Wait for a key press indefinitely

    cv::Mat filtered_laplacian;
    cv::bilateralFilter(normalized_abs_laplacian, filtered_laplacian, -1, 40, 10);
    cv::imshow("filtered laplacian", filtered_laplacian);
    cv::waitKey(0);

    cv::imshow("gradient magnitude", grad_magn);
    cv::waitKey(0);

    cv::Mat grad_magn_bilateral;
    cv::bilateralFilter(grad_magn, grad_magn_bilateral, -1, 20, 10);
    cv::imshow("grad magn bilateral", grad_magn_bilateral);
    cv::waitKey(0);

    cv::Mat edges;
    cv::Canny(grad_magn, edges, 50, 400);
    cv::imshow("canny", edges);
    cv::waitKey(0);

    cv::Mat medianblurred;
    cv::medianBlur(grad_magn_bilateral, medianblurred, 3);
    cv::Mat bilateralblurred;
    //cv::bilateralFilter(medianblurred, bilateralblurred, -1,20,10);
    //cv::imshow("bilateral filtered2", medianblurred);
    cv::waitKey(0);

    cv::Mat element = cv::getStructuringElement( 
                        cv::MORPH_CROSS, cv::Size(3,3)); 

    cv::dilate(edges,edges,element,cv::Point(-1,1),1);
    cv::imshow("dilated canny", edges);
    cv::waitKey(0);

    cv::Mat erodeg; 
    cv::erode(edges, edges, element, cv::Point(-1, -1), 1); 
    cv::imshow("opened canny",edges);
    cv::waitKey(0);

    //cv::Mat gmagthold;
    //cv::threshold( erodeg, gmagthold, 110, 255,  cv::THRESH_BINARY);
    //cv::imshow("gmagthold", gmagthold);
    //cv::waitKey(0);

    cv::Mat dilate = grad_magn+normalized_abs_laplacian; 
    // ok with 5-6 dilations for normal and flipped. Good with 8 for normal
    cv::Mat dilate_pre_filter;
    //cv::dilate(grad_magn, dilate_pre_filter, element, cv::Point(-1, -1), 1);
    //cv::bilateralFilter(dilate_pre_filter, dilate, -1, 20, 10);
    // dilate = grad_magn;
    //dilate = applyGammaTransform(dilate,1.2);
    dilate = grad_magn+normalized_abs_laplacian;
    // cv::threshold(dilate,dilate,80,255,cv::THRESH_BINARY);
    // cv::imshow("dilated", dilate);
// 
    // cv::dilate(dilate,erodeg,element, cv::Point(-1, -1), 2);
    // cv::imshow("erodeg",erodeg);
// 
    // cv::erode(erodeg,erodeg,element, cv::Point(-1, -1), 1);
    // cv::imshow("erodeg",erodeg);

    // Probabilistic Line Transform
    std::vector<cv::Vec4i> linesP; // will hold the results of the detection
    cv::HoughLinesP(edges, linesP, 6, CV_PI/180, 60, 70, 15 ); // runs the actual detection
    // Draw the lines
    for( size_t i = 0; i < linesP.size(); i++ )
    {
        cv::Vec4i l = linesP[i];
        line( image, cv::Point(l[0], l[1]), cv::Point(l[2], l[3]), cv::Scalar(0,0,255), 3, cv::LINE_AA);
    }

    cv::imshow("result",image);
    cv::waitKey(0);

    // Create a FastLineDetector object
    cv::Ptr<cv::ximgproc::FastLineDetector> fld = cv::ximgproc::createFastLineDetector(20,2,100,500,0,true);

    // Detect lines in the image
    std::vector<cv::Vec4f> lines;
    fld->detect(edges, lines);

    // Draw the detected lines on the original image
    cv::Mat img_with_lines;
    cv::cvtColor(gs, img_with_lines, cv::COLOR_GRAY2BGR);

    for (size_t i = 0; i < lines.size(); i++) {
        cv::Vec4f l = lines[i];
        cv::line(img_with_lines, cv::Point(l[0], l[1]), cv::Point(l[2], l[3]), cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
    }

    // Display the result
    cv::imshow("Detected Lines", img_with_lines);
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