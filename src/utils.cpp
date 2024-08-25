#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp> 
#include <iostream>

#include <filesystem>

#include "utils.hpp"


namespace fs = std::filesystem;

const std::string FRAMES_FOLDER = "frames";
const int BASE_SEQUENCE_INDEX = 0;
const std::string SEQUENCE = "sequence";


const std::string SLASH = "/";

const std::string MASKS_FOLDER = "masks";


cv::Mat produceSingleImage(const std::vector<cv::Mat>& images, int imagesPerLine) {
    // TODO understand if the exceptions I throw are okay
    
    // Check if at least one  image was provided
    if(images.size() < 1) {
        throw std::invalid_argument("No image was provided");
    }
    
    // Check if all images have the same size
    int requiredCols = images[0].cols;
    int requiredRows = images[0].rows;
    for(int i = 1; i < images.size(); i++) {
        if(requiredCols != images[i].cols) {
            throw std::invalid_argument("Images have different number of columns");
        }
        else if(requiredRows != images[i].rows) {
            throw std::invalid_argument("Images have different number of rows");
        }
    }
    // If total number of images smaller then images per line reduce the images per line
    if(imagesPerLine > images.size()) {
        imagesPerLine = images.size();
    }

    // Initialize the final image
    int imagesPerColumn = std::ceil(images.size() / static_cast<float>(imagesPerLine));
    int totalCols = images[0].cols * imagesPerLine;
    int totalRows = images[0].rows * imagesPerColumn;
    cv::Mat result = cv::Mat::zeros(cv::Size(totalCols, totalRows), images[0].type());
    
    // Copy the images into the final image
    for(int i = 0; i < images.size(); i++) {
        int x = i % imagesPerLine;
        int y = i / imagesPerLine;
        cv::Mat position(result,
                        cv::Range(images[i].rows * y, images[i].rows * y + images[i].rows),
                        cv::Range(images[i].cols * x, images[i].cols * x + images[i].cols)
                        );
        images[i].copyTo(position);
    }

    // TODO remove this resize (used only to be able to fit the image in our screen)
    cv::resize(result, result, cv::Size(result.cols/2, result.rows/2));

    return result;
}


void loadBaseSequenceFrames(const std::string& datasetPath, std::vector<cv::Mat> &images) {
    std::string folderPath = datasetPath + SLASH + SEQUENCE + std::to_string(BASE_SEQUENCE_INDEX) + SLASH + FRAMES_FOLDER;
    loadImages(folderPath, images);
}

void loadSequencesFrames(const std::string& datasetPath, int numSequences, std::vector<std::vector<cv::Mat>> &images) {
    for(int i = 1; i <= numSequences; i++) {
        std::vector<cv::Mat> empty;   
        images.push_back(empty);
        std::string folderPath = datasetPath + SLASH + SEQUENCE + std::to_string(i) + SLASH + FRAMES_FOLDER;
        loadImages(folderPath, images[i - 1]);
    }
}

void loadImages(std::string path, std::vector<cv::Mat> &images) {
    for(const auto& entry : std::filesystem::directory_iterator(path)) {
        if(entry.is_regular_file()){
            std::string filePath = entry.path().string();
            cv::Mat image = cv::imread(filePath);
            if(!image.empty()){
                images.push_back(image);
                std::cout << "Read image: " << filePath << std::endl; // TODO maybe use exception
            } else
                std::cerr << "Could not read image: " << filePath << std::endl; // TODO maybe use exception
        }
    }
}


void loadSequencesSegMasks(const std::string& datasetPath, const int numSequences, std::vector<std::vector<cv::Mat>> &segMasks) {
    // Move over all the sequences
    for(int i = 0; i < numSequences; i++) {
        // Initialize the vector relative to the i-th sequence
        std::vector<cv::Mat> empty;   
        segMasks.push_back(empty);

        // Create the path relative to the i-th sequence
        std::string folderPath = datasetPath + SLASH + SEQUENCE + std::to_string(i + 1) + SLASH + MASKS_FOLDER;
        loadImages(folderPath, segMasks[i]);
        // Conver the images to greyscale
        for(int j = 0; j < segMasks[i].size(); j++) {
            cv::cvtColor(segMasks[i][j], segMasks[i][j], cv::COLOR_BGR2GRAY);
        }
    }
}

void convertGreyscaleToBGR(const std::vector<std::vector<cv::Mat>> &srcImages, std::vector<std::vector<cv::Mat>> &dstImages) {
    for(int i = 0; i < srcImages.size(); i++) {
        for(int j = 0; j < srcImages[i].size(); j++) {
            dstImages[i][j] =  cv::Mat(srcImages[i][j].rows,srcImages[i][j].cols, CV_8UC3, cv::Scalar(0 ,0 ,0));
            cv::Mat &dst = dstImages[i][j];
            const cv::Mat &src = srcImages[i][j];
            for(int x = 0; x < src.cols; x++) {
                for(int y = 0; y < src.rows; y++) {
                    if (src.at<uchar>(y, x) == 0) {
                        cv::Vec3b &color = dst.at<cv::Vec3b>(y, x);
                        color[0] = 128;
                        color[1] = 128;
                        color[2] = 128;
                    }
                    if (src.at<uchar>(y, x) == 1) {
                        cv::Vec3b &color = dst.at<cv::Vec3b>(y, x);
                        color[0] = 255;
                        color[1] = 0;
                        color[2] = 0;
                    }
                    if (src.at<uchar>(y, x) == 2) {
                        cv::Vec3b &color = dst.at<cv::Vec3b>(y, x);
                        color[0] = 0;
                        color[1] = 255;
                        color[2] = 0;
                    }
                }
            }
        }
    }
}