#include "utils.hpp"

/**
 * Creates a single image by concatenating a sequence of input images in a grid format.
 * The number of images per row is specified by the `imagesPerLine` parameter.
 * The function checks that all images have the same size and adjusts the number of images per row if necessary.
 *
 * @param images         A vector of cv::Mat objects representing the input images to be concatenated.
 * @param imagesPerLine The number of images to place per row in the output image.
 * @return              A cv::Mat object containing the concatenated image.
 * @throws std::invalid_argument if no images are provided or if images have different sizes.
 */
cv::Mat produceSingleImage(const std::vector<cv::Mat>& images, int imagesPerLine) {

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

    cv::resize(result, result, cv::Size(result.cols/2, result.rows/2));

    return result;
}
/**
 * Loads frames from a base sequence dataset directory into a vector of cv::Mat objects.
 * The frames are loaded from a predefined folder path that includes the base sequence index.
 *
 * @param datasetPath The path to the dataset directory.
 * @param images      A reference to a vector of cv::Mat objects where the loaded frames will be stored.
 */
void loadBaseSequenceFrames(const std::string& datasetPath, std::vector<cv::Mat> &images) {
    std::string folderPath = datasetPath + SLASH + SEQUENCE + std::to_string(BASE_SEQUENCE_INDEX) + SLASH + FRAMES_FOLDER;
    loadImages(folderPath, images);
}

/**
 * Loads frames from multiple sequences of a dataset into a vector of vectors of cv::Mat objects.
 * Frames are loaded from directories corresponding to each sequence.
 *
 * @param datasetPath The path to the dataset directory.
 * @param numSequences The number of sequences to load.
 * @param images      A reference to a vector of vectors of cv::Mat objects where the loaded frames will be stored.
 */
void loadSequencesFrames(const std::string& datasetPath, int numSequences, std::vector<std::vector<cv::Mat>> &images) {
    for(int i = 1; i <= numSequences; i++) {
        std::vector<cv::Mat> empty;   
        images.push_back(empty);
        std::string folderPath = datasetPath + SLASH + SEQUENCE + std::to_string(i) + SLASH + FRAMES_FOLDER;
        loadImages(folderPath, images[i - 1]);
    }
}

/**
 * Loads images from a specified directory into a vector of cv::Mat objects.
 * Only regular files are processed, and empty or unreadable images are reported.
 *
 * @param path   The path to the directory containing images.
 * @param images A reference to a vector of cv::Mat objects where the loaded images will be stored.
 * @throws std::invalid_argument if the provided path is not a directory.
 * @throws std::runtime_error if an image file is empty or cannot be read.
 */
void loadImages(const std::string path, std::vector<cv::Mat> &images) {
    namespace fs = std::filesystem;

    if (!fs::is_directory(path)) {
        throw std::invalid_argument("The provided path is not a directory.");
    }

    for(const auto& entry : fs::directory_iterator(path)) {
        if(entry.is_regular_file()){
            try {
                std::string filePath = entry.path().string();
                cv::Mat image = cv::imread(filePath);

                if (image.empty()) {
                    throw std::runtime_error("Image file is empty or cannot be read.");
                }

                images.push_back(image);
                std::cout << "Read image: " << filePath << std::endl;
            } 
            catch (const std::exception& e) {
                std::cerr << "Error processing file " << entry.path().string() << ": " << e.what() << std::endl;
            }
        }
    }
}

/**
 * Loads segmentation masks for multiple sequences of a dataset into a vector of vectors of cv::Mat objects.
 * Each mask image is converted to grayscale.
 *
 * @param datasetPath The path to the dataset directory.
 * @param numSequences The number of sequences to load.
 * @param segMasks    A reference to a vector of vectors of cv::Mat objects where the loaded masks will be stored.
 */
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
            maskRightTopCorner(segMasks[i][j]);
        }
    }
}

/**
 * Loads ground truth data for the base sequence from the specified dataset path.
 *
 * This function retrieves all ground truth files for the base sequence (indexed by `BASE_SEQUENCE_INDEX`)
 * from a designated folder within the dataset path. Each file contains parking spot annotations, which are
 * parsed into `ParkingSpot` objects. The parsed data for each file is appended to the `groundTruth` vector.
 * 
 * The ground truth data is organized as a vector of vectors, where each inner vector contains the parking
 * spots for a frame.
 * 
 * @param datasetPath The base path to the dataset containing the base sequence folder.
 * @param groundTruth A vector of vectors of `ParkingSpot` objects where each inner vector contains the parking
 *                    spots for a frame in the base sequence.
 */
void loadBaseSequenceGroundTruth(const std::string& datasetPath, std::vector<std::vector<ParkingSpot>> &groundTruth){
    std::string folderPath = datasetPath + SLASH + SEQUENCE + std::to_string(BASE_SEQUENCE_INDEX) + SLASH + BOUNDING_BOX_FOLDER;
    std::vector<std::string> files;
        for (const auto& entry : std::filesystem::directory_iterator(folderPath)) {
            if (std::filesystem::is_regular_file(entry)) {
                files.push_back(entry.path().filename().string());
            }
        }
        
        for (int j = 0; j < files.size(); j++) {
            std::string filePath = folderPath + SLASH + files[j];
            
            std::vector<ParkingSpot> spots;
            loadGroundTruth(filePath, spots);

            groundTruth.push_back(spots);
        }
}

/**
 * Loads ground truth data for a series of sequences from the specified dataset path.
 *
 * This function iterates over a number of sequences, each stored in its own folder within the dataset path.
 * For each sequence, it collects all ground truth files (containing parking spot annotations) and parses them
 * into `ParkingSpot` objects. The parsed data is organized into a nested vector structure:
 * `groundTruth[sequenceIndex][frameIndex]` contains the parking spots for a particular frame in the sequence.
 *
 * @param datasetPath The base path to the dataset containing sequence folders.
 * @param numSequences The number of sequences to process.
 * @param groundTruth A nested vector of `ParkingSpot` objects where each entry represents the ground truth
 *                    for a specific frame within a specific sequence.
 */
void loadSequencesGroundTruth(const std::string& datasetPath, int numSequences, std::vector<std::vector<std::vector<ParkingSpot>>> &groundTruth) {
    for (int i = 0; i < numSequences; i++) {
        
        std::vector<std::vector<ParkingSpot>> sequence;
        groundTruth.push_back(sequence);

        std::string folderPath = datasetPath + SLASH + SEQUENCE + std::to_string(i + 1) + SLASH + BOUNDING_BOX_FOLDER;
        
        std::vector<std::string> files;
        for (const auto& entry : std::filesystem::directory_iterator(folderPath)) {
            if (std::filesystem::is_regular_file(entry)) {
                files.push_back(entry.path().filename().string());
            }
        }
        
        for (int j = 0; j < files.size(); j++) {
            std::string filePath = folderPath + SLASH + files[j];
            
            std::vector<ParkingSpot> spots;
            loadGroundTruth(filePath, spots);

            groundTruth[i].push_back(spots);
        }
    }
}

/**
 * Loads ground truth data from an XML file into a vector of ParkingSpot objects.
 *
 * @param path        The path to the XML file containing the ground truth data.
 * @param groundTruth A reference to a vector of ParkingSpot objects where the loaded ground truth data will be stored.
 */
void loadGroundTruth(std::string path, std::vector<ParkingSpot> &groundTruth) {
    parseXML(path, groundTruth);
}

/**
 * Retrieves the filename at the specified index from a folder.
 *
 * This function collects all regular files in the specified directory and returns the filename
 * of the file at the given index. If the index is out of bounds, an empty string is returned.
 *
 * @param folderPath The path to the folder containing the files.
 * @param index The index of the file to retrieve.
 * @return The filename of the file at the specified index, or an empty string if the index is invalid.
 */
std::string getFileInFolder(const std::string& folderPath, int index) {
    std::vector<std::string> files;
    
    for (const auto& entry : std::filesystem::directory_iterator(folderPath)) {
        if (std::filesystem::is_regular_file(entry)) {
            files.push_back(entry.path().filename().string());
        }
    }
    
    if (index >= 0 && index < files.size()) {
        return files[index];
    } else {
        return ""; 
    }
}

/**
 * Converts a vector of grayscale images to a vector of BGR images.
 * The grayscale values are mapped to specific BGR colors.
 *
 * @param greyImages A vector of vectors of grayscale images to be converted.
 * @param BGRImages  A reference to a vector of vectors of BGR images where the converted images will be stored.
 */
void convertGreyMasksToBGR(const std::vector<std::vector<cv::Mat>> &greyImages, std::vector<std::vector<cv::Mat>> &BGRImages) {
    for(int i = 0; i < greyImages.size(); i++) {
        std::vector<cv::Mat> empty;
        BGRImages.push_back(empty);
        convertGreyMaskToBGR(greyImages[i], BGRImages[i]);
    }
}

/**
 * Converts a vector of grayscale images to BGR images.
 * The grayscale values are mapped to specific BGR colors.
 *
 * @param greyImage A vector of grayscale images to be converted.
 * @param BGRImage  A reference to a vector of BGR images where the converted images will be stored.
 */
void convertGreyMaskToBGR(const std::vector<cv::Mat> &greyImage, std::vector<cv::Mat> &BGRImage) {
    for(int i = 0; i < greyImage.size(); i++) {
        BGRImage.push_back(cv::Mat(greyImage[i].rows,greyImage[i].cols, CV_8UC3, cv::Scalar(0 ,0 ,0)));
        cv::Mat &dst = BGRImage[i];
        const cv::Mat &src = greyImage[i];
        for(int x = 0; x < src.cols; x++) {
            for(int y = 0; y < src.rows; y++) {
                cv::Vec3b &color = dst.at<cv::Vec3b>(y, x);
                if (src.at<uchar>(y, x) == 0) {
                    color[0] = 128;
                    color[1] = 128;
                    color[2] = 128;
                }
                else if (src.at<uchar>(y, x) == 1) {
                    color[0] = 0;
                    color[1] = 0;
                    color[2] = 255;
                }
                else if (src.at<uchar>(y, x) == 2) {
                    color[0] = 0;
                    color[1] = 255;
                    color[2] = 0;
                }
            }
        }
    }
}

/**
 * Prints performance metrics including mean Average Precision (mAP) and Intersection over Union (IoU) for each frame.
 * The metrics are formatted in a table with fixed precision.
 *
 * @param mAPs A vector of mean Average Precision values for each frame.
 * @param IoUs A vector of Intersection over Union values for each frame.
 */
void printPerformanceMetrics(const std::vector<double>& mAPs, const std::vector<double>& IoUs) {
    std::cout << std::left << std::setw(12) << "Frame" << std::setw(12) << "mAP" << std::setw(12) << "IoU" << std::endl;
    std::cout << std::string(36, '-') << std::endl;
    
    for (int i = 0; i < mAPs.size(); i++) {
        std::cout << std::left << std::setw(12) << i + 1 << std::setw(12) << std::fixed << std::setprecision(4) << mAPs[i]
                  << std::setw(12) << std::fixed << std::setprecision(4) << IoUs[i] << std::endl;
    }
}

/**
 * Draws bounding boxes for parking spots on the given sequence of images.
 *
 * For each image in the base sequence, this function creates a copy of the image and draws 
 * bounding boxes around each parking spot using the rectangle coordinates from the `ParkingSpot` objects.
 * The bounding boxes are drawn in red with a thickness of 2 pixels.
 * The modified images with bounding boxes are then stored in the `baseSequenceBBoxes` vector.
 *
 * @param parkingSpot A vector of vector of `ParkingSpot` objects representing the parking spots to be drawn.
 * @param baseSequence A vector of `cv::Mat` objects where each `cv::Mat` represents an image in the base sequence.
 * @param baseSequenceBBoxes A vector of `cv::Mat` objects where each `cv::Mat` will store the image with drawn bounding boxes.
 */
void printParkingSpot(const std::vector<std::vector<ParkingSpot>>& parkingSpot, const std::vector<cv::Mat>& baseSequence, std::vector<cv::Mat>& baseSequenceBBoxes) {
    for(int i = 0; i < baseSequence.size(); i++) {
        cv::Mat output = baseSequence[i].clone();
        for(int j = 0; j < parkingSpot[i].size(); j++) {
            cv::Point2f vertices[4];
            parkingSpot[i][j].rect.points(vertices);
            if(parkingSpot[i][j].occupied) {
                for (int l = 0; l < 4; l++) {
                    cv::line(output, vertices[l], vertices[(l + 1) % 4], cv::Scalar(0, 0, 255), 2);
                }
            }else{
                for (int l = 0; l < 4; l++) {
                    cv::line(output, vertices[l], vertices[(l + 1) % 4], cv::Scalar(255, 0, 0), 2);
                }
            }

        }
        baseSequenceBBoxes.push_back(output);
    }
}

/**
 * @brief Masks the region above a specified line in the image by setting pixel values to 0.
 * 
 * This function modifies the input image by setting all pixel values to 0 for those located above
 * a line defined by two points: (850, 0) and (width, 230). The line equation is used to determine the
 * boundary, and all pixels with y-coordinates less than the line's y-value at the corresponding x-coordinate
 * are set to 0.
 * 
 * @param img The input image (cv::Mat) to be modified. This image is assumed to be of type CV_8UC1 
 *            (8-bit single-channel), typically used for grayscale images.
 * 
 * @note The function assumes that the input image is in grayscale format. For color images or other
 *       image types, modifications to the function may be necessary.
 * 
 * @warning This function performs pixel-wise operations and may be slow for large images. Optimization
 *          may be required for performance-critical applications.
 */
void maskRightTopCorner(cv::Mat& img) {

    int width = img.cols;
    int height = img.rows;

    const int x1 = 850;
    const int y1 = 0;
    const int x2 = width;
    const int y2 = 230;

    cv::Point pt1(x1, y1);
    cv::Point pt2(x2, y2);

    double m = static_cast<double>(pt2.y - pt1.y) / (pt2.x - pt1.x);
    double q = pt1.y - m * pt1.x;

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            double lineY = m * x + q;

            if (y < lineY) {
                img.at<uchar>(y, x) = 0;
            }
        }
    }
}