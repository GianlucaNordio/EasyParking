/*#include <iostream>
#include <opencv2/highgui.hpp>
#include <filesystem>

#include <fstream>


class ParkingSpot {
    int id;
    bool occupied;
    int center_x;
    int center_y;
    int width;
    int height;
    int angle;
};

int main() {
    std::cout << "Hello, World!" << std::endl << std::filesystem::current_path()  << std::endl;
    
    cv::Mat img = cv::imread("../src/test_image.jpg");
    cv::namedWindow("Test image");
    cv::imshow("Test image", img);
    cv::waitKey(0);

    // FROM WHAT I RED THIS WORKS ONLY WITH OPENCV FORMAT FILES
    //cv::FileStorage fs("test_bbox.xml", cv::FileStorage::READ);
    //fs.release();    // explicit close

    std::ifstream read("test_bbox.xml");
    std::vector<ParkingSpot> bboxes;
    
    for(std::string line; std::getline(read, line); ) {
        std::stringstream ss(line);
        std::cout << line << std::endl;
    }
    
    read.close();

    return 0;
}
*/

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>
#include <map>

struct Point {
    int x, y;
};

struct RotatedRect {
    Point center;
    int width, height;
    int angle;
};

struct Space {
    int id;
    bool occupied;
    RotatedRect rect;
    std::vector<Point> contour;
};

std::vector<Space> parseXML(const std::string& filename) {
    std::vector<Space> spaces;
    std::ifstream file(filename);
    std::string line;
    Space currentSpace;

    while (std::getline(file, line)) {
        std::istringstream iss(line);
        std::string token = line;

        //std::cout<<line<<std::endl;

        while (iss >> token) {
            std::cout << token<<std::endl;
            if (token.find("space") != std::string::npos) {
                if (token.find("id=") != std::string::npos) {
                    currentSpace.id = std::stoi(token.substr(token.find("\"") + 1, token.find_last_of("\"") - token.find("\"") - 1));
                }
                if (token.find("occupied=") != std::string::npos) {
                    currentSpace.occupied = std::stoi(token.substr(token.find("occupied=\"") + 10, 1));
                }
            } else if (token.find("center") != std::string::npos) {
                iss >> token;
                currentSpace.rect.center.x = std::stoi(token.substr(token.find("=") + 2));
                iss >> token;
                currentSpace.rect.center.y = std::stoi(token.substr(token.find("=") + 2));
            } else if (token.find("size") != std::string::npos) {
                iss >> token;
                currentSpace.rect.width = std::stoi(token.substr(token.find("=") + 2));
                iss >> token;
                currentSpace.rect.height = std::stoi(token.substr(token.find("=") + 2));
            } else if (token.find("angle") != std::string::npos) {
                iss >> token;
                currentSpace.rect.angle = std::stoi(token.substr(token.find("=") + 2));
            } else if (token.find("point") != std::string::npos) {
                Point point;
                iss >> token;
                point.x = std::stoi(token.substr(token.find("=") + 2));
                iss >> token;
                point.y = std::stoi(token.substr(token.find("=") + 2));
                currentSpace.contour.push_back(point);
            } else if (token.find("</space>") != std::string::npos) {
                spaces.push_back(currentSpace);
                currentSpace = Space();  // Reset for next space
            }
        }
    }

    return spaces;
}

int main() {
    std::vector<Space> spaces = parseXML("../src/test_bbox.xml");
    std::cout<< spaces.size();
    for (const auto& space : spaces) {
        std::cout << "Space ID: " << space.id << "\n";
        std::cout << "Occupied: " << space.occupied << "\n";
        std::cout << "Center: (" << space.rect.center.x << ", " << space.rect.center.y << ")\n";
        std::cout << "Size: " << space.rect.width << "x" << space.rect.height << "\n";
        std::cout << "Angle: " << space.rect.angle << "\n";
        std::cout << "Contour points: ";
        for (const auto& point : space.contour) {
            std::cout << "(" << point.x << ", " << point.y << ") ";
        }
        std::cout << "\n\n";
    }

    return 0;
}
