#include <iostream>
#include <opencv2/highgui.hpp>
#include <fstream>


class ParkingSpot {
    public:
        int id;
        bool occupied;
        int center_x;
        int center_y;
        int width;
        int height;
        int angle;
};

std::vector<ParkingSpot> parseXML(const std::string& filename) {
    std::vector<ParkingSpot> spaces;
    std::ifstream file(filename);
    std::string line;
    ParkingSpot currentSpace;

    std::getline(file, line); // to skip the first line containing parking (the first id would cause problems)
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        std::string token;
        while(iss >> token) {
            std::cout<<token<<std::endl;
            if(token.find("<space") != std::string::npos) {
                currentSpace = ParkingSpot();
            }
            else if(token.find("</space>") != std::string::npos) {
                spaces.push_back(currentSpace);
            }
            else if(token.find("id=") != std::string::npos) {
                currentSpace.id = std::stoi(token.substr(token.find("\"") + 1, token.find_last_of("\"") - token.find("\"") - 1));
            }
            else if (token.find("occupied=") != std::string::npos) {
                currentSpace.occupied = std::stoi(token.substr(token.find("occupied=\"") + 10, 1));
            } else if (token.find("center") != std::string::npos) {
                iss >> token;
                currentSpace.center_x = std::stoi(token.substr(token.find("=") + 2));
                iss >> token;
                currentSpace.center_y = std::stoi(token.substr(token.find("=") + 2));
            } else if (token.find("size") != std::string::npos) {
                iss >> token;
                currentSpace.width = std::stoi(token.substr(token.find("=") + 2));
                iss >> token;
                currentSpace.height = std::stoi(token.substr(token.find("=") + 2));
            } else if (token.find("angle") != std::string::npos) {
                iss >> token;
                currentSpace.angle = std::stoi(token.substr(token.find("=") + 2));
            }
        }
        
        /*
        else if (token.find("point") != std::string::npos) {
                Point point;
                iss >> token;
                point.x = std::stoi(token.substr(token.find("=") + 2));
                iss >> token;
                point.y = std::stoi(token.substr(token.find("=") + 2));
                currentSpace.contour.push_back(point);
            }
        */
        
    }
    return spaces;
}

int main() {
    std::vector<ParkingSpot> spaces = parseXML("../src/test_bbox.xml");
    std::cout<< spaces.size();
    for (const auto& space : spaces) {
        std::cout << "Space ID: " << space.id << "\n";
        std::cout << "Occupied: " << space.occupied << "\n";
        std::cout << "Center: (" << space.center_x << ", " << space.center_y << ")\n";
        std::cout << "Size: " << space.width << "x" << space.height << "\n";
        std::cout << "Angle: " << space.angle << "\n";
        //std::cout << "Contour points: ";
        //for (const auto& point : space.contour) {
        //    std::cout << "(" << point.x << ", " << point.y << ") ";
        //}
        
        std::cout << "\n\n";
    }

    return 0;
}
