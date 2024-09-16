#include "parser.hpp"

/**
 * Parses an XML file to extract information about parking spots and stores them in the provided vector.
 * Each parking spot is represented by its ID, occupancy status, and bounding box (a rotated rectangle).
 *
 * The XML file is expected to have a format with `<space>` tags that contain attributes like `id`, `occupied`,
 * `center` (with x and y coordinates), `size` (with width and height), and `angle` (rotation angle).
 *
 * @param filePath        The path to the XML file to be parsed.
 * @param parkingSpot     A reference to a vector of ParkingSpot objects where the parsed data will be stored.
 */
void parseXML(const std::string& filePath, std::vector<ParkingSpot> &parkingSpot) {
    std::ifstream file(filePath);
    std::string line;
    ParkingSpot currentSpace;

    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filePath << std::endl;
        return;
    }

    // Skip the first line containing parking (the first id would cause problems)
    std::getline(file, line); 

    while (std::getline(file, line)) {
        std::istringstream iss(line);
        std::string token;
        while(iss >> token) {
            if(token.find("<space") != std::string::npos) {
                currentSpace = ParkingSpot();
            } else if(token.find("</space>") != std::string::npos) {
                parkingSpot.push_back(currentSpace);
            } else if(token.find("id=") != std::string::npos) {
                currentSpace.id = std::stoi(token.substr(token.find("\"") + 1, token.find_last_of("\"") - token.find("\"") - 1));
            } else if (token.find("occupied=") != std::string::npos) {
                currentSpace.occupied = std::stoi(token.substr(token.find("occupied=\"") + 10, 1));
            } else if (token.find("center") != std::string::npos) {
                // Read the center position
                iss >> token;
                int center_x = std::stoi(token.substr(token.find("=") + 2));
                iss >> token;
                int center_y = std::stoi(token.substr(token.find("=") + 2));
                //Add center position
                currentSpace.rect.center = cv::Point2f(center_x, center_y);
            } else if (token.find("size") != std::string::npos) {
                //Read the rectangle size
                iss >> token;
                int width = std::stoi(token.substr(token.find("=") + 2));
                iss >> token;
                int height = std::stoi(token.substr(token.find("=") + 2));
                // Add rectangle size
                currentSpace.rect.size = cv::Size2f(width, height);
            } else if (token.find("angle") != std::string::npos) {
                iss >> token;
                currentSpace.rect.angle = std::stoi(token.substr(token.find("=") + 2));
            }
        }        
    }

    // Pop back the last three element as they are the spots on the top right
    parkingSpot.pop_back();
    parkingSpot.pop_back();
    parkingSpot.pop_back();
                
}