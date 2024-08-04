cmake_minimum_required(VERSION 2.8)
#set(CMAKE_CXX_STANDARD 11)
#Set the project name
project(EasyParking)

# Looks for OpenCVConfig.cmake (where some OpenCV related vars are defined)
find_package(OpenCV REQUIRED)

# Tells the compiler to look for the headers also in the folders defined in this variable
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(./include)

#Add the executable
add_executable(${PROJECT_NAME} src/main.cpp)
target_link_libraries(${PROJECT_NAME}
    ${OpenCV_LIBS}
)
