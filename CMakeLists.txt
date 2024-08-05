cmake_minimum_required(VERSION 3.10)
project(EasyParking)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Trova OpenCV
find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(include)

file(GLOB_RECURSE SOURCES
    src/*.cpp
    include/parkingSpot/*.cpp
    include/parkingSpotDetector/*.cpp
    include/parser/*.cpp
)

add_executable(EasyParking ${SOURCES})

target_include_directories(EasyParking PRIVATE include)
target_link_libraries(EasyParking ${OpenCV_LIBS})
