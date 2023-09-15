cmake_minimum_required(VERSION 3.12)

project(PeopleCounter)
set(CMAKE_BUILD_TYPE Debug)
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )
add_executable( PeopleCounter main.cpp peopleTracker/MultipleObjectTracker.cpp peopleTracker/ObjectTracker.cpp peopleDetector/PeopleDetector.cpp common/Constants.cpp common/Constants.hpp)
include_directories(common)
target_link_libraries( PeopleCounter ${OpenCV_LIBS} )