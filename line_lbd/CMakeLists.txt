cmake_minimum_required(VERSION 2.8.3)
project(line_lbd)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
set(OpenCV_DIR /Users/Brian/Documents/build_opencv_4.3.0/lib/cmake/opencv4/)

find_package(OpenCV REQUIRED)


###########
## Build ##
###########
include_directories(${OpenCV_INCLUDE_DIRS})

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(include)

#set(CMAKE_CXX_STANDARD 11)
#set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS
   "${CMAKE_CXX_FLAGS} ${SSE_FLAGS} -march=native  -std=c++0x"
) 

## Declare a C++ library

add_library(line_lbd_lib SHARED
  libs/binary_descriptor.cpp libs/binary_descriptor_matcher.cpp libs/LSDDetector.cpp  libs/lsd.cpp libs/bitarray.hpp  
  libs/bitops.hpp libs/draw.cpp libs/precomp.hpp libs/types.hpp  class/line_lbd_allclass.cpp
)
target_link_libraries(line_lbd_lib ${OpenCV_LIBRARIES}) 
## Add cmake target dependencies of the library
## as an example, code may need to be generated before libraries
## either from message generation or dynamic reconfigure
# add_dependencies(line_lbd ${${PROJECT_NAME}_EXPORTED_TARGETS})

## Declare a C++ executable

# add_executable(linelbd_detect_node src/detect_lines.cpp)
# target_link_libraries(linelbd_detect_node  line_lbd_lib ${OpenCV_LIBRARIES})


#add_executable(linelbd_descrip_node src/compute_descriptors.cpp)
#target_link_libraries(linelbd_descrip_node  line_lbd_lib ${OpenCV_LIBRARIES})


## Add cmake target dependencies of the executable
## same as for the library above
# add_dependencies(line_lbd_node ${${PROJECT_NAME}_EXPORTED_TARGETS})

## Specify libraries to link a library or executable target against
# target_link_libraries(line_lbd_node
# )

