find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

add_library(
	CameraLib
	mainCamera.hpp
	mainCamera.cpp
)

target_include_directories(CameraLib PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")

# find_package( OpenCV REQUIRED )
# include_directories( ${OpenCV_INCLUDE_DIRS} )
# add_executable(CameraFile main.cpp)
# target_link_libraries(CameraFile ${OpenCV_LIBS} )

# target_link_libraries(CameraFile Mylib)