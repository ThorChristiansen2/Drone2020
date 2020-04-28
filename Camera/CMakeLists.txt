
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )
add_executable(CameraFile main.cpp )
target_link_libraries(CameraFile ${OpenCV_LIBS} )

target_link_libraries(CameraFile Mylib)