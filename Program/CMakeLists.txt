

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

add_executable(mainprogram main.cpp )

target_link_libraries(mainprogram ${OpenCV_LIBS} )

target_link_libraries(mainprogram CameraLib)
target_link_libraries(mainprogram Mylib)