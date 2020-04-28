add_library(
	Mylib
	src/say-hello/hello.hpp
	src/say-hello/hello.cpp
)

target_include_directories(Mylib PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/src")