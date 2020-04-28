#include <iostream>
#include <say-hello/hello.hpp>
#include <mainCamera.hpp>

int main() {
	std::cout << "main Program!\n";
	hello::say_hello();
	draw::circles();

}
