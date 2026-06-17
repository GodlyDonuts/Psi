cmake_minimum_required(VERSION 3.16)
project(psi LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

add_compile_options(-O2 -Wall -Wextra)

# Step 0 — scalar reverse-mode autograd + XOR MLP + finite-difference grad check.
add_executable(step0 src/step0_scalar_autograd/main.cpp)
