cmake_minimum_required(VERSION 3.15)

project(test)

set(CMAKE_CXX_COMPILER g++)
set(CMAKE_C_COMPILER gcc)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
add_compile_options(-O3)

set(LIBRARY_OUTPUT_PATH ${CMAKE_SOURCE_DIR})

find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(functions functions.cpp)
pybind11_add_module(structs structs.cpp)
pybind11_add_module(classes classes.cpp)
# identical to
#add_library(example MODULE example.cpp)  # if no MODULE, then outputs a static '.a' library
#target_link_libraries(example PRIVATE pybind11::module)
