cmake_minimum_required(VERSION 3.5.0)
project(cpp_onnx_proj VERSION 0.1.0)

include(CTest)
enable_testing()


set(OpenCV_DIR "C:/opencv/build/")
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

add_executable(cpp_onnx_proj src/main.cpp
    src/printMessages/printMessages.cpp
    src/printMessages/printMessages.h
    src/opencv_scripts/onnx_model/onnx_model.cpp
    src/opencv_scripts/onnx_model/onnx_model.h
    src/opencv_scripts/dispImage/dispImage.cpp
    src/opencv_scripts/dispImage/dispImage.h
    src/data_structures/vector_indexed_probabilities/vector_indexed_probabilities.h
    src/data_structures/vector_indexed_probabilities/vector_indexed_probabilities.cpp
    src/data_structures/vector_indexed_probabilities/struct_indexed_probability/struct_indexed_probability.h
    src/data_structures/vector_indexed_probabilities/struct_indexed_probability/struct_indexed_probability.cpp)

target_link_libraries( cpp_onnx_proj ${OpenCV_LIBS} )

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)