cmake_minimum_required(VERSION 3.15)
project(yolo_tasks_cpp)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ------------------------------------------------------------------
#                           Options
# ------------------------------------------------------------------

option(USE_CUDA "Enable CUDA support" ON)

# ------------------------------------------------------------------
#                           OpenCV
# ------------------------------------------------------------------

set(OpenCV_DIR "C:/opencv/build/x64/vc16/lib")
find_package(OpenCV REQUIRED)
message(STATUS "OpenCV: ${OpenCV_VERSION}")

# ------------------------------------------------------------------
#                        ONNX Runtime
# ------------------------------------------------------------------

set(ONNXRUNTIME_DIR "C:/onnxruntime")
set(ONNXRUNTIME_INCLUDE "${ONNXRUNTIME_DIR}/include")
set(ONNXRUNTIME_LIB "${ONNXRUNTIME_DIR}/lib")

if(NOT EXISTS "${ONNXRUNTIME_INCLUDE}/onnxruntime_cxx_api.h")
    message(FATAL_ERROR "ONNX Runtime not found at ${ONNXRUNTIME_DIR}")
endif()
message(STATUS "ONNX Runtime: ${ONNXRUNTIME_DIR}")

# Check CUDA support
set(CUDA_AVAILABLE FALSE)
if(USE_CUDA AND EXISTS "${ONNXRUNTIME_LIB}/onnxruntime_providers_cuda.lib")
    set(CUDA_AVAILABLE TRUE)
    message(STATUS "CUDA: Enabled")
else()
    message(STATUS "CUDA: Disabled")
endif()

# ------------------------------------------------------------------
#                       Helper Function
# ------------------------------------------------------------------

function(add_yolo_target TARGET_NAME SOURCE_FILE)

    add_executable(${TARGET_NAME}
        src/${SOURCE_FILE}
        src/utils.cpp
    )

    target_include_directories(${TARGET_NAME} PRIVATE
        ${OpenCV_INCLUDE_DIRS}
        ${ONNXRUNTIME_INCLUDE}
    )

    target_link_libraries(${TARGET_NAME} PRIVATE
        ${OpenCV_LIBS}
        "${ONNXRUNTIME_LIB}/onnxruntime.lib"
    )

    # Link CUDA libraries if available
    if(CUDA_AVAILABLE)
        target_link_libraries(${TARGET_NAME} PRIVATE
            "${ONNXRUNTIME_LIB}/onnxruntime_providers_shared.lib"
            "${ONNXRUNTIME_LIB}/onnxruntime_providers_cuda.lib"
        )
    endif()

    # Copy DLLs after build
    add_custom_command(TARGET ${TARGET_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${ONNXRUNTIME_LIB}/onnxruntime.dll"
            "$<TARGET_FILE_DIR:${TARGET_NAME}>"
    )

    if(EXISTS "${ONNXRUNTIME_LIB}/onnxruntime_providers_shared.dll")
        add_custom_command(TARGET ${TARGET_NAME} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${ONNXRUNTIME_LIB}/onnxruntime_providers_shared.dll"
                "$<TARGET_FILE_DIR:${TARGET_NAME}>"
        )
    endif()

    if(CUDA_AVAILABLE)
        add_custom_command(TARGET ${TARGET_NAME} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${ONNXRUNTIME_LIB}/onnxruntime_providers_cuda.dll"
                "$<TARGET_FILE_DIR:${TARGET_NAME}>"
        )
    endif()

    # Copy models folder
    add_custom_command(TARGET ${TARGET_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
            "${CMAKE_SOURCE_DIR}/models"
            "$<TARGET_FILE_DIR:${TARGET_NAME}>/models"
    )

    message(STATUS "Target: ${TARGET_NAME}")

endfunction()

# ------------------------------------------------------------------
#                          Targets
# ------------------------------------------------------------------

add_yolo_target(task1_detector "detector.cpp")
add_yolo_target(task2_segmenter "segmenter.cpp")

# ------------------------------------------------------------------
#                          Summary
# ------------------------------------------------------------------

message(STATUS "")
message(STATUS "Build Configuration:")
message(STATUS "  C++ Standard : C++${CMAKE_CXX_STANDARD}")
message(STATUS "  OpenCV       : ${OpenCV_VERSION}")
message(STATUS "  ONNX Runtime : ${ONNXRUNTIME_DIR}")
message(STATUS "  CUDA         : ${CUDA_AVAILABLE}")
message(STATUS "  Targets      : task1_detector, task2_segmenter")
message(STATUS "")