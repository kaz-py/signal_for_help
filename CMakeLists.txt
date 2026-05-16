cmake_minimum_required(VERSION 3.16)
project(signal_for_help VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ----- OpenCV ---------------------------------------------------------------
find_package(OpenCV 4 REQUIRED COMPONENTS core imgproc highgui videoio dnn)
message(STATUS "OpenCV version: ${OpenCV_VERSION}")

# ----- Sources --------------------------------------------------------------
set(SOURCES
    src/main.cpp
    src/skin_detector.cpp
    src/hand_pose.cpp
    src/gesture_detector.cpp
    src/alert_system.cpp
)

add_executable(signal_for_help ${SOURCES})

target_include_directories(signal_for_help PRIVATE include)
target_link_libraries(signal_for_help PRIVATE ${OpenCV_LIBS})

# Compiler warnings
if(MSVC)
    target_compile_options(signal_for_help PRIVATE /W4)
else()
    target_compile_options(signal_for_help PRIVATE -Wall -Wextra -O2)
endif()

# ----- Install models symlink -----------------------------------------------
# Copy the models/ directory next to the executable so the default path works
add_custom_command(TARGET signal_for_help POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E create_symlink
        "${CMAKE_SOURCE_DIR}/models"
        "$<TARGET_FILE_DIR:signal_for_help>/models"
    COMMENT "Linking models/ directory to build output"
)
