cmake_minimum_required(VERSION 3.10)
project(MyProject)

set(CMAKE_CXX_STANDARD 11)

# OpenCV paketini bul
find_package(OpenCV REQUIRED)

# Eigen paketini bul
find_package(Eigen3 REQUIRED)

find_package(OpenMP REQUIRED)

if(OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

# OpenCV başlık dosyalarını dahil edin
include_directories(${OpenCV_INCLUDE_DIRS})

# Eigen başlık dosyalarını dahil edin
include_directories(${EIGEN3_INCLUDE_DIR})

# Kaynak dosyalarını ekleyin
add_executable(MyProject src/moving_median.cpp)

# OpenCV kütüphanelerini bağlayın
target_link_libraries(MyProject ${OpenCV_LIBS} ${OpenMP_CXX_LIBRARIES})