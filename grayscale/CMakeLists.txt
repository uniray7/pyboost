cmake_minimum_required(VERSION 2.8 FATAL_ERROR)
project("pbcvt")

FIND_PACKAGE(PythonInterp)
FIND_PACKAGE(PythonLibs)
if(APPLE AND ${PYTHON_VERSION_MAJOR} EQUAL 3)
    FIND_PACKAGE(Boost COMPONENTS python3)
else()
    FIND_PACKAGE(Boost COMPONENTS python)
endif()

SET(REQUIRED_INCLUDES ${Boost_INCLUDE_DIRS} ${PYTHON_INCLUDE_DIRS})
SET(REQUIRED_LIBRARIES ${Boost_LIBRARIES} ${PYTHON_LIBRARIES})

set(PROJ_OCV_DEP_MODULES core imgcodecs imgproc)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
find_package( "OpenCV" 3.1 REQUIRED ${PROJ_OCV_DEP_MODULES}
    PATHS /usr/local/opt/opencv3 # for homebrew keg-only (not linked)
)


file(GLOB project_sources ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)

PYTHON_ADD_MODULE(${PROJECT_NAME} SHARED ${project_sources} ${CMAKE_CURRENT_SOURCE_DIR}/include/pyboostcvconverter.hpp ${CMAKE_CURRENT_SOURCE_DIR}/include/grayscale.hpp)


target_include_directories(${PROJECT_NAME} PUBLIC
        ${REQUIRED_INCLUDES}
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
        ${OpenCV_INCLUDE_DIRS}
        )

target_link_libraries(${PROJECT_NAME}
        ${REQUIRED_LIBRARIES}
        ${OpenCV_LIBS}
        )
FILE(COPY test.jpg DESTINATION .)
FILE(COPY gray.py DESTINATION .)

