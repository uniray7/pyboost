cmake_minimum_required(VERSION 2.8 FATAL_ERROR)
project("pbcvt")

##----------------------------CMAKE & GLOBAL PROPERTIES-------------------------#
#list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
#
####============= C++11 support====================================
#include(CheckCXXCompilerFlag)
#CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
#CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
#if (COMPILER_SUPPORTS_CXX11)
#    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
#elseif (COMPILER_SUPPORTS_CXX0X)
#    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
#else ()
#    message(FATAL_ERROR "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
#endif ()
##=================================================================
## PYTHON option
#
#set(PYTHON_OPTIONS "2.X" "3.X")
#set(PYTHON_DESIRED_VERSION "3.X" CACHE STRING "Choose which python version to use, options are: ${PYTHON_OPTIONS}.")
#set_property(CACHE PYTHON_DESIRED_VERSION PROPERTY STRINGS ${PYTHON_OPTIONS})
#
#
##=============== Find Packages 
#find_package(OpenCV COMPONENTS core REQUIRED)
#include("DetectPython")
#if (${PYTHON_DESIRED_VERSION} STREQUAL "2.X")
#    set(Python_ADDITIONAL_VERSIONS ${PYTHON2_VERSION_MAJOR}.${PYTHON2_VERSION_MINOR})
#    find_package(Boost COMPONENTS python-py${PYTHON2_VERSION_MAJOR}${PYTHON2_VERSION_MINOR} REQUIRED)
#else ()
#    set(Python_ADDITIONAL_VERSIONS ${PYTHON3_VERSION_MAJOR}.${PYTHON3_VERSION_MINOR})
#    find_package(Boost COMPONENTS python-py${PYTHON3_VERSION_MAJOR}${PYTHON3_VERSION_MINOR} REQUIRED)
#endif ()


##========pick python stuff========================================
#if (${PYTHON_DESIRED_VERSION} STREQUAL "2.X")
#    SET(PYTHON_INCLUDE_DIRS ${PYTHON2_INCLUDE_DIR} ${PYTHON2_INCLUDE_DIR2} ${PYTHON2_NUMPY_INCLUDE_DIRS})
#    SET(PYTHON_LIBRARIES ${PYTHON2_LIBRARY})
#    SET(PYTHON_EXECUTABLE ${PYTHON2_EXECUTABLE})
#    SET(PYTHON_PACKAGES_PATH ${PYTHON2_PACKAGES_PATH})
#    SET(ARCHIVE_OUTPUT_NAME pbcvt_py2)
#else ()
#    SET(PYTHON_INCLUDE_DIRS ${PYTHON3_INCLUDE_DIR} ${PYTHON3_INCLUDE_DIR2} ${PYTHON3_NUMPY_INCLUDE_DIRS})
#    SET(PYTHON_LIBRARIES ${PYTHON3_LIBRARY})
#    SET(PYTHON_EXECUTABLE ${PYTHON3_EXECUTABLE})
#    SET(PYTHON_PACKAGES_PATH ${PYTHON3_PACKAGES_PATH})
#    SET(ARCHIVE_OUTPUT_NAME pbcvt_py3)
#endif ()


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

PYTHON_ADD_MODULE(${PROJECT_NAME} SHARED ${project_sources} ${CMAKE_CURRENT_SOURCE_DIR}/include/pyboostcvconverter/pyboostcvconverter.hpp)


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

##---------------------------   INSTALLATION    -----------------------------------------------------
##-get proper extension for python binary shared object on this platform
#execute_process(COMMAND ${PYTHON_EXECUTABLE} -c "import distutils.sysconfig; print(distutils.sysconfig.get_config_var('SO'))"
#        RESULT_VARIABLE PYTHON_${PROJECT_NAME}_PY_PROCESS
#        OUTPUT_VARIABLE ${PROJECT_NAME}_PY_SUFFIX
#        OUTPUT_STRIP_TRAILING_WHITESPACE)
#
#set_target_properties(${PROJECT_NAME} PROPERTIES
#        ARCHIVE_OUTPUT_NAME ${ARCHIVE_OUTPUT_NAME}  # prevent name conflict for python2/3 outputs
#        PREFIX ""
#        OUTPUT_NAME pbcvt
#        SUFFIX ${${PROJECT_NAME}_PY_SUFFIX})
#
#
#install(TARGETS ${PROJECT_NAME}
#        ${PYTHON_INSTALL_CONFIGURATIONS}
#        RUNTIME DESTINATION ${PYTHON_PACKAGES_PATH} COMPONENT python
#        LIBRARY DESTINATION ${PYTHON_PACKAGES_PATH} COMPONENT python
#        ${PYTHON_INSTALL_ARCHIVE}
#        )
#
