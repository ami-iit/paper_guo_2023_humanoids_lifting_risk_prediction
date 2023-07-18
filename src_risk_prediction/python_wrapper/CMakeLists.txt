cmake_minimum_required(VERSION 3.12)

project(hdeModulePythonWrapper LANGUAGES C CXX
                             VERSION 0.0.1)

set(CMAKE_CXX_STANDARD 17)

# Find YARP for bindings-only builds
find_package(YARP COMPONENTS conf os sig dev math gsl REQUIRED)
foreach(_component conf os sig dev math gsl)
  get_property(YARP_${_component}_INCLUDE_DIRS TARGET YARP::YARP_${_component} PROPERTY INTERFACE_INCLUDE_DIRECTORIES)
  include_directories(${YARP_${_component}_INCLUDE_DIRS})
endforeach()

find_package(HumanDynamicsEstimation REQUIRED)

#include_directories(${CMAKE_SOURCE_DIR}/viewIHumanDatas)

# for yarp.i
include_directories(${YARP_BINDINGS})

option(BUILD_SHARED_LIBS "build shared libraris" ON)
option(CREATE_PYTHON "Create Python interface" ON)

find_package(SWIG REQUIRED)
include(${SWIG_USE_FILE})
message(STATUS "${SWIG_USE_FILE}")

set_source_files_properties(hdeModulePythonWrapper.i PROPERTIES CPLUSPLUS ON)

#add_subdirectory(viewIHumanDatas)
#target_link_libraries(hdeModulePythonWrapper PUBLIC viewIHumanDatas)
add_library(viewIHumanDatas viewIHumanDatas.cpp)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
install(TARGETS viewIHumanDatas)

if(CREATE_PYTHON)
    find_package(Python REQUIRED COMPONENTS Interpreter Development)
    set(CMAKE_SWIG_FLAGS "-Wall;-module;hdeModulePythonWrapper;-threads")

    swig_add_library(hdeModulePythonWrapper
                     LANGUAGE python
                     SOURCES hdeModulePythonWrapper.i)
    target_link_libraries(${SWIG_MODULE_hdeModulePythonWrapper_REAL_NAME} Python::Python viewIHumanDatas ${YARP_LIBRARIES})

    set_target_properties(${SWIG_MODULE_hdeModulePythonWrapper_REAL_NAME} PROPERTIES OUTPUT_NAME "_hdeModulePythonWrapper")

    # installation path is determined reliably on most platforms using distutils
    execute_process(COMMAND ${Python_EXECUTABLE} -c "from distutils import sysconfig; print(sysconfig.get_python_lib(1,0,prefix=''))"
                    OUTPUT_VARIABLE PYTHON_INSTDIR
                    OUTPUT_STRIP_TRAILING_WHITESPACE)

    install(FILES ${CMAKE_CURRENT_BINARY_DIR}/hdeModulePythonWrapper.py DESTINATION ${PYTHON_INSTDIR})

    set(_CMAKE_INSTALL_PYTHONDIR "${PYTHON_INSTDIR}")
    set(CMAKE_INSTALL_PYTHONDIR ${_CMAKE_INSTALL_PYTHONDIR} CACHE PATH "python bindings (${_CMAKE_INSTALL_PYTHONDIR})")
    mark_as_advanced(CMAKE_INSTALL_PYTHONDIR)
    if(NOT IS_ABSOLUTE ${CMAKE_INSTALL_PYTHONDIR})
      set(CMAKE_INSTALL_FULL_PYTHONDIR "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_PYTHONDIR}")
    else()
      set(CMAKE_INSTALL_FULL_PYTHONDIR "${CMAKE_INSTALL_PYTHONDIR}")
    endif()

    # Update RPATH
    if(NOT CMAKE_SKIP_RPATH AND NOT CMAKE_SKIP_INSTALL_RPATH)
      file(RELATIVE_PATH _rel_path "${CMAKE_INSTALL_FULL_PYTHONDIR}" "${CMAKE_INSTALL_FULL_LIBDIR}")
      get_target_property(_current_rpath ${SWIG_MODULE_hdeModulePythonWrapper_REAL_NAME} INSTALL_RPATH)
      if (${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
        list(APPEND _current_rpath "@loader_path/${_rel_path}")
      else()
        list(APPEND _current_rpath "\$ORIGIN/${_rel_path}")
      endif()
      set_target_properties("${SWIG_MODULE_hdeModulePythonWrapper_REAL_NAME}" PROPERTIES INSTALL_RPATH "${_current_rpath}")
    endif()

    install(TARGETS ${SWIG_MODULE_hdeModulePythonWrapper_REAL_NAME} DESTINATION ${PYTHON_INSTDIR})
endif()

