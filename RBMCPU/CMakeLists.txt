cmake_minimum_required(VERSION 3.0.0 FATAL_ERROR)

################### Variables. ####################
# Change if you want modify path or other values. #
###################################################

set(PROJECT_NAME RBMCPU)
# Output Variables
set(OUTPUT_DEBUG Debug/bin)
set(OUTPUT_RELEASE Release/bin)
# Folders files
set(CPP_DIR_1 ../CudaTest)
set(CPP_DIR_2 ./)
set(CPP_DIR_3 ./)
set(CPP_DIR_4 ./)
set(CPP_DIR_5 ./)
set(CPP_DIR_6 ./)
set(CPP_DIR_7 ./)
set(CPP_DIR_8 ./)
set(CPP_DIR_9 ./)
set(CPP_DIR_10 ./)
set(CPP_DIR_11 ./)
set(CPP_DIR_12 ./)
set(CPP_DIR_13 ./)
set(CPP_DIR_14 ./)
set(CPP_DIR_15 ./)
set(CPP_DIR_16 ./)
set(CPP_DIR_17 ./)
set(CPP_DIR_18 ./)
set(CPP_DIR_19 ./)
set(CPP_DIR_20 ./)
set(CPP_DIR_21 ./)
set(CPP_DIR_22 ./)
set(CPP_DIR_23 ./)
set(CPP_DIR_24 ./)
set(CPP_DIR_25 ./)
set(CPP_DIR_26 ./)
set(CPP_DIR_27 ./)
set(CPP_DIR_28 ./)
set(CPP_DIR_29 ./)
set(CPP_DIR_30 ./)
set(CPP_DIR_31 ./)
set(CPP_DIR_32 ./)
set(CPP_DIR_33 ./)
set(CPP_DIR_34 ./)
set(CPP_DIR_35 ./)
set(CPP_DIR_36 ./)
set(CPP_DIR_37 ./)
set(CPP_DIR_38 ./)
set(CPP_DIR_39 ./)
set(CPP_DIR_40 ./)
set(CPP_DIR_41 ./)
set(CPP_DIR_42 ./)
set(CPP_DIR_43 ./)
set(CPP_DIR_44 ./)
set(CPP_DIR_45 ./)
set(HEADER_DIR_1 ../CudaTest)
set(HEADER_DIR_2 )

############## CMake Project ################
#        The main options of project        #
#############################################

project(${PROJECT_NAME} CXX)

# Define Release by default.
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release")
  message(STATUS "Build type not specified: Use Release by default.")
endif(NOT CMAKE_BUILD_TYPE)

# Definition of Macros
add_definitions(
   -DNDEBUG 
   -D_CONSOLE 
   -DUNICODE
   -D_UNICODE
)

############## Artefacts Output #################
# Defines outputs , depending Debug or Release. #
#################################################

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/${OUTPUT_DEBUG}")
  set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/${OUTPUT_DEBUG}")
  set(CMAKE_EXECUTABLE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/${OUTPUT_DEBUG}")
else()
  set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/${OUTPUT_REL}")
  set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/${OUTPUT_REL}")
  set(CMAKE_EXECUTABLE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/${OUTPUT_REL}")
endif()

################# Flags ################
# Defines Flags for Windows and Linux. #
########################################

if(MSVC)
   set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /W3 /MD /Zi /EHsc")
   set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /W3 /GL /Od /Oi /Gy /Zi /EHsc")
endif(MSVC)
if(NOT MSVC)
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
   if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
       set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libc++")
   endif()
endif(NOT MSVC)

################ Files ################
#   --   Add files to project.   --   #
#######################################

file(GLOB SRC_FILES
    ${CPP_DIR_1}/*.cpp
    ${CPP_DIR_2}/*.cpp
    ${CPP_DIR_3}/*.cpp
    ${CPP_DIR_4}/*.cpp
    ${CPP_DIR_5}/*.cpp
    ${CPP_DIR_6}/*.cpp
    ${CPP_DIR_7}/*.cpp
    ${CPP_DIR_8}/*.cpp
    ${CPP_DIR_9}/*.cpp
    ${CPP_DIR_10}/*.cpp
    ${CPP_DIR_11}/*.cpp
    ${CPP_DIR_12}/*.cpp
    ${CPP_DIR_13}/*.cpp
    ${CPP_DIR_14}/*.cpp
    ${CPP_DIR_15}/*.cpp
    ${CPP_DIR_16}/*.cpp
    ${CPP_DIR_17}/*.cpp
    ${CPP_DIR_18}/*.cpp
    ${CPP_DIR_19}/*.cpp
    ${CPP_DIR_20}/*.cpp
    ${CPP_DIR_21}/*.cpp
    ${CPP_DIR_22}/*.cpp
    ${CPP_DIR_23}/*.cpp
    ${CPP_DIR_24}/*.cpp
    ${CPP_DIR_25}/*.cpp
    ${CPP_DIR_26}/*.cpp
    ${CPP_DIR_27}/*.cpp
    ${CPP_DIR_28}/*.cpp
    ${CPP_DIR_29}/*.cpp
    ${CPP_DIR_30}/*.cpp
    ${CPP_DIR_31}/*.cpp
    ${CPP_DIR_32}/*.cpp
    ${CPP_DIR_33}/*.cpp
    ${CPP_DIR_34}/*.cpp
    ${CPP_DIR_35}/*.cpp
    ${CPP_DIR_36}/*.cpp
    ${CPP_DIR_37}/*.cpp
    ${CPP_DIR_38}/*.cpp
    ${CPP_DIR_39}/*.cpp
    ${CPP_DIR_40}/*.cpp
    ${CPP_DIR_41}/*.cpp
    ${CPP_DIR_42}/*.cpp
    ${CPP_DIR_43}/*.cpp
    ${CPP_DIR_44}/*.cpp
    ${CPP_DIR_45}/*.cpp
    ${HEADER_DIR_1}/*.h
    ${HEADER_DIR_2}/*.h
)

# Add executable to build.
add_executable(${PROJECT_NAME} 
   ${SRC_FILES}
)

if(MSVC)
   target_link_libraries(${PROJECT_NAME} kernel32.lib user32.lib gdi32.lib winspool.lib comdlg32.lib advapi32.lib shell32.lib ole32.lib oleaut32.lib uuid.lib odbc32.lib odbccp32.lib )
endif(MSVC)
