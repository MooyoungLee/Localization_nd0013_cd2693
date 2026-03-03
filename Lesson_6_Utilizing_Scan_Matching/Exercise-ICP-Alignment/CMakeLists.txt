cmake_minimum_required (VERSION 3.10)
if(POLICY CMP0074)
  cmake_policy(SET CMP0074 NEW)
endif()
project(Localization)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# -g allows for gdb debugging
# turn on -03 for best performance
if(MSVC)
  add_compile_options(/W4 /O2 /DNDEBUG)
else()
  add_definitions(-std=c++14 -O3)
  set(CXX_FLAGS "-pthread -fPIC -DNDEBUG -Wextra")
  set(CMAKE_CXX_FLAGS "${CXX_FLAGS} -Wl,-Bstatic -lboost_filesystem -Wl,-Bdynamic")
endif()

set(sources sm1-main.cpp helper.cpp)

find_package(PCL 1.2 REQUIRED)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")

endif(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")

add_executable(scan_matching_1 ${sources} )

target_link_libraries(scan_matching_1 ${PCL_LIBRARIES} )
