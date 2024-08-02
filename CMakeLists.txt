cmake_minimum_required(VERSION 2.8)
project( CCM )

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )
# add_executable( CCM src/main.cpp src/Linear_CCM.cpp src/Linear_CCM.hpp)
# add_executable( CCM src/test.cpp src/Linear_CCM.cpp src/Linear_CCM.hpp)
# add_executable( CCM src/process_image.cpp src/Linear_CCM.cpp src/Linear_CCM.hpp)
# add_executable( CCM src/applyvideo2ccm.cpp src/mylib/Linear_CCM.cpp src/mylib/Linear_CCM.hpp)

add_executable( CCM src/applyhsl2video.cpp src/mylib/hsl.hpp src/mylib/hsl.cpp)
# add_executable(CCM src/load_ccm2video.cpp)


target_link_libraries( CCM ${OpenCV_LIBS})