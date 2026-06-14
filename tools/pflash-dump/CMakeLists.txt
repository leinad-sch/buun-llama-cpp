# llama-pflash-dump: offline PFlash importance harness (EXP-15d content axis)

set(TARGET llama-pflash-dump)

add_executable(${TARGET} pflash-dump.cpp)
target_link_libraries(${TARGET} PRIVATE llama-common llama ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)

find_package(OpenMP QUIET)
if(OpenMP_CXX_FOUND)
    target_link_libraries(${TARGET} PRIVATE OpenMP::OpenMP_CXX)
endif()

if(LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET} RUNTIME)
endif()
