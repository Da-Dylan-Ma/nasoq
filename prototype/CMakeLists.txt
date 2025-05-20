cmake_minimum_required(VERSION 3.10)

# Allow this to be included in the parent project or built standalone
if(NOT DEFINED PROJECT_NAME)
    project(nasoq_prototype)
endif()

# Set C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Library sources
set(PROTOTYPE_SOURCES
    src/hybrid_active_nasoq.cpp
    src/batch_constraint_handler.cpp
    src/matrix_free_solvers.cpp
    src/constraint_relaxation.cpp
    src/memory_manager.cpp
)

# Library headers
set(PROTOTYPE_HEADERS
    include/hybrid_active_nasoq.h
    include/memory_manager.h
    include/matrix_free_solvers.h
)

# Create the library
add_library(nasoq_prototype ${PROTOTYPE_SOURCES})

# Include directories
target_include_directories(nasoq_prototype
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# Link dependencies
target_link_libraries(nasoq_prototype
    PUBLIC
        nasoq
)

# Example applications
if(NASOQ_BUILD_EXAMPLES)
    add_executable(simple_qp_example examples/simple_qp.cpp)
    target_link_libraries(simple_qp_example nasoq_prototype)

    add_executable(mpc_example examples/mpc_example.cpp)
    target_link_libraries(mpc_example nasoq_prototype)

    add_executable(benchmark examples/benchmark.cpp)
    target_link_libraries(benchmark nasoq_prototype)
endif()

# Tests
if(NASOQ_BUILD_TESTS)
    # Find Google Test
    find_package(GTest QUIET)
    
    if(GTest_FOUND)
        enable_testing()
        
        # Add test executables
        add_executable(simple_qp_tests tests/simple_qp_tests.cpp)
        target_link_libraries(simple_qp_tests nasoq_prototype GTest::GTest GTest::Main)
        add_test(NAME simple_qp_tests COMMAND simple_qp_tests)
        
        add_executable(batch_processing_tests tests/batch_processing_tests.cpp)
        target_link_libraries(batch_processing_tests nasoq_prototype GTest::GTest GTest::Main)
        add_test(NAME batch_processing_tests COMMAND batch_processing_tests)
        
        add_executable(matrix_free_tests tests/matrix_free_tests.cpp)
        target_link_libraries(matrix_free_tests nasoq_prototype GTest::GTest GTest::Main)
        add_test(NAME matrix_free_tests COMMAND matrix_free_tests)
        
        add_executable(memory_tests tests/memory_tests.cpp)
        target_link_libraries(memory_tests nasoq_prototype GTest::GTest GTest::Main)
        add_test(NAME memory_tests COMMAND memory_tests)
    else()
        message(STATUS "Google Test not found, skipping tests")
    endif()
endif()

# Installation
install(
    TARGETS nasoq_prototype
    EXPORT nasoq_prototype-targets
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
    INCLUDES DESTINATION include
)

install(
    FILES ${PROTOTYPE_HEADERS}
    DESTINATION include/nasoq/prototype
)

# Export targets
install(
    EXPORT nasoq_prototype-targets
    FILE nasoq_prototype-targets.cmake
    NAMESPACE nasoq::
    DESTINATION lib/cmake/nasoq_prototype
) 