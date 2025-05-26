include(catch2)
include(nlohmann_json)

add_executable(NASOQ_Catch2_Tests
    "main.cpp"
	"test_1.cpp"
)
target_include_directories(NASOQ_Catch2_Tests PRIVATE
    "${PROJECT_SOURCE_DIR}/include/nasoq/")

# Only link to eigen_interface if not in EMBEDDED_ONLY mode
if(NOT EMBEDDED_ONLY)
    target_link_libraries(NASOQ_Catch2_Tests PRIVATE
        nasoq::eigen_interface
        Catch2::Catch2
        nlohmann_json::nlohmann_json
    )
else()
    target_link_libraries(NASOQ_Catch2_Tests PRIVATE
        nasoq::nasoq
        Catch2::Catch2
        nlohmann_json::nlohmann_json
    )
endif()

if(NASOQ_USE_BLAS_MKL AND NASOQ_USE_CLAPACK AND NOT EMBEDDED_ONLY)
    # compares the CLAPACK implementation of the needed LAPACKE functions to the MKL ones
    target_sources(NASOQ_Catch2_Tests PRIVATE "test_clapacke.cpp")
    target_link_libraries(NASOQ_Catch2_Tests PRIVATE mkl::mkl)
endif()

# Add embedded BLAS test when EMBEDDED is enabled
if(EMBEDDED OR EMBEDDED_ONLY)
    # Standalone embedded test executable
    add_executable(EMBEDDED_BLAS_Test 
        "main.cpp"
        "EMBEDDED_Test.cpp"
    )
    target_link_libraries(EMBEDDED_BLAS_Test PRIVATE
        nasoq::nasoq
        Catch2::Catch2
    )
    target_compile_definitions(EMBEDDED_BLAS_Test PRIVATE -DTEST_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
    set_target_properties(EMBEDDED_BLAS_Test PROPERTIES FOLDER tests)
endif()

target_compile_definitions(NASOQ_Catch2_Tests PRIVATE -DTEST_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
set_target_properties(NASOQ_Catch2_Tests PROPERTIES FOLDER tests)
