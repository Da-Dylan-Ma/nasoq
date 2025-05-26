add_executable(NASOQ_Test "nasoq_main.cpp")
target_link_libraries(NASOQ_Test PRIVATE nasoq::nasoq)
target_include_directories(NASOQ_Test PRIVATE
        "${PROJECT_SOURCE_DIR}/include/")
        
# When building with embedded implementation, add EMBEDDED define to NASOQ_Test
if(EMBEDDED)
    target_compile_definitions(NASOQ_Test PRIVATE "EMBEDDED")
    message(STATUS "NASOQ_Test will use embedded BLAS implementation")
endif()

set_target_properties(NASOQ_Test PROPERTIES FOLDER examples)

add_executable(NASOQ_Step_Test "nasoq_step_main.cpp")
target_link_libraries(NASOQ_Step_Test PRIVATE nasoq::nasoq)
target_include_directories(NASOQ_Step_Test PRIVATE
        "${PROJECT_SOURCE_DIR}/include/")
set_target_properties(NASOQ_Step_Test PROPERTIES FOLDER examples)

add_executable(LBL_Test "lbl_main.cpp")
target_link_libraries(LBL_Test PRIVATE nasoq::nasoq)
set_target_properties(LBL_Test PROPERTIES FOLDER examples)

add_executable(LBLSOMOD_Test "lbl_somod_main.cpp")
target_link_libraries(LBLSOMOD_Test PRIVATE nasoq::nasoq)
target_include_directories(LBLSOMOD_Test PRIVATE
        "${PROJECT_SOURCE_DIR}/include/")
set_target_properties(LBLSOMOD_Test PROPERTIES FOLDER examples)

add_executable(EMBEDDED_Test "embedded_test.cpp")
target_link_libraries(EMBEDDED_Test PRIVATE nasoq::nasoq)
target_include_directories(EMBEDDED_Test PRIVATE
        "${PROJECT_SOURCE_DIR}/include/")
if(EMBEDDED)
    target_compile_definitions(EMBEDDED_Test PRIVATE "EMBEDDED")
endif()
set_target_properties(EMBEDDED_Test PROPERTIES FOLDER examples)
