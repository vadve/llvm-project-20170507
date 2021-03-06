set(HWASAN_LIT_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})

set(HWASAN_TESTSUITES)

set(HWASAN_TEST_ARCH ${HWASAN_SUPPORTED_ARCH})

foreach(arch ${HWASAN_TEST_ARCH})
  set(HWASAN_TEST_TARGET_ARCH ${arch})
  string(TOLOWER "-${arch}" HWASAN_TEST_CONFIG_SUFFIX)
  get_test_cc_for_arch(${arch} HWASAN_TEST_TARGET_CC HWASAN_TEST_TARGET_CFLAGS)
  string(TOUPPER ${arch} ARCH_UPPER_CASE)
  set(CONFIG_NAME ${ARCH_UPPER_CASE})

  configure_lit_site_cfg(
    ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.in
    ${CMAKE_CURRENT_BINARY_DIR}/${CONFIG_NAME}/lit.site.cfg)
  list(APPEND HWASAN_TESTSUITES ${CMAKE_CURRENT_BINARY_DIR}/${CONFIG_NAME})
endforeach()

set(HWASAN_TEST_DEPS ${SANITIZER_COMMON_LIT_TEST_DEPS})
if(NOT COMPILER_RT_STANDALONE_BUILD)
  list(APPEND HWASAN_TEST_DEPS hwasan)
endif()

add_lit_testsuite(check-hwasan "Running the HWAddressSanitizer tests"
  ${HWASAN_TESTSUITES}
  DEPENDS ${HWASAN_TEST_DEPS}
  )
set_target_properties(check-hwasan PROPERTIES FOLDER "Compiler-RT Misc")
