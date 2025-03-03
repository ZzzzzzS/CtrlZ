cmake_minimum_required(VERSION 3.24)

get_property(
  BITBOT_CMAKE_MODULE_INITIALIZED GLOBAL ""
  PROPERTY BITBOT_CMAKE_MODULE_INITIALIZED
  SET
)
if(BITBOT_CMAKE_MODULE_INITIALIZED)
  return()
endif()

set_property(GLOBAL PROPERTY BITBOT_CMAKE_MODULE_INITIALIZED true)

include(FetchContent)

function(BitbotProxyUrl input_url output_url)
  string(FIND ${input_url} "http://" USE_HTTP)
  string(FIND ${input_url} "https://" USE_HTTPS)
  if(NOT ${USE_HTTP} EQUAL "-1")
    string(REPLACE "http://" "http://mirror.ghproxy.com/" input_url ${input_url})
  elseif(NOT ${USE_HTTPS} EQUAL "-1")
    string(REPLACE "https://" "https://mirror.ghproxy.com/" input_url ${input_url})
  endif()
  set(${output_url} ${input_url} PARENT_SCOPE)
endfunction(BitbotProxyUrl)

# splits a package option
function(bitbot_parse_option OPTION)
  string(REGEX MATCH "^[^ ]+" OPTION_KEY "${OPTION}")
  string(LENGTH "${OPTION}" OPTION_LENGTH)
  string(LENGTH "${OPTION_KEY}" OPTION_KEY_LENGTH)
  if(OPTION_KEY_LENGTH STREQUAL OPTION_LENGTH)
    # no value for key provided, assume user wants to set option to "ON"
    set(OPTION_VALUE "ON")
  else()
    math(EXPR OPTION_KEY_LENGTH "${OPTION_KEY_LENGTH}+1")
    string(SUBSTRING "${OPTION}" "${OPTION_KEY_LENGTH}" "-1" OPTION_VALUE)
  endif()
  set(OPTION_KEY
      "${OPTION_KEY}"
      PARENT_SCOPE
  )
  set(OPTION_VALUE
      "${OPTION_VALUE}"
      PARENT_SCOPE
  )
endfunction()

macro(BitbotAddDependency)
  set(optionsArgs "")
  set(oneValueArgs
    NAME
    GIT_REPO
    GIT_TAG
    GIT_SHALLOW
    URL
    OVERRIDE_FIND_PACKAGE
    FILE_NAME
    FILE_PATH
    USE_PROXY
    USE_LOCAL_FILE
  )
  set(multiValueArgs OPTIONS)
  cmake_parse_arguments(BIT_ARGS "${optionsArgs}" "${oneValueArgs}" "${multiValueArgs}" "${ARGN}")

  if(NOT DEFINED BIT_ARGS_USE_PROXY)
    set(BIT_ARGS_USE_PROXY FALSE)
  endif()
  if(NOT DEFINED BIT_ARGS_USE_LOCAL_FILE)
    set(BIT_ARGS_USE_LOCAL_FILE FALSE)
  endif()

  foreach(OPTION ${BIT_ARGS_OPTIONS})
    bitbot_parse_option("${OPTION}")
    set(${OPTION_KEY} "${OPTION_VALUE}")
  endforeach()

  if(${BIT_ARGS_USE_LOCAL_FILE})
    string(PREPEND BIT_ARGS_FILE_NAME "${BIT_ARGS_FILE_PATH}/")
    message(STATUS "Bitbot dependency:${BIT_ARGS_NAME} local_file:${BIT_ARGS_FILE_NAME}")
    FetchContent_Declare(
      ${BIT_ARGS_NAME}
      OVERRIDE_FIND_PACKAGE ${BIT_ARGS_OVERRIDE_FIND_PACKAGE}
      URL           "${BIT_ARGS_FILE_NAME}"
    )
    FetchContent_MakeAvailable(${BIT_ARGS_NAME})
  elseif(DEFINED BIT_ARGS_URL)
    set(USE_URL TRUE)
    if(${BIT_ARGS_USE_PROXY})
      BitbotProxyUrl(${BIT_ARGS_URL} BIT_ARGS_URL)
    endif()
    message(STATUS "Bitbot dependency:${BIT_ARGS_NAME} URL:${BIT_ARGS_URL}")
    FetchContent_Declare(
      ${BIT_ARGS_NAME}
      OVERRIDE_FIND_PACKAGE ${BIT_ARGS_OVERRIDE_FIND_PACKAGE}
      URL           "${BIT_ARGS_URL}"
    )
    FetchContent_MakeAvailable(${BIT_ARGS_NAME})
  elseif(DEFINED BIT_ARGS_GIT_REPO)
    if(${BIT_ARGS_USE_PROXY})
      BitbotProxyUrl(${BIT_ARGS_GIT_REPO} BIT_ARGS_GIT_REPO)
    endif()
    message(STATUS "Bitbot dependency:${BIT_ARGS_NAME} GIT_REPO:${BIT_ARGS_GIT_REPO}")
    FetchContent_Declare(
      ${BIT_ARGS_NAME}
      GIT_REPOSITORY ${BIT_ARGS_GIT_REPO}
      GIT_TAG        ${BIT_ARGS_GIT_TAG}
      GIT_SHALLOW    ${BIT_ARGS_GIT_SHALLOW}
      OVERRIDE_FIND_PACKAGE ${BIT_ARGS_OVERRIDE_FIND_PACKAGE}
    )
    FetchContent_MakeAvailable(${BIT_ARGS_NAME})
  endif()
endmacro(BitbotAddDependency)
