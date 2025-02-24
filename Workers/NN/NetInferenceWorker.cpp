#include "NetInferenceWorker.h"
#include "onnxruntime_cxx_api.h"
#include <string>
#include <locale>
#include <codecvt>
#include <iostream>

/**
 * @brief singleton Ort::Env object
 */
Ort::Env& GetOrtEnv()
{
	static Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "NetInferenceWorker");
	return env;
}

std::wstring string_to_wstring(const std::string& str) {
	std::wstring_convert<std::codecvt_utf8_utf16<wchar_t>> converter;
	return converter.from_bytes(str);
}