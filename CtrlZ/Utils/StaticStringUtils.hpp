/**
 * @file StaticStringUtils.hpp
 * @author zishun zhou
 * @brief
 *
 * @date 2025-03-10
 *
 * @copyright Copyright (c) 2025
 *
 */
#pragma once
#include <string>
#include <array>
#include <algorithm>
#include <tuple>
#include <string_view>
#include <iostream>
#include <type_traits>
#include <any>

#include "MathTypes.hpp"

namespace z
{
    /**
     * @brief compile time string
     *
     * @tparam N size of the string
     */
    template<size_t N>
    struct CTString {
        constexpr CTString(const char(&str)[N])
        {
            std::copy_n(str, N, value);
        }
        char value[N]{};
    };


    /**
     * @brief concat multiple compile time strings
     *
     * @tparam N size of strings
     * @return constexpr auto concatenated string
     */
    template<size_t ...N>
    constexpr auto concat(const CTString<N>& ...strs)
    {
        constexpr size_t total_size = (N + ...) - sizeof...(N) + 1;
        char result[total_size]{};
        size_t offset = 0;
        ((std::copy_n(strs.value, N - 1, result + offset), offset += N - 1), ...);
        return CTString<total_size>{ result };
    }

    /**
     * @brief concat a compile time string with a const string
     *
     * @tparam N size of first string
     * @tparam M size of second string
     * @return constexpr auto concatenated string
     */
    template<size_t N, size_t M>
    constexpr auto concat(const CTString<N>& str1, const char(&str2)[M])
    {
        char result[N + M - 1]{};
        std::copy_n(str1.value, N - 1, result);
        std::copy_n(str2, M, result + N - 1);
        return CTString<N + M - 1>{ result };
    }

    /**
     * @brief concat multiple compile time strings
     *
     * @tparam N size of strings
     * @return constexpr auto concatenated string
     */
    template<size_t ...N>
    constexpr auto concat(const char(&...str)[N])
    {
        constexpr size_t total_size = (N + ...) - sizeof...(N) + 1;
        char result[total_size]{};
        size_t offset = 0;
        ((std::copy_n(str, N - 1, result + offset), offset += N - 1), ...);
        return CTString<total_size>{ result };
    }





    /**
     * @brief compile time string key-value pair
     *
     * @tparam CT compile time string key
     * @tparam T value type
     */
    template <CTString CT, typename T>
    struct CTSPair {
        static constexpr CTString str{ CT };
        T* type = {};
        static constexpr bool isArray = false;
        static constexpr size_t dim = 1;
    };

    /**
     * @brief compile time string key-value pair for array (std array)
     *
     * @tparam CT compile time string key
     * @tparam T value type array
     * @tparam Dim array size
     */
    template <CTString CT, typename T, size_t Dim>
    struct CTSPair<CT, std::array<T, Dim>>
    {
        static constexpr CTString str{ CT };
        std::array<T, Dim>* type = {};
        static constexpr bool isArray = true;
        static constexpr size_t dim = Dim;
    };

    /**
     * @brief compile time string key-value pair for array (z::math::Vector)
     *
     * @tparam CT compile time string key
     * @tparam T value type array
     * @tparam Dim array size
     *
     * @details constexpr z::CTSPair<"InferenceTime", float> InferenceTimePair;
     * //explain this create a compile time string key-value pair with key "InferenceTime" and value type float.
     * //this can be used in CTSMap and work as fundimation to get and set value in DataCenter, Scheduler, and Workers classes.
     */
    template <CTString CT, typename T, size_t Dim>
    struct CTSPair<CT, math::Vector<T, Dim>>
    {
        static constexpr CTString str{ CT };
        math::Vector<T, Dim>* type = {};
        static constexpr bool isArray = true;
        static constexpr size_t dim = Dim;
    };


    /**
     * @brief compile time string key-value pair for array (POD array)
     *
     * @tparam CT compile time string key
     * @tparam T value type array
     * @tparam Dim array size
     */
    template <CTString CT, typename T, size_t Dim>
    struct CTSPair<CT, T[Dim]>
    {
        static constexpr CTString str{ CT };
        T* type = {};
        static constexpr bool isArray = true;
        static constexpr size_t dim = Dim;
    };

    /**
     * @brief print compile time string key-value pair info
     *
     * @tparam CTS
     */
    template<CTSPair CTS>
    void PrintCTSPairInfo()
    {
        std::cout << "[\"" << CTS.str.value << "\"] = " << typeid(typename std::remove_pointer_t<decltype(CTS.type)>).name() << std::endl;
    }

    /**
     * @brief compile time string array
     *
     * @tparam CTS compile time strings
     */
    template<CTString ...CTS>
    class CTSArray
    {
    public:
        /**
         * @brief construct a new CTSArray object
         *
         */
        constexpr CTSArray() {};

        /**
         * @brief get the size of the array in compile time
         *
         * @return constexpr size_t
         */
        static constexpr size_t size()
        {
            return sizeof...(CTS);
        }

        /**
         * @brief get the index of a compile time string in the array during compile time
         *
         * @tparam CT compile time string to search
         * @return constexpr size_t index of the compile time string in the array
         */
        template<CTString CT>
        static constexpr size_t index()
        {
            constexpr auto it = std::find(string_array_.begin(), string_array_.end(), std::string_view(CT.value));
            constexpr size_t idx = it != string_array_.end() ? std::distance(string_array_.begin(), it) : sizeof...(CTS);

            //constexpr auto error_message = concat("CTS not found in CTSArray: ", CT.value);

            static_assert(idx != sizeof...(CTS), "CTS not found in CTSMap");
            return idx;
        }

        /**
         * @brief get the index of a compile time string in the array during runtime
         *
         * @param str  runtime string to search
         * @return static size_t index of the compile time string in the array
         */
        static size_t index(const std::string_view& str)
        {
            const auto it = std::find(string_array_.begin(), string_array_.end(), str);
            if (it != string_array_.end())
            {
                return std::distance(string_array_.begin(), it);
            }
            else
            {
                return sizeof...(CTS);
            }
        }

    private:
        constexpr static std::array<std::string_view, sizeof...(CTS)> string_array_{ CTS.value... };
    };



    /**
     * @brief Compile time string map
     *
     * @tparam CTS compile time string key-value pairs
     * @details CTSMap<CTSPair<"key1", int>, CTSPair<"key2", float>> ctsMap;
     */
    template<CTSPair ...CTS>
    class CTSMap
    {
    public:
        /**
         * @brief Construct a new CTSMap object
         *
         */
        CTSMap() {}

        /**
         * @brief Destroy the CTSMap object
         *
         */
        ~CTSMap() {}

        /**
         * @brief get the value of a compile time string key
         *
         * @tparam CT compile time string key
         * @tparam T value type
         * @param value value to get
         */
        template<CTString CT, typename T>
        void get(T& value)
        {
            constexpr size_t idx = ctsArray.template index<CT>();
            //constexpr auto error_message = concat("CTS not found in CTSMap: ", CT.value);
            static_assert(idx != sizeof...(CTS), "CTS not found in CTSMap");
            value = std::get<idx>(ctsValue);
        }

        /**
         * @brief get the size of the map
         *
         * @return size_t size of the map
         */
        size_t size()
        {
            return sizeof...(CTS);
        }

        /**
         * @brief get the index of a compile time string key
         *
         * @tparam CT compile time string key
         * @return size_t index of the compile time string key
         */
        template<CTString CT>
        static constexpr size_t index()
        {
            return ctsArray.template index<CT>();
        }

        /**
         * @brief get the index of a runtime string key
         *
         * @param str runtime string key
         * @return size_t index of the runtime string key
         */
        size_t index(const std::string_view& str)
        {
            return ctsArray.index(str);
        }


        /**
         * @brief set the value of a compile time string key
         *
         * @tparam CT compile time string key
         * @tparam T value type
         * @param value value to set
         */
        template<CTString CT, typename T>
        void set(const T& value)
        {
            constexpr size_t idx = ctsArray.template index<CT>();
            //constexpr auto error_message = concat("CTS not found in CTSMap: ", CT.value);
            static_assert(idx != sizeof...(CTS), "CTS not found in CTSMap");
            std::get<idx>(ctsValue) = value;
        }

    private:
        /// @brief compile time string array
        static constexpr CTSArray<CTS.str...> ctsArray = {};

        /// @brief compile time string value
        std::tuple<std::remove_pointer_t<decltype(CTS.type)>...> ctsValue;
    };
};