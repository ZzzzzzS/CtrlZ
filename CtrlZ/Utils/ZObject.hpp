/**
 * @file ZObject.hpp
 * @author Zishun Zhou
 * @brief
 *
 * @date 2025-03-10
 *
 * @copyright Copyright (c) 2025
 *
 */
#pragma once
#include <iostream>

namespace z
{
    /**
     * @brief ZObject z fundamental object
     *
     */
    class ZObject
    {
    public:
        ZObject() {}
        ~ZObject() {}

        /**
         * @brief print a split line, default length is 60 and character is '-'
         *
         * @param length length of the line
         * @param c character of the line
         */
        void PrintSplitLine(size_t length = 60, char c = '-')
        {
            for (size_t i = 0; i < length; i++)
            {
                std::cout << c;
            }
            std::cout << std::endl;
        }
    };
};