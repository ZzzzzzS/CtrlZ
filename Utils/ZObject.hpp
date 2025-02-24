#pragma once
#include <iostream>

namespace zzs
{
    /**
     * @brief ZObject zzs fundamental object
     *
     */
    class ZObject
    {
    public:
        ZObject() {}
        ~ZObject() {}

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