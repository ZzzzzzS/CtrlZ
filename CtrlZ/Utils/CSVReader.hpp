/****************************************************************************
MIT License

Copyright (c) 2024 zishun zhou (zhouzishun@mail.zzshub.cn)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*****************************************************************************/
#pragma once

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <map>
#include <type_traits>


namespace z
{
    template<typename T>
    class CSVReader
    {
    public:
        using Ptr = std::shared_ptr<CSVReader<T>>;
        static Ptr Create(const std::string& _FileName, const bool& _HasHeader = true)
        {
            return std::make_shared<CSVReader<T>>(_FileName, _HasHeader);
        }
        static_assert(std::is_arithmetic<T>::value, "CSVReader only supports numeric types.");

    public:
        CSVReader(std::string filename, bool has_header = true)
            : filename_(filename)
        {

            //open file and read data
            std::ifstream file(filename_);
            if (!file.is_open())
            {
                std::cout << "Error opening file" << std::endl;
            }
            else
            {
                int row_count = 0;
                std::string line;
                while (std::getline(file, line))
                {
                    line.erase(std::remove_if(line.begin(), line.end(), ::isspace), line.end());
                    std::vector<std::string> row;
                    row = stringSplit(line, ',');
                    if (has_header && row_count == 0)
                    {
                        for (int i = 0; i < row.size(); i++)
                        {
                            if (!row[i].empty())
                                header_[row[i]] = i;
                        }
                    }
                    else
                    {
                        //data_.push_back(row);
                        std::vector<T> row_numeric;
                        for (const auto& item : row)
                        {
                            if (!item.empty())
                                row_numeric.push_back(std::stod(item));
                        }
                        data_numeric_.push_back(row_numeric);
                    }
                    row_count++;
                }
            }
        }

        size_t RowSize() const
        {
            return this->data_numeric_.size();
        }

        size_t ColumnSize() const
        {
            if (this->data_numeric_.empty())
            {
                return 0;
            }
            return this->data_numeric_[0].size();
        }

        std::vector<T> getRow(int Row)
        {
            std::vector<T> row;
            if (Row < data_numeric_.size())
            {
                row = data_numeric_[Row];
            }
            return row;
        }

        std::vector<T> getColumn(int Column)
        {
            std::vector<T> column;
            if (Column < data_numeric_[0].size())
            {
                for (int i = 0; i < data_numeric_.size(); i++)
                {
                    column.push_back(data_numeric_[i][Column]);
                }
            }
            return column;
        }

        T getItem(int row, int column)
        {
            if (row < data_numeric_.size() && column < data_numeric_[0].size())
            {
                return data_numeric_[row][column];
            }
            else
            {
                std::cout << "Row " << row << " or Column " << column << " not found in data" << std::endl;
                return 0;
            }
        }

        std::vector<T> getItems(const std::vector<std::string>& items, size_t row)
        {
            std::vector<T> result;
            //check row 
            if (row >= data_numeric_.size())
            {
                std::cout << "Row " << row << " not found in data" << std::endl;
                return result;
            }
            for (const auto& item : items)
            {
                if (header_.find(item) != header_.end())
                {
                    result.push_back(data_numeric_[row][header_[item]]);
                }
                else
                {
                    std::cout << "Item " << item << " not found in header" << std::endl;
                }
            }
            return result;
        }

        friend std::ostream& operator<<(std::ostream& os, const CSVReader& reader)
        {
            for (const auto& row : reader.data_numeric_)
            {
                for (const auto& item : row)
                {
                    //print item
                    os << item << " ";
                }
                os << std::endl;
            }
            return os;
        }


    private:
        std::vector<std::string> stringSplit(const std::string& str, char delim) {
            std::size_t previous = 0;
            std::size_t current = str.find(delim);
            std::vector<std::string> elems;
            while (current != std::string::npos) {
                if (current > previous) {
                    elems.push_back(str.substr(previous, current - previous));
                }
                previous = current + 1;
                current = str.find(delim, previous);
            }
            if (previous != str.size()) {
                elems.push_back(str.substr(previous));
            }
            return elems;
        }

    private:
        std::string filename_;
        std::map<std::string, int> header_;
        std::vector<std::vector<T>> data_numeric_;

    };
};
