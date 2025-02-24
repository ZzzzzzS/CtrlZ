#pragma once

#include <string>
#include <vector>
#include <map>
#include <array>
#include <any>
#include <mutex>
#include "../Utils/ZObject.hpp"
#include "../Utils/StaticStringUtils.hpp"
#include "atomic"

namespace zzs
{
    /*
    高性能读写队列
    我们要实现的这个不是一个队列，在读取完数据之后不会把数据pop出去，而是会保留下来，以便下一次继续读取。
    使用类似无锁队列的方式。有一个时间戳来指定读写的先后顺序。
    有一个最大并发读写数，这个数值是固定的，不会改变。
    队列底层实现是ringbuffer，读写指针分开，读写指针之间的数据是可以被读取的。

    两个环形缓冲区，一个用来存储数据，一个用来标记读写占用情况?再来一个缓冲区用来存储时间戳？
    每次写新数据必然会往后push一位，所以并发写入是可以实现的。
    并发写入时加一个shared_mutex， 读的时候从最后一位开始读，并一直寻找到一个可以读的位置并加上读锁开始读。

    是否需要保证一个数据在一个时间戳内只能写一次？我觉得这是有必要的，防止数据污染。
    如果只需要保证一个时间戳内只能写一次，那么是不是可以简化读写锁的实现?好像只能手动限制，按照目前的读写锁+多缓冲区的实现方式来看，
    看起来似乎是不太能直接实现这个？

    看来不能使用锁的实现，因为锁在wait的时候必然导致上下文切换。原子类型和锁的不同在于原子类型是不会阻塞的，原子类型不会堵住线程，所以不需要进行上下文切换，所以快。

    写的时候写指针自动向后移动一位，所以多并发写入是不需要锁的。
    读的时候也是可以不用锁的，反正从后往前找到一个写标志位为0的位置就可以了。
    多个读也可以一起读，多个读就多个增加flag。写的时候判断一下循环缓冲区中读的flag是否为0，如果不为零就报错，或者自动扩充？或者就只能等着了。

    */
    //TODO: 无锁ring buffer真tmd难，稍后再实现这个吧
    template<CTSPair ...CTS>
    class DataCenter
    {
    public:
        DataCenter()
        {

        }

        ~DataCenter()
        {
        }

        template<CTString CT, typename T>
        void SetData(size_t TimeStamp, const T& data)
        {
            size_t idx = this->dataBuffers_.template index<CT>();
            this->timeStamps_[idx] = TimeStamp;

            this->writelocks_[idx].lock();
            this->dataBuffers_.template set<CT, T>(data);
            this->writelocks_[idx].unlock();
        }

        template<CTString CT, typename T>
        size_t GetData(T& data)
        {
            size_t idx = this->dataBuffers_.template index<CT>();
            this->writelocks_[idx].lock();
            this->dataBuffers_.template get<CT, T>(data);
            this->writelocks_[idx].unlock();
            return this->timeStamps_[idx];
        }


    private:
        std::array<std::atomic_size_t, sizeof...(CTS)> timeStamps_{ 0 };
        std::array<std::mutex, sizeof...(CTS)> writelocks_;

        CTSMap<CTS...> dataBuffers_{};

    };
};