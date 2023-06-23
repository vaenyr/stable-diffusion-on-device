#ifndef LIBSD_BUFFER_H
#define LIBSD_BUFFER_H

template <class T>
class Buffer;

template <>
class Buffer<void> {
public:
    Buffer();
    Buffer(void* ptr);

};

template <class T>
class Buffer : Buffer<void> {

};



#endif // LIBSD_BUFFER_H
