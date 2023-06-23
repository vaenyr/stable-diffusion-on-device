#ifndef LIBSD_ERRORS_H
#define LIBSD_ERRORS_H

#include "context.h"

namespace libsd {

enum ErrorCode {
    NO_ERROR,
    INVALID_CONTEXT,
    INVALID_ARGUMENT,
    FAILED_ALLOCATION,
    RUNTIME_ERROR,
    INTERNAL_ERROR,
};

constexpr unsigned int LIBSD_NUM_ERRORS = 6;

using ErrorTable = std::string*;

ErrorTable allocate_error_table();
bool is_valid_error_code(int errorcode);


void record_error(ErrorTable tab, ErrorCode error);
void record_error(ErrorTable tab, ErrorCode error, std::string const& extra_info);
void record_error(ErrorTable tab, ErrorCode error, std::string&& extra_info);


const char* get_error_str(ErrorCode code);
const char* get_last_error_info(ErrorTable tab, ErrorCode error);


class libsd_exception : public std::exception {
public:
    libsd_exception(ErrorCode code, std::string msg, const char* func, const char* file, const char* line)
        : _code(code), _reason(std::move(msg)), _func(func), _file(file), _line(line)
    {}

    virtual const char* what() const noexcept;

    ErrorCode code() const noexcept { return _code; }
    const char* reason() const noexcept { return _reason.c_str(); }
    const char* func() const noexcept { return _func.c_str(); }
    const char* file() const noexcept { return _file.c_str(); }
    const char* line() const noexcept { return _line.c_str(); }

private:
    ErrorCode _code;
    std::string _reason;
    std::string _func;
    std::string _file;
    std::string _line;
};

}

#endif // LIBSD_ERRORS_H
