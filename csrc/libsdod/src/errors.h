#ifndef LIBSDOD_ERRORS_H
#define LIBSDOD_ERRORS_H

#include <array>
#include <optional>
#include <string>
#include <memory>


namespace libsdod {

enum class ErrorCode : int {
    NO_ERROR,
    INVALID_CONTEXT,
    INVALID_ARGUMENT,
    FAILED_ALLOCATION,
    RUNTIME_ERROR,
    INTERNAL_ERROR,
};

constexpr unsigned int LIBSDOD_NUM_ERRORS = 6;

using ErrorTable = std::shared_ptr<std::array<std::optional<std::string>, LIBSDOD_NUM_ERRORS>>;

ErrorTable allocate_error_table();

bool is_valid_error_code(int errorcode);

void record_error(ErrorTable tab, ErrorCode error);
void record_error(ErrorTable tab, ErrorCode error, std::string const& extra_info);
void record_error(ErrorTable tab, ErrorCode error, std::string&& extra_info);


const char* get_error_str(ErrorCode code);
const char* get_last_error_info(ErrorTable tab, ErrorCode error);


class libsdod_exception : public std::exception {
public:
    libsdod_exception(ErrorCode code, std::string msg, const char* func, const char* file, const char* line);

    virtual const char* what() const noexcept { return _what.c_str(); }

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

    std::string _what;
};

}

#endif // LIBSDOD_ERRORS_H
