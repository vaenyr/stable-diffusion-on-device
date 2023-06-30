#include "tokenizer.h"
#include "utils.h"
#include "errors.h"

#include <locale>
#include <cstdlib>
#include <cassert>
#include <regex>


using namespace libsdod;

namespace {


inline Tokenizer::internal_str_type& pb(Tokenizer::internal_str_type& str, const Tokenizer::internal_str_type::value_type c) {
    str.push_back(c);
    return str;
}

Tokenizer::internal_str_type bytes_translate(std::u8string_view const& str) {
    // translates a utf-8 encoded string, byte-by-byte, according to the "bytes_to_unicode" method from the CLIP codebase.
    //
    // Python code, being Python, returns just "str" with the details about the underlying encoding being abstracted away
    // In C/C++ we don't have this luxury, so we have to chose which encoding to use henceforth.
    // Since the bpe used in Python code (which we also use) is decoded in utf-8, here we chose to return utf-8 as well,
    // so we can compare returned values with the values read from the bpe file directly, without any extra processing needed.
    // Note, however, that if in the future a different encoding is more desired, this function would need to be adjusted.
    Tokenizer::internal_str_type ret;
    ret.reserve(str.size()*2);
    for (auto c : str)
        if (c < 33)
            pb(pb(ret, 196), c+128);
        else if (c < 127)
            pb(ret, c);
        else if (c < 158)
            pb(pb(ret, 196), c+34);
        else if (c < 161)
            pb(pb(ret, 197), c-30);
        else if (c < 173)
            pb(pb(ret, 194), c);
        else if (c == 173)
            pb(pb(ret, 197), 131);
        else if (c < 192)
            pb(pb(ret, 194), c);
        else
            pb(pb(ret, 195), c-64);

    return ret;
}

Tokenizer::input_str_type sanitize(Tokenizer::input_str_type const& str) {
    // remove leading and trailing blanks, as well as substitutes sequences of blanks with a single ASCII space
    // character are also converted to lowercase
    if (str.empty())
        return Tokenizer::input_str_type();

    auto ptr = str.data();
    auto size = str.size();
    auto very_end = ptr + size;

    wchar_t wide = 0;
    Tokenizer::input_str_type ret;
    std::shared_ptr<char> tmp{ new char[MB_CUR_MAX] };

    ret.reserve(str.size());

    bool found_char = false;
    bool last_blank = false;
    while (true) {
        if (ptr >= very_end)
            break;
        auto status = std::mbtowc(&wide, reinterpret_cast<const char*>(ptr), size);
        if (status == -1)
            throw libsdod_exception(ErrorCode::INVALID_ARGUMENT, "Invalid UTF-8 string", __func__, __FILE__, STR(__LINE__));
        if (status == 0) {
            assert(ptr+1 == very_end);
            break;
        }

        auto is_blank = std::iswblank(wide);
        if (!is_blank)
            found_char = true;

        if (!is_blank) {
            auto lower = std::tolower(wide);
            if (lower == wide)
                ret.append(ptr, status);
            else {
                auto new_bytes = std::wctomb(tmp.get(), lower);
                ret.append(reinterpret_cast<char8_t*>(tmp.get()), new_bytes);
            }
        } else if (found_char && !last_blank)
            ret.push_back(' ');

        ptr += status;
        size -= status;
        last_blank = is_blank;
    }

    if (found_char && last_blank)
        ret.pop_back();

    return ret;
}




std::regex token_patern{ R"('s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+)", std::regex_constants::icase };


struct token_iter {
    token_iter(Tokenizer::input_str_type const& str) : ptr(str.data()), len(0), end(str.data()+str.size()) {
        _find_token_end();
    }

    operator bool() const { return ptr < end; }
    std::u8string_vew get_token() const { return std::u8string_view{ ptr, ptr+len }; }

    void next() {
        ptr+=len;
        _find_token_end();
    }

    const char8_t* ptr;
    std::size_t len;
    const char8_t* end;

    void _find_token_end() {
        if (ptr >= end) {
            ptr = end;
            len = 0;
            return;
        }

        auto rem = end-ptr;
        wchar_t wide = 0;

        int type = 0;
        while (true) {
            if (!rem) {
                ptr = end;
                len = 0;
                return;
            }
            assert(ptr+len < end);
            if (!type && rem > 1) {
                if (ptr[0] == '\'') {
                    switch (ptr[1]) {
                    case 's': // 's
                    case 't': // 't
                    case 'm': // 'm
                    case 'd': // 'd
                        len = 2;
                        return;
                    }

                    if (rem > 2) {
                        if ((ptr[1] == 'r' && ptr[2] == 'e') || // 're
                            (ptr[1] == 'v' && ptr[2] == 'e') || // 've
                            (ptr[1] == 'l' && ptr[2] == 'l')) { // 'll
                            len = 3;
                            return;
                        }
                    }
                }
            }

            auto status = std::mbtowc(&wide, reinterpret_cast<const char*>(ptr + len), rem);
            if (status <= 0)
                throw libsdod_exception(ErrorCode::INVALID_ARGUMENT, "Invalid UTF-8 string", __func__, __FILE__, STR(__LINE__));

            rem -= status;

            bool is_num = std::iswdigit(wide);
            bool is_letter = std::iswalpha(wide);
            bool is_space = std::iswblank(wide);
            if (!type) {
                assert(len == 0);
                if (is_num) {
                    //we found a number: [\p{N}]
                    len = status;
                    return;
                }
                if (is_letter) {
                    //we found the first element of: [\p{L}]+
                    type = 1;
                    len = status;
                    continue;
                }
                if (!is_space) {
                    //we found the first element of: [^\s\p{L}\p{N}]+
                    type = 2;
                    len = status;
                    continue;
                }

                //the character does not match anything...
                //advance ptr but keep len==0 and search for a beginning of a token again
                ptr += status;
                continue;
            } else if (type == 1) {
                //we are finding a sequence of letters: [\p{L}]+
                if (is_letter) //sequence continues, add current character to len
                    len += status;
                else //sequence finished, return the current len
                    return;
            } else if (type == 2) {
                ///we are finding a sequence of other stuff: [^\s\p{L}\p{N}]+
                if (!is_num && !is_letter && !is_space)
                    len += status;
                else
                    return;
            } else
                assert(false);
        }
    }
}


}


Tokenizer::Tokenizer(std::string const& bpe_file) {
}


std::vector<Tokenizer::token_type> Tokenizer::tokenize(Tokenizer::input_str_type const& str, unsigned int context_len) const {
    std::string prev_loc = std::setlocale(LC_ALL, nullptr);
    auto&& locale_guard = scope_guard([](){ std::setlocale(LC_ALL, "en_US.utf8"); }, [&prev_loc](){ std::setlocale(LC_ALL, prev_loc.c_str()); });
    (void)locale_guard;

    std::vector<Tokenizer::token_type> ret;

    auto copy{ sanitize(str) };
    token_iter it{ copy };
    while (it) {
        auto&& token = bytes_translate(it.get_token());
        it.next();
        bpe(ret, std::move(token));
    }

    return ret;
}

void Tokenizer::bpe(std::vector<token_type>& buff, internal_str_type token) const {
    std::list<internal_str_type> word;
    wchar_t wide = 0;
    auto ptr = token.data();
    auto len = token.size();

    while (len) {
        auto status = std::mbtowc(&wide, reinterpret_cast<const char*>(ptr), len);
        assert(status > 0);
        word.emplace_back(ptr, ptr+status);
        ptr += status;
        len -= status;
    }

    word.back().append(u8"</w>");
    
}