#include "tokenizer.h"
#include "utils.h"
#include "errors.h"

#include <locale>
#include <cstdlib>
#include <cassert>
#include <fstream>
#include <clocale>


using namespace libsdod;

namespace {


inline std::string& pb(std::string& str, const unsigned char c) {
    str.push_back(reinterpret_cast<const char&>(c));
    return str;
}

std::string bytes_translate(std::string_view const& str) {
    // translates a utf-8 encoded string, byte-by-byte, according to the "bytes_to_unicode" method from the CLIP codebase.
    //
    // Python code, being Python, returns just "str" with the details about the underlying encoding being abstracted away
    // In C/C++ we don't have this luxury, so we have to chose which encoding to use henceforth.
    // Since the bpe used in Python code (which we also use) is decoded in utf-8, here we chose to return utf-8 as well,
    // so we can compare returned values with the values read from the bpe file directly, without any extra processing needed.
    // Note, however, that if in the future a different encoding is more desired, this function would need to be adjusted.
    std::string ret;
    ret.reserve(str.size()*2);
    for (auto _c : str) {
        auto& c = reinterpret_cast<const unsigned char&>(_c);
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
    }

    return ret;
}

std::string sanitize(std::string const& str) {
    // remove leading and trailing blanks, as well as substitutes sequences of blanks with a single ASCII space
    // character are also converted to lowercase
    if (str.empty())
        return std::string{};

    auto ptr = str.data();
    auto size = str.size();
    auto very_end = ptr + size;

    wchar_t wide = 0;
    std::string ret;
    std::shared_ptr<char> tmp{ new char[MB_CUR_MAX] };

    ret.reserve(str.size());

    bool found_char = false;
    bool last_blank = false;
    while (true) {
        if (ptr >= very_end)
            break;
        auto status = std::mbtowc(&wide, ptr, size);
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
                ret.append(tmp.get(), new_bytes);
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


// std::regex token_patern{ R"('s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+)", std::regex_constants::icase };

struct token_iter {
    token_iter(std::string const& str) : ptr(str.data()), len(0), end(str.data()+str.size()) {
        _find_token_end();
    }

    operator bool() const { return ptr < end; }
    std::string_view get_token() const { return std::string_view{ ptr, len }; }

    void next() {
        ptr += len;
        len = 0;
        _find_token_end();
    }

    const char* ptr;
    std::size_t len;
    const char* end;

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

            auto status = std::mbtowc(&wide, ptr + len, rem);
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
};


}


Tokenizer::Tokenizer(std::string const& bpe_file) {
    std::ifstream in(bpe_file, std::ios::binary);
    if (!in)
        throw libsdod_exception(ErrorCode::INVALID_ARGUMENT, format("Tokenizer file {} does not exist", bpe_file), __func__, __FILE__, STR(__LINE__));

    std::string str;

    unsigned int next_rank = 0;
    token_type next_token = 0;

    while (in) {
        std::getline(in, str);
        if (str.empty())
            continue;
        auto space = str.find(' ');
        if (space == str.npos) {
            tokens.emplace(std::move(str), next_token++);
        } else {
            std::string first{ str.data(), space };
            std::string second{ str.data()+space+1, str.size()-space-1 };
            tokens.emplace(first+second, next_token++);
            ranks.emplace(std::make_pair(std::move(first), std::move(second)), next_rank++);
        }
    }

    start_token = next_token++;
    end_token = next_token++;
}


void Tokenizer::tokenize(std::vector<Tokenizer::token_type>& out, std::string const& str, unsigned int context_len) const {
    std::string prev_loc = std::setlocale(LC_ALL, nullptr);
    auto&& locale_guard = scope_guard([](){ std::setlocale(LC_ALL, "en_US.utf8"); }, [&prev_loc](){ std::setlocale(LC_ALL, prev_loc.c_str()); });
    (void)locale_guard;

    out.clear();
    out.push_back(start_token);

    auto copy{ sanitize(str) };
    token_iter it{ copy };
    while (it) {
        auto&& token = bytes_translate(it.get_token());
        bpe(out, std::move(token), context_len-1);
        it.next();
    }

    while (out.size() < context_len)
        out.push_back(end_token);
}


void Tokenizer::bpe(std::vector<token_type>& buff, std::string token, unsigned max_len) const {
    if (buff.size() >= max_len)
        return;

    std::list<std::string> word;
    std::list<std::pair<std::string, std::string>> pairs;

    wchar_t wide = 0;
    auto ptr = token.data();
    auto len = token.size();

    while (len) {
        auto status = std::mbtowc(&wide, ptr, len);
        assert(status > 0);
        word.emplace_back(ptr, ptr+status);
        ptr += status;
        len -= status;
    }

    word.back().append("</w>");

    auto&& get_pairs = [&](){
        pairs.clear();
        if (word.size() == 1)
            return;
        auto&& itr = word.begin();
        auto&& itr2 = word.begin();
        auto&& end = word.end();
        ++itr;

        do {
            pairs.emplace_back(*itr2, *itr);
            ++itr;
            ++itr2;
        } while (itr != end);
    };

    auto&& get_min_rank = [&]() {
        unsigned int ret = 0;
        std::pair<std::string, std::string>* min = nullptr;
        for (auto&& pair : pairs) {
            auto i = ranks.find(pair);
            if (i != ranks.end() && (min == nullptr || i->second < ret)) {
                ret = i->second;
                min = &pair;
            }
        }

        return min;
    };

    get_pairs();
    if (pairs.empty())
        return buff.push_back(tokens.at(token + "</w>"));

    std::list<std::string> new_word;
    while (true) {
        auto bigram = get_min_rank();
        if (bigram == nullptr)
            break;
        bool prev_first = false;
        for (auto&& w : word) {
            if (prev_first) {
                if (w == bigram->second)
                    new_word.emplace_back(bigram->first + bigram->second);
                else {
                    new_word.emplace_back(bigram->first);
                    new_word.emplace_back(std::move(w));
                }
                prev_first = false;
            } else if (w == bigram->first) {
                prev_first = true;
            } else {
                new_word.emplace_back(std::move(w));
                prev_first = false;
            }
        }

        std::swap(new_word, word);
        new_word.clear();
        if (word.size() == 1)
            break;
        get_pairs();
    }

    for (auto&& w : word) {
        buff.push_back(tokens.at(w));
        if (buff.size() >= max_len)
            return;
    }
}
