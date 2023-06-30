#ifndef LIBSDOD_TOKENIZER_H
#define LIBSDOD_TOKENIZER_H

#include <string>
#include <vector>
#include <unordered_map>


namespace libsdod {

class Tokenizer {
public:
    typedef std::u8string input_str_type; // expects utf-8 encoded str as input
    typedef std::u8string internal_str_type; // uses utf-8 encoded str internally
    typedef uint16_t token_type;

public:
    Tokenizer(std::string const& bpe_file);

    std::vector<token_type> tokenize(input_str_type const& str, unsigned int context_len = 77) const;

private:
    std::unordered_map<internal_str_type,token_type> tokens;
    std::unordered_map<std::pair<internal_str_type, internal_str_type>,token_type> ranks;

    token_type start_token;
    token_type end_token;

    void bpe(std::vector<token_type>& buff, internal_str_type token) const;
};

}

#endif //LIBSDOD_TOKENIZER_H
