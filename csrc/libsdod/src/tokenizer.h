#ifndef LIBSDOD_TOKENIZER_H
#define LIBSDOD_TOKENIZER_H

#include <string>
#include <vector>
#include <unordered_map>


namespace libsdod {

struct MergeHash {
    std::size_t operator()(std::pair<std::string, std::string> const& p) const {
        return std::hash<std::string>()(p.first + p.second);
    }
};

class Tokenizer {
public:
    typedef uint16_t token_type;

public:
    Tokenizer(std::string const& bpe_file);

    void tokenize(std::vector<token_type>& out, std::string const& str, unsigned int context_len = 77) const;

    std::vector<token_type> tokenize(std::string const& str, unsigned int context_len = 77) const {
        std::vector<token_type> ret;
        ret.reserve(context_len);
        tokenize(ret, str, context_len);
        return ret;
    }

private:
    std::unordered_map<std::string, token_type> tokens;
    std::unordered_map<std::pair<std::string, std::string>, unsigned int, MergeHash> ranks;

    token_type start_token;
    token_type end_token;

    void bpe(std::vector<token_type>& buff, std::string token, unsigned int max_len) const;
};




}

#endif //LIBSDOD_TOKENIZER_H
