#include "tokenizer.h"
#include "utils.h"

#include <iostream>
#include <string>



int main() {
    libsdod::Tokenizer t("../../../../dlc/ctokenizer.txt");
    std::string s;
    while (std::cin) {
        std::getline(std::cin, s);
        auto&& ret = t.tokenize(s);
        std::cout << libsdod::format("{}", ret) << std::endl;
    }

    return 0;
}
