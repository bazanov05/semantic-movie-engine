#include "normalizer.hpp"


TextNormalizer::TextNormalizer(const std::unordered_set<std::string>& stop_words): stop_words(stop_words){}


std::string TextNormalizer::clean(std::string& dirty_text) const{
    // remove all punctuation marks from dirty text
    // return an iterator to where the "deleted" elements start after rearranging
    auto new_end = std::remove_if(dirty_text.begin(), dirty_text.end(), 
                    [](unsigned char ch) -> bool {
                        if(ch < 128){
                            return std::ispunct(ch) != 0;
                        }
                        return false;   // do not delete non-ASCII bytes
                    }
    );

    // delete punctuation marks based on iterator where they start 
    dirty_text.erase(new_end, dirty_text.end());

    // create a stream from our dirty text to pull words from it based on the spaces 
    std::stringstream stream(dirty_text);

    std::string word;
    std::string clean_text = "";

    while(stream >> word){
        // make every word consist of lower letters only 
        std::transform(word.begin(), word.end(), word.begin(), 
                        [](unsigned char ch) -> unsigned char{
                            if(ch < 128){
                                return std::tolower(ch);
                            }
                            return ch;
                        }
        );
        // if this word is not a stop word - add it to clean text 
        if(this -> stop_words.find(word) == this -> stop_words.end()){
            clean_text += word;
            clean_text += " ";
        }
    }

    // safely delete last space after the last word
    if (!clean_text.empty()) {
        clean_text.pop_back();
    }

    return clean_text;
}