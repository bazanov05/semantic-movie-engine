#include <string>
#include <unordered_set>
#include <algorithm>
#include <cctype>
#include <sstream>


/**
 * @brief Cleans raw text by removing punctuation, lowercasing, and filtering stopwords.
 *
 * Designed to normalize film overviews before they are passed into the embedding model.
 * Reducing noise from punctuation and semantically empty words produces tighter,
 * more meaningful vector representations.
 */
class TextNormalizer{
    private:
    std::unordered_set<std::string> stop_words;

    public:
    /**
     * @brief Constructs a TextNormalizer with a custom stopword list.
     *
     * @param stop_words An unordered set of lowercase words to filter out during cleaning.
     *                   Words like "the", "a", "and" carry no semantic meaning for embeddings.
     */
    TextNormalizer(const std::unordered_set<std::string>& stop_words);

    /**
     * @brief Cleans a raw text string for embedding ingestion.
     *
     * Performs three sequential operations:
     * 1. Strips all punctuation characters from the input.
     * 2. Lowercases each word.
     * 3. Removes words found in the stopword set.
     *
     * @param dirty_text The raw input string to clean.
     * @return A cleaned string containing only lowercase, non-stopword tokens separated by spaces.
     */
    std::string clean(std::string& dirty_text) const;
}; 