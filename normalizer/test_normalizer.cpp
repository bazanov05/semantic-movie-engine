#include <gtest/gtest.h>
#include "normalizer.hpp"
#include <unordered_set>
#include <string>


TEST(TextNormalizerTest, HandlesBasicTextNoPunctuation) {
    std::unordered_set<std::string> stop_words = {"the", "is"};
    TextNormalizer normalizer(stop_words);
    
    std::string input = "apple orange banana";
    std::string result = normalizer.clean(input);
    
    EXPECT_EQ(result, "apple orange banana");
}

TEST(TextNormalizerTest, RemovesPunctuationAndLowercases) {
    std::unordered_set<std::string> stop_words = {};
    TextNormalizer normalizer(stop_words);
    
    // Testing capitals, commas, and exclamation marks
    std::string input = "Hello, World! THIS is C++.";
    std::string result = normalizer.clean(input);
    
    // C++ gets stripped to "c" because '+' is punctuation
    EXPECT_EQ(result, "hello world this is c");
}

TEST(TextNormalizerTest, FiltersStopWords) {
    std::unordered_set<std::string> stop_words = {"a", "the", "and", "in"};
    TextNormalizer normalizer(stop_words);
    
    std::string input = "The cat and a dog are in the yard";
    std::string result = normalizer.clean(input);
    
    EXPECT_EQ(result, "cat dog are yard");
}

TEST(TextNormalizerTest, CombinedBaseCase) {
    std::unordered_set<std::string> stop_words = {"this", "is", "a"};
    TextNormalizer normalizer(stop_words);
    
    std::string input = "This is a GREAT, fantastic test!";
    std::string result = normalizer.clean(input);
    
    EXPECT_EQ(result, "great fantastic test");
}

TEST(TextNormalizerTest, HandlesEmptyString) {
    std::unordered_set<std::string> stop_words = {"the"};
    TextNormalizer normalizer(stop_words);
    
    std::string input = "";
    std::string result = normalizer.clean(input);
    
    EXPECT_EQ(result, "");
}

TEST(TextNormalizerTest, HandlesStringWithOnlyStopWords) {
    std::unordered_set<std::string> stop_words = {"this", "is", "a", "test"};
    TextNormalizer normalizer(stop_words);
    
    std::string input = "This is a test";
    std::string result = normalizer.clean(input);
    
    EXPECT_EQ(result, "");
}

TEST(TextNormalizerTest, HandlesStringWithOnlyPunctuation) {
    std::unordered_set<std::string> stop_words = {"the"};
    TextNormalizer normalizer(stop_words);
    
    std::string input = "!@# $%^ &*()";
    std::string result = normalizer.clean(input);
    
    EXPECT_EQ(result, "");
}

TEST(TextNormalizerTest, HandlesMultipleConsecutiveSpaces) {
    std::unordered_set<std::string> stop_words = {"is"};
    TextNormalizer normalizer(stop_words);
    
    // String stream  naturally handles and collapses multiple spaces
    std::string input = "Here    is   some   spaced    text";
    std::string result = normalizer.clean(input);
    
    EXPECT_EQ(result, "here some spaced text");
}

TEST(TextNormalizerTest, HandlesStopWordsWithPunctuationAttached) {
    std::unordered_set<std::string> stop_words = {"stop"};
    TextNormalizer normalizer(stop_words);
    
    // The normalizer strips punctuation first, so "Stop!" becomes "stop",
    // which should then be caught by the stop_words filter.
    std::string input = "Please Stop! Do not pass.";
    std::string result = normalizer.clean(input);
    
    EXPECT_EQ(result, "please do not pass");
}