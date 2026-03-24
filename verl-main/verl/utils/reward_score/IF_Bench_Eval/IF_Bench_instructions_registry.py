# Copyright 2025 Allen Institute for AI.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Registry of all IF_Bench_instructions."""

from . import IF_Bench_instructions


INSTRUCTION_DICT = {
    "count:word_count_range": IF_Bench_instructions.WordCountRangeChecker,
    "count:unique_word_count" : IF_Bench_instructions.UniqueWordCountChecker,
    "ratio:stop_words" : IF_Bench_instructions.StopWordPercentageChecker,
    "ratio:sentence_type" : IF_Bench_instructions.SentTypeRatioChecker,
    "ratio:sentence_balance" : IF_Bench_instructions.SentBalanceChecker,
    "count:conjunctions" : IF_Bench_instructions.ConjunctionCountChecker,
    "count:person_names" : IF_Bench_instructions.PersonNameCountChecker,
    "ratio:overlap" : IF_Bench_instructions.NGramOverlapChecker,
    "count:numbers" : IF_Bench_instructions.NumbersCountChecker,
    "words:alphabet" : IF_Bench_instructions.AlphabetLoopChecker,
    "words:vowel" : IF_Bench_instructions.SingleVowelParagraphChecker,
    "words:consonants" : IF_Bench_instructions.ConsonantClusterChecker,
    "sentence:alliteration_increment" : IF_Bench_instructions.IncrementingAlliterationChecker,
    "words:palindrome" : IF_Bench_instructions.PalindromeChecker,
    "count:punctuation" : IF_Bench_instructions.PunctuationCoverChecker,
    "format:parentheses" : IF_Bench_instructions.NestedParenthesesChecker,
    "format:quotes" : IF_Bench_instructions.NestedQuotesChecker,
    "words:prime_lengths" : IF_Bench_instructions.PrimeLengthsChecker,
    "format:options" : IF_Bench_instructions.OptionsResponseChecker,
    "format:newline" : IF_Bench_instructions.NewLineWordsChecker,
    "format:emoji" : IF_Bench_instructions.EmojiSentenceChecker,
    "ratio:sentence_words" : IF_Bench_instructions.CharacterCountUniqueWordsChecker,
    "count:words_japanese" : IF_Bench_instructions.NthWordJapaneseChecker,
    "words:start_verb" : IF_Bench_instructions.StartWithVerbChecker,
    "words:repeats" : IF_Bench_instructions.LimitedWordRepeatChecker,
    "sentence:keyword" : IF_Bench_instructions.IncludeKeywordChecker,
    "count:pronouns" : IF_Bench_instructions.PronounCountChecker,
    "words:odd_even_syllables" : IF_Bench_instructions.AlternateParitySyllablesChecker,
    "words:last_first" : IF_Bench_instructions.LastWordFirstNextChecker,
    "words:paragraph_last_first" : IF_Bench_instructions.ParagraphLastFirstWordMatchChecker,
    "sentence:increment" : IF_Bench_instructions.IncrementingWordCountChecker,
    "words:no_consecutive" : IF_Bench_instructions.NoConsecutiveFirstLetterChecker,
    "format:line_indent" : IF_Bench_instructions.IndentStairsChecker,
    "format:quote_unquote" : IF_Bench_instructions.QuoteExplanationChecker,
    "format:list" : IF_Bench_instructions.SpecialBulletPointsChecker,
    "format:thesis" : IF_Bench_instructions.ItalicsThesisChecker,
    "format:sub-bullets" : IF_Bench_instructions.SubBulletPointsChecker,
    "format:no_bullets_bullets" : IF_Bench_instructions.SomeBulletPointsChecker,
    "custom:multiples" : IF_Bench_instructions.PrintMultiplesChecker,
    "custom:mcq_count_length": IF_Bench_instructions.MultipleChoiceQuestionsChecker,
    "custom:reverse_newline": IF_Bench_instructions.ReverseNewlineChecker,
    "custom:word_reverse": IF_Bench_instructions.WordReverseOrderChecker,
    "custom:character_reverse": IF_Bench_instructions.CharacterReverseOrderChecker,
    "custom:sentence_alphabet": IF_Bench_instructions.SentenceAlphabetChecker,
    "custom:european_capitals_sort": IF_Bench_instructions.EuropeanCapitalsSortChecker,
    "custom:csv_city": IF_Bench_instructions.CityCSVChecker,
    "custom:csv_special_character": IF_Bench_instructions.SpecialCharacterCSVChecker,
    "custom:csv_quotes": IF_Bench_instructions.QuotesCSVChecker,
    "custom:date_format_list": IF_Bench_instructions.DateFormatListChecker,
    "count:keywords_multiple" : IF_Bench_instructions.KeywordsMultipleChecker,
    "words:keywords_specific_position" : IF_Bench_instructions.KeywordSpecificPositionChecker,
    "words:words_position" : IF_Bench_instructions.WordsPositionChecker,
    "repeat:repeat_change" : IF_Bench_instructions.RepeatChangeChecker,
    "repeat:repeat_simple" : IF_Bench_instructions.RepeatSimpleChecker,
    "repeat:repeat_span" : IF_Bench_instructions.RepeatSpanChecker,
    "format:title_case" : IF_Bench_instructions.TitleCaseChecker,
    "format:output_template" : IF_Bench_instructions.OutputTemplateChecker,
    "format:no_whitespace" : IF_Bench_instructions.NoWhitespaceChecker,
}
