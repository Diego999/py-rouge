import nltk
import os
import re
import itertools
import collections
import pkg_resources
import sys


class Rouge:
    DEFAULT_METRICS = ["rouge-n"]
    DEFAULT_N = 1
    STATS = ["f", "p", "r"]
    AVAILABLE_METRICS = {"rouge-n"}
    AVAILABLE_LENGTH_LIMIT_TYPES = {'words', 'bytes'}
    REMOVE_CHAR_PATTERN = re.compile('[^A-Za-z0-9]')

    # Hack to not tokenize "cannot" to "can not" and consider them different as in the official ROUGE script
    KEEP_CANNOT_IN_ONE_WORD = re.compile('cannot')
    KEEP_CANNOT_IN_ONE_WORD_REVERSED = re.compile('_cannot_')

    WORDNET_KEY_VALUE = {}
    WORDNET_DB_FILEPATH = 'wordnet_key_value.txt'
    WORDNET_DB_FILEPATH_SPECIAL_CASE = 'wordnet_key_value_special_cases.txt'
    WORDNET_DB_DELIMITER = '|'
    STEMMER = None

    def __init__(self, metrics=None, max_n=None, limit_length=True, length_limit=665, length_limit_type='bytes', apply_avg=True, apply_best=False, stemming=True, alpha=0.5, ensure_compatibility=True):
        """
        Handle the ROUGE score computation as in the official perl script.

        Note 1: Small differences might happen if the resampling of the perl script is not high enough (as the average depends on this).
        Note 2: Stemming of the official Porter Stemmer of the ROUGE perl script is slightly different and the Porter one implemented in NLTK. However, special cases of DUC 2004 have been traited.
                The solution would be to rewrite the whole perl stemming in python from the original script

        Args:
          metrics: What ROUGE score to compute. Available: ROUGE-N. Default: ROUGE-N
          max_n: N-grams for ROUGE-N if specify. Default:1
          limit_length: If the summaries must be truncated. Defaut:True
          length_limit: Number of the truncation where the unit is express int length_limit_Type. Default:665 (bytes) (0: unlimited)
          length_limit_type: Unit of length_limit. Available: words, bytes. Default: 'bytes'
          apply_avg: If we should average the score of multiple samples. Default: True. If apply_Avg & apply_best = False, then each ROUGE scores are independant
          apply_best: Take the best instead of the average. Default: False, then each ROUGE scores are independant
          steemming: Apply stemming to summaries. Default: True
          alpha: Alpha use to compute f1 score: P*R/((1-a)*P + a*R). Default:0.5
          ensure_compatibility: Use same stemmer and special "hacks" to product same results as in the official perl script (besides the number of sampling if not high enough)

        Raises:
          ValueError: raises exception if metric is not among AVAILABLE_METRICS
          ValueError: raises exception if length_limit_type is not among AVAILABLE_LENGTH_LIMIT_TYPES
        """
        self.metrics = metrics if metrics is not None else Rouge.DEFAULT_METRICS
        for m in self.metrics:
            if m not in Rouge.AVAILABLE_METRICS:
                raise ValueError("Unknown metric '{}'".format(m))

        self.max_n = max_n if "rouge-n" in self.metrics else None
        # Add all rouge-n metrics
        if self.max_n is not None:
            index_rouge_n = self.metrics.index('rouge-n')
            self.metrics[index_rouge_n] = ['rouge-{}'.format(n) for n in range(1, self.max_n + 1)]
            self.metrics = sum(self.metrics, [])

        self.limit_length = limit_length
        if self.limit_length:
            if length_limit_type not in Rouge.AVAILABLE_LENGTH_LIMIT_TYPES:
                raise ValueError("Unknown length_limit_type '{}'".format(length_limit_type))

        self.length_limit = length_limit
        if self.length_limit == 0:
            self.length_limit = sys.maxsize
        self.length_limit_type = length_limit_type
        self.stemming = stemming

        self.apply_avg = apply_avg
        self.apply_best = apply_best
        self.alpha = alpha
        self.ensure_compatibility = ensure_compatibility

        # Load static objects
        if len(Rouge.WORDNET_KEY_VALUE) == 0:
            Rouge.load_wordnet_db(ensure_compatibility)
        if Rouge.STEMMER is None:
            Rouge.load_stemmer(ensure_compatibility)

    @staticmethod
    def load_stemmer(ensure_compatibility):
        """
        Load the stemmer that is going to be used if stemming is enabled
        Args
            ensure_compatibility: Use same stemmer and special "hacks" to product same results as in the official perl script (besides the number of sampling if not high enough)
        """
        Rouge.STEMMER = nltk.stem.porter.PorterStemmer('ORIGINAL_ALGORITHM') if ensure_compatibility else nltk.stem.porter.PorterStemmer()

    @staticmethod
    def load_wordnet_db(ensure_compatibility):
        """
        Load WordNet database to apply specific rules instead of stemming + load file for special cases to ensure kind of compatibility (at list with DUC 2004) with the original stemmer used in the Perl script
        Args
            ensure_compatibility: Use same stemmer and special "hacks" to product same results as in the official perl script (besides the number of sampling if not high enough)

        Raises:
            FileNotFoundError: If one of both databases is not found
        """
        files_to_load = [Rouge.WORDNET_DB_FILEPATH]
        if ensure_compatibility:
            files_to_load.append(Rouge.WORDNET_DB_FILEPATH_SPECIAL_CASE)

        for wordnet_db in files_to_load:
            filepath = pkg_resources.resource_filename(__name__, wordnet_db)
            if not os.path.exists(filepath):
                raise FileNotFoundError("The file '{}' does not exist".format(filepath))

            with open(filepath, 'r', encoding='utf-8') as fp:
                for line in fp:
                    k, v = line.strip().split(Rouge.WORDNET_DB_DELIMITER)
                    assert k not in Rouge.WORDNET_KEY_VALUE
                    Rouge.WORDNET_KEY_VALUE[k] = v

    @staticmethod
    def tokenize_text(text, language='english'):
        """
        Tokenize text in the specific language

        Args:
          text: The string text to tokenize
          language: Language of the text

        Returns:
          List of tokens of text
        """
        return nltk.word_tokenize(text, language)

    @staticmethod
    def stem_tokens(tokens):
        """
        Apply WordNetDB rules or Stem each token of tokens

        Args:
          tokens: List of tokens to apply WordNetDB rules or to stem

        Returns:
          List of final stems
        """
        # Stemming & Wordnet apply only if token has at least 3 chars
        for i, token in enumerate(tokens):
            if len(token) > 0:
                if len(token) > 3:
                    if token in Rouge.WORDNET_KEY_VALUE:
                        token = Rouge.WORDNET_KEY_VALUE[token]
                    else:
                        token = Rouge.STEMMER.stem(token)
                    tokens[i] = token

        return tokens


    @staticmethod
    def _get_ngrams(n, text):
        """
        Calcualtes n-grams.

        Args:
          n: which n-grams to calculate
          text: An array of tokens

        Returns:
          A set of n-grams with their number of occurences
        """
        # Modified from https://github.com/pltrdy/seq2seq/blob/master/seq2seq/metrics/rouge.py
        ngram_set = collections.defaultdict(int)
        max_index_ngram_start = len(text) - n
        for i in range(max_index_ngram_start + 1):
            ngram_set[tuple(text[i:i + n])] += 1
        return ngram_set

    @staticmethod
    def _split_into_words(sentences):
        """
        Splits multiple sentences into words and flattens the result

        Args:
          sentences: list of string

        Returns:
          A list of words (split by white space)
        """
        # Modified from https://github.com/pltrdy/seq2seq/blob/master/seq2seq/metrics/rouge.py
        return list(itertools.chain(*[_.split() for _ in sentences]))

    @staticmethod
    def _get_word_ngrams_and_length(n, sentences):
        """
        Calculates word n-grams for multiple sentences.

        Args:
          n: wich n-grams to calculate
          sentences: list of string

        Returns:
          A set of n-grams and #n-grams in sentences
        """
        # Modified from https://github.com/pltrdy/seq2seq/blob/master/seq2seq/metrics/rouge.py
        assert len(sentences) > 0
        assert n > 0

        words = Rouge._split_into_words(sentences)
        return Rouge._get_ngrams(n, words), len(words) - (n - 1)

    @staticmethod
    def _compute_p_r_f_score(evaluated_count, reference_count, overlapping_count, alpha=0.5):
        """
        Compute precision, recall and f1_score (with alpha: P*R / ((1-alpha)*P + alpha*R))

        Args:
          evaluated_count: #n-grams in the hypothesis
          reference_count: #n-grams in the reference
          overlapping_count: #n-grams in common between hypothesis and reference
          alpha: Value to use for the F1 score (default: 0.5)

        Returns:
          A dict with 'p', 'r' and 'f' as keys fore precision, recall, f1 score
        """
        precision = 0.0 if evaluated_count == 0 else overlapping_count / evaluated_count
        recall = 0.0 if reference_count == 0 else overlapping_count / reference_count
        f1_score = 0.0 if (recall == 0.0 or precision == 0.0) else precision * recall / ((1 - alpha) * precision + alpha * recall)
        return {"f": f1_score, "p": precision, "r": recall}

    @staticmethod
    def _compute_ngrams(evaluated_sentences, reference_sentences, n):
        """
        Computes n-grams overlap of two text collections of sentences.
        Sourece: http://research.microsoft.com/en-us/um/people/cyl/download/
        papers/rouge-working-note-v1.3.1.pdf

        Args:
          evaluated_sentences: The sentences that have been picked by the
                               summarizer
          reference_sentences: The sentences from the referene set
          n: Size of ngram

        Returns:
          Number of n-grams for evaluated_sentences, reference_sentences and intersection of both.
          intersection of both count multiple of occurences in n-grams match several times

        Raises:
          ValueError: raises exception if a param has len <= 0
        """
        # Modified from https://github.com/pltrdy/seq2seq/blob/master/seq2seq/metrics/rouge.py
        if len(evaluated_sentences) <= 0 or len(reference_sentences) <= 0:
            raise ValueError("Collections must contain at least 1 sentence.")

        evaluated_ngrams, evaluated_count = Rouge._get_word_ngrams_and_length(n, evaluated_sentences)
        reference_ngrams, reference_count = Rouge._get_word_ngrams_and_length(n, reference_sentences)

        # Gets the overlapping ngrams between evaluated and reference
        overlapping_ngrams = set(evaluated_ngrams.keys()).intersection(set(reference_ngrams.keys()))
        overlapping_count = 0
        for ngram in overlapping_ngrams:
            overlapping_count += min(evaluated_ngrams[ngram], reference_ngrams[ngram])

        return evaluated_count, reference_count, overlapping_count

    def get_scores(self, hypothesis, references):
        """
        Compute precision, recall and f1 score between hypothesis and references

        Args:
          hypothesis: hypothesis summary, string
          references: reference summary/ies, either string or list of strings (if multiple)

        Returns:
          Return precision, recall and f1 score between hypothesis and references

        Raises:
          ValueError: raises exception if a type of hypothesis is different than the one of reference
          ValueError: raises exception if a len of hypothesis is different than the one of reference
        """
        if isinstance(hypothesis, str):
            hypothesis, references = [hypothesis], [references]

        if type(hypothesis) != type(references):
            raise ValueError("'hyps' and 'refs' are not of the same type")

        if len(hypothesis) != len(references):
            raise ValueError("'hyps' and 'refs' do not have the same length")

        return self._get_scores(hypothesis, references)

    def _get_scores(self, all_hypothesis, all_references):
        """
        Computes precision, recall and f1 score between all hypothesis and references

        Args:
          hypothesis: hypothesis summary, string
          references: reference summary/ies, either string or list of strings (if multiple)

        Returns:
          Return precision, recall and f1 score between all hypothesis and references
        """
        if self.apply_avg:
            scores = {metric: {stat:0.0 for stat in Rouge.STATS} for metric in self.metrics}
        elif self.apply_best:
            scores = {metric: {stat: [] for stat in Rouge.STATS} for metric in self.metrics}
        else:
            scores = {metric: [{stat:[] for stat in Rouge.STATS} for _ in range(len(all_hypothesis))] for metric in self.metrics}

        for sample_id, (hypothesis, references) in enumerate(zip(all_hypothesis, all_references)):
            assert isinstance(hypothesis, str)
            has_multiple_references = False
            if isinstance(references, list):
                has_multiple_references = len(references) > 1
                if not has_multiple_references:
                    references = references[0]

            # Prepare hypothesis and reference(s)
            hypothesis = self._preprocess_summary(hypothesis)
            references = [self._preprocess_summary(reference) for reference in references] if has_multiple_references else [self._preprocess_summary(references)]

            # Compute scores
            for metric in self.metrics:
                suffix = metric.split('-')[-1]
                n = -1
                if suffix.isdigit():
                    n = int(suffix)

                if n > 0:
                    total_hypothesis_ngrams_count = 0
                    total_reference_ngrams_count = 0
                    total_ngrams_overlapping_count = 0

                    # Aggregate
                    if self.apply_avg:
                        # AVG
                        for reference in references:
                            hypothesis_count, reference_count, overlapping_ngrams = Rouge._compute_ngrams(hypothesis, reference, n)
                            total_hypothesis_ngrams_count += hypothesis_count
                            total_reference_ngrams_count += reference_count
                            total_ngrams_overlapping_count += overlapping_ngrams

                        score = Rouge._compute_p_r_f_score(total_hypothesis_ngrams_count, total_reference_ngrams_count, total_ngrams_overlapping_count, self.alpha)

                        for stat in Rouge.STATS:
                            scores[metric][stat] += score[stat]
                    else:
                        # Best sample
                        if self.apply_best:
                            best_current_score = None
                            for reference in references:
                                hypothesis_count, reference_count, overlapping_ngrams = Rouge._compute_ngrams(hypothesis, reference, n)
                                score = Rouge._compute_p_r_f_score(hypothesis_count, reference_count, overlapping_ngrams, self.alpha)
                                if best_current_score is None or score['r'] > best_current_score['r']:
                                    best_current_score = score
                            for stat in Rouge.STATS:
                                scores[metric][stat].append(best_current_score[stat])
                        else: # Keep all
                            for reference in references:
                                hypothesis_count, reference_count, overlapping_ngrams = Rouge._compute_ngrams(hypothesis, reference, n)
                                score = Rouge._compute_p_r_f_score(hypothesis_count, reference_count, overlapping_ngrams, self.alpha)
                                for stat in Rouge.STATS:
                                    scores[metric][sample_id][stat].append(score)

        # Compute final score with the average or the the max
        if self.apply_avg and len(all_hypothesis) > 1:
            for metric in self.metrics:
                for stat in Rouge.STATS:
                    scores[metric][stat] /= len(all_hypothesis)
        elif self.apply_best: # Has to take the best among a list, even if the list has only one element
            for metric in self.metrics:
                idx_best = list(sorted([(i, val) for i, val in enumerate(scores[metric]['r'])], key=lambda x:x[1], reverse=True))[0][0]
                for stat in Rouge.STATS:
                    scores[metric][stat] = scores[metric][stat][idx_best]

        return scores

    def _preprocess_summary(self, summary):
        """
        Preprocess (truncate text if enable, tokenization, stemming if enable, lowering) of a summary

        Args:
          summary: string of the summary

        Returns:
          Return the preprocessed summary (string)
        """
        # Truncate
        if self.limit_length:
            all_tokens = summary.split() # Counting as in the perls script
            if self.length_limit_type == 'words':
                summary = ' '.join(all_tokens[:self.length_limit])
            elif self.length_limit_type == 'bytes':
                summary = summary[:self.length_limit]

        # Preprocess. Hack: because official ROUGE script bring "cannot" as "cannot" and "can not" as "can not", we have to hack nltk tokenizer to not transform "cannot/can not" to "can not"
        if self.ensure_compatibility:
            tokens = self.tokenize_text(Rouge.KEEP_CANNOT_IN_ONE_WORD.sub('_cannot_', Rouge.REMOVE_CHAR_PATTERN.sub(' ', summary.lower().strip())))
            stems = self.stem_tokens(tokens) # stemming in-place
            preprocessed_summary = [Rouge.KEEP_CANNOT_IN_ONE_WORD_REVERSED.sub('cannot', ' '.join(stems))]
        else:
            tokens = self.tokenize_text(Rouge.REMOVE_CHAR_PATTERN.sub(' ', summary.lower().strip()))
            stems = self.stem_tokens(tokens) # stemming in-place
            preprocessed_summary = [' '.join(stems)]

        return preprocessed_summary