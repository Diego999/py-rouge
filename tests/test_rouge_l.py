from unittest import TestCase
from tests.utils import *
import os
import logging
logging.disable(logging.CRITICAL)


class RougeLTest(TestCase):
    def setUp(self):
        self.rouge_dir = os.path.abspath('ROUGE-1.5.5')
        self.N = 4
        self.metrics = ["rouge-l"]
        self.epsilon_ngrams_count_and_hits = 1e-5
        self.epsilon_avg_with_resampling = 3e-5 # We have to compare with a value higher than 1e-5 as the resampling might affect the precision of the true mean (especially with small truncation)

    def test_apply_avg(self):
        stemming = True
        alpha = 0.5
        limit_length = True
        length_limit_type = 'words'
        length_limit = 100

        apply_avg = True
        apply_best = False
        all_asserts = run_a_single_t_est_on_all_files_rouge_l(self.metrics, self.N, alpha, apply_avg, apply_best, length_limit, length_limit_type, limit_length, self.rouge_dir, stemming, self.epsilon_ngrams_count_and_hits, self.epsilon_avg_with_resampling)
        for assert_result, message in all_asserts:
            self.assertTrue(assert_result, message)

    def test_apply_best(self):
        stemming = True
        alpha = 0.5
        limit_length = True
        length_limit_type = 'words'
        length_limit = 100

        apply_avg = False
        apply_best = True
        all_asserts = run_a_single_t_est_on_all_files_rouge_l(self.metrics, self.N, alpha, apply_avg, apply_best, length_limit, length_limit_type, limit_length, self.rouge_dir, stemming, self.epsilon_ngrams_count_and_hits, self.epsilon_avg_with_resampling)
        for assert_result, message in all_asserts:
            self.assertTrue(assert_result, message)

    def test_truncate_words(self):
        apply_avg = True
        apply_best = False
        stemming = True
        alpha = 0.5
        length_limit_type = 'words'

        limit_length = True
        length_limit_type = length_limit_type
        length_limit = 0 # Unlimited
        all_asserts = run_a_single_t_est_on_all_files_rouge_l(self.metrics, self.N, alpha, apply_avg, apply_best, length_limit, length_limit_type, limit_length, self.rouge_dir, stemming, self.epsilon_ngrams_count_and_hits, self.epsilon_avg_with_resampling)
        for assert_result, message in all_asserts:
            self.assertTrue(assert_result, message)

        limit_length = True
        length_limit_type = length_limit_type
        length_limit = 10
        all_asserts = run_a_single_t_est_on_all_files_rouge_l(self.metrics, self.N, alpha, apply_avg, apply_best, length_limit, length_limit_type, limit_length, self.rouge_dir, stemming, self.epsilon_ngrams_count_and_hits, self.epsilon_avg_with_resampling)
        for assert_result, message in all_asserts:
            self.assertTrue(assert_result, message)

        limit_length = True
        length_limit_type = length_limit_type
        length_limit = 100
        all_asserts = run_a_single_t_est_on_all_files_rouge_l(self.metrics, self.N, alpha, apply_avg, apply_best, length_limit, length_limit_type, limit_length, self.rouge_dir, stemming, self.epsilon_ngrams_count_and_hits, self.epsilon_avg_with_resampling)
        for assert_result, message in all_asserts:
            self.assertTrue(assert_result, message)

        limit_length = True
        length_limit_type = length_limit_type
        length_limit = 300
        all_asserts = run_a_single_t_est_on_all_files_rouge_l(self.metrics, self.N, alpha, apply_avg, apply_best, length_limit, length_limit_type, limit_length, self.rouge_dir, stemming, self.epsilon_ngrams_count_and_hits, self.epsilon_avg_with_resampling)
        for assert_result, message in all_asserts:
            self.assertTrue(assert_result, message)

        limit_length = True
        length_limit_type = length_limit_type
        length_limit = 1000
        all_asserts = run_a_single_t_est_on_all_files_rouge_l(self.metrics, self.N, alpha, apply_avg, apply_best, length_limit, length_limit_type, limit_length, self.rouge_dir, stemming, self.epsilon_ngrams_count_and_hits, self.epsilon_avg_with_resampling)
        for assert_result, message in all_asserts:
            self.assertTrue(assert_result, message)

    def test_truncate_bytes(self):
        apply_avg = True
        apply_best = False
        stemming = True
        alpha = 0.5
        length_limit_type = 'bytes'

        limit_length = True
        length_limit_type = length_limit_type
        length_limit = 0 # Unlimited
        all_asserts = run_a_single_t_est_on_all_files_rouge_l(self.metrics, self.N, alpha, apply_avg, apply_best, length_limit, length_limit_type, limit_length, self.rouge_dir, stemming, self.epsilon_ngrams_count_and_hits, self.epsilon_avg_with_resampling)
        for assert_result, message in all_asserts:
            self.assertTrue(assert_result, message)

        limit_length = True
        length_limit_type = length_limit_type
        length_limit = 10
        all_asserts = run_a_single_t_est_on_all_files_rouge_l(self.metrics, self.N, alpha, apply_avg, apply_best, length_limit, length_limit_type, limit_length, self.rouge_dir, stemming, self.epsilon_ngrams_count_and_hits, self.epsilon_avg_with_resampling)
        for assert_result, message in all_asserts:
            self.assertTrue(assert_result, message)

        limit_length = True
        length_limit_type = length_limit_type
        length_limit = 50
        all_asserts = run_a_single_t_est_on_all_files_rouge_l(self.metrics, self.N, alpha, apply_avg, apply_best, length_limit, length_limit_type, limit_length, self.rouge_dir, stemming, self.epsilon_ngrams_count_and_hits, self.epsilon_avg_with_resampling)
        for assert_result, message in all_asserts:
            self.assertTrue(assert_result, message)

        limit_length = True
        length_limit_type = length_limit_type
        length_limit = 100
        all_asserts = run_a_single_t_est_on_all_files_rouge_l(self.metrics, self.N, alpha, apply_avg, apply_best, length_limit, length_limit_type, limit_length, self.rouge_dir, stemming, self.epsilon_ngrams_count_and_hits, self.epsilon_avg_with_resampling)
        for assert_result, message in all_asserts:
            self.assertTrue(assert_result, message)

        limit_length = True
        length_limit_type = length_limit_type
        length_limit = 300
        all_asserts = run_a_single_t_est_on_all_files_rouge_l(self.metrics, self.N, alpha, apply_avg, apply_best, length_limit, length_limit_type, limit_length, self.rouge_dir, stemming, self.epsilon_ngrams_count_and_hits, self.epsilon_avg_with_resampling)
        for assert_result, message in all_asserts:
            self.assertTrue(assert_result, message)

        limit_length = True
        length_limit_type = length_limit_type
        length_limit = 665
        all_asserts = run_a_single_t_est_on_all_files_rouge_l(self.metrics, self.N, alpha, apply_avg, apply_best, length_limit, length_limit_type, limit_length, self.rouge_dir, stemming, self.epsilon_ngrams_count_and_hits, self.epsilon_avg_with_resampling)
        for assert_result, message in all_asserts:
            self.assertTrue(assert_result, message)

        limit_length = True
        length_limit_type = length_limit_type
        length_limit = 1000
        all_asserts = run_a_single_t_est_on_all_files_rouge_l(self.metrics, self.N, alpha, apply_avg, apply_best, length_limit, length_limit_type, limit_length, self.rouge_dir, stemming, self.epsilon_ngrams_count_and_hits, self.epsilon_avg_with_resampling)
        for assert_result, message in all_asserts:
            self.assertTrue(assert_result, message)

        limit_length = True
        length_limit_type = length_limit_type
        length_limit = 10000
        all_asserts = run_a_single_t_est_on_all_files_rouge_l(self.metrics, self.N, alpha, apply_avg, apply_best, length_limit, length_limit_type, limit_length, self.rouge_dir, stemming, self.epsilon_ngrams_count_and_hits, self.epsilon_avg_with_resampling)
        for assert_result, message in all_asserts:
            self.assertTrue(assert_result, message)

    def test_stemming(self):
        alpha = 0.5
        limit_length = True
        length_limit_type = 'words'
        length_limit = 100
        apply_avg = True
        apply_best = False

        stemming = True
        all_asserts = run_a_single_t_est_on_all_files_rouge_l(self.metrics, self.N, alpha, apply_avg, apply_best, length_limit, length_limit_type, limit_length, self.rouge_dir, stemming, self.epsilon_ngrams_count_and_hits, self.epsilon_avg_with_resampling)
        for assert_result, message in all_asserts:
            self.assertTrue(assert_result, message)

        stemming = False
        all_asserts = run_a_single_t_est_on_all_files_rouge_l(self.metrics, self.N, alpha, apply_avg, apply_best, length_limit, length_limit_type, limit_length, self.rouge_dir, stemming, self.epsilon_ngrams_count_and_hits, self.epsilon_avg_with_resampling)
        for assert_result, message in all_asserts:
            self.assertTrue(assert_result, message)

    def test_alpha(self):
        stemming = True
        limit_length = True
        length_limit_type = 'words'
        length_limit = 100
        apply_avg = True
        apply_best = False

        for alpha in [0.0, 0.4, 0.8, 1.0]:
            all_asserts = run_a_single_t_est_on_all_files_rouge_l(self.metrics, self.N, alpha, apply_avg, apply_best, length_limit, length_limit_type, limit_length, self.rouge_dir, stemming, self.epsilon_ngrams_count_and_hits, self.epsilon_avg_with_resampling)
            for assert_result, message in all_asserts:
                self.assertTrue(assert_result, message)
