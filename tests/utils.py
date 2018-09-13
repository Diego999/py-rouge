import glob
import os
import shutil
import time
import pyrouge
import regex
import collections
import rouge


def parse_rouge_for_results(output):
    """
    Read the ROUGE scores from the official scripts
    """
    # ROUGE-N
    pattern = regex.compile('^..ROUGE-(\d|L|(W-\d\.\d*))\s+Average_(R|P|F):')
    parsed_output = [line for line in output.strip().split('\n') if pattern.match(line)]

    results = {}
    for line in parsed_output:
        split_output = line.split()
        assert len(split_output) == 8 # Average
        rouge_test = split_output[1]
        if rouge_test.startswith('ROUGE-W'):
            rouge_test = 'ROUGE-W'
        if rouge_test not in results:
            results[rouge_test] = {}

        _, _, avg, val, _, conf95_low, _, conf95_high = split_output
        avg = avg[-2]
        val = float(val)
        conf95_low = float(conf95_low)
        conf95_high = float(conf95_high[:-1])
        results[rouge_test][avg] = {'val': val, 'conf_95_low': conf95_low, 'conf_95_high': conf95_high}

    return results


def parse_rouge_for_ngrams_counts(output):
    """
    Read the number of n-grams for peer and model summaries + overlap.
    """
    # Try for ROUGE-N
    pattern_ngrams = regex.compile('^total\s+\d-gram')
    pattern_score = regex.compile('^total\sROUGE-\d-(R|P|F):')
    parsed_output = [line for line in output.strip().split('\n') if pattern_ngrams.match(line) or pattern_score.match(line)]

    if len(parsed_output) > 1: # ROUGE-N
        results = []
        for i, line in enumerate(parsed_output):
            if i % 6 == 0: # 6 entries per summary
                results.append(collections.defaultdict(dict))
            split_output = line.replace(':', '').split()
            if len(split_output) == 5 or len(split_output) == 4: # ngram
                key, type_, val = split_output[1], split_output[2], split_output[-1]
                results[-1][key][type_] = val
            elif len(split_output) == 3: # rouge
                key, val = split_output[1], split_output[-1]
                results[-1][key] = val
    else:
        # ROUGE-L
        pattern_ngrams = regex.compile('^total\s+ROUGE-L\s+')
        pattern_score = regex.compile('^total\s+ROUGE-L-(R|P|F)')
        parsed_output = [line for line in output.strip().split('\n') if pattern_ngrams.match(line) or pattern_score.match(line)]
        metric = 'ROUGE-L'
        metric_len = len(metric)

        if len(parsed_output) > 1:
            results = []
            for i, line in enumerate(parsed_output):
                if i % 6 == 0:  # 6 entries per summary
                    results.append(collections.defaultdict(dict))
                split_output = line.replace(':', '').split()
                key = split_output[1]
                if key == metric: # count ngrams
                    type_, val = split_output[2], split_output[-1]
                    results[-1][key][type_] = val
                elif len(key) == metric_len + 2: # ROUGE-L-R
                    key, val = split_output[1], split_output[-1]
                    key = key[key.rfind('-')+1:]
                    results[-1][key] = val
        else:
            # ROUGE-W
            pattern_ngrams = regex.compile('^total\s+ROUGE-W-\d\.\d*\s+')
            pattern_score = regex.compile('^total\sROUGE-(\d|(W-\d\.\d*))-(R|P|F)')
            pattern_metric = regex.compile('ROUGE-W-\d\.\d*')
            parsed_output = [regex.sub(pattern_metric, 'ROUGE-W', line) for line in output.strip().split('\n') if pattern_ngrams.match(line) or pattern_score.match(line)]

            metric = 'ROUGE-W'
            metric_len = len(metric)

            results = []
            for i, line in enumerate(parsed_output):
                if i % 6 == 0:  # 6 entries per summary
                    results.append(collections.defaultdict(dict))
                split_output = line.replace(':', '').split()
                key = split_output[1]
                if key == metric:  # count ngrams
                    type_, val = split_output[2], split_output[-1]
                    results[-1][key][type_] = val
                elif len(key) == metric_len + 2:  # ROUGE-W-R
                    key, val = split_output[1], split_output[-1]
                    key = key[key.rfind('-') + 1:]
                    results[-1][key] = val

    return results


def setup_rouge_python(metrics, N, stemming, apply_avg, apply_best, alpha, limit_length, length_type, length_limit, weight_factor):
    return rouge.Rouge(metrics=metrics,
                       max_n=N,
                       limit_length=limit_length,
                       length_limit=length_limit,
                       length_limit_type=length_type,
                       apply_avg=apply_avg,
                       apply_best=apply_best,
                       alpha=alpha,
                       weight_factor=weight_factor,
                       stemming=stemming)


def setup_rouge_perl(rouge_dir, rouge_args, model_dir, system_dir):
    r = pyrouge.Rouge155(rouge_dir=rouge_dir, rouge_args=rouge_args, log_level=10)
    r.model_dir = model_dir
    r.system_dir = system_dir
    r.system_filename_pattern = "(\\d+).txt"
    r.model_filename_pattern = "#ID#.[A-Z].txt"
    return r


def get_peers_models(system_dir, model_dir):
    hyps = []
    for file in sorted([f for f in glob.glob('{}/*'.format(system_dir))]):
        with open(file, 'r', encoding='utf-8') as fp:
            hyps.append('\n'.join([line.strip() for line in fp]))

    refs = []
    for file in sorted([f for f in glob.glob('{}/*'.format(model_dir))]):
        filename = os.path.basename(file)
        key = filename[:-5] if filename.endswith('.txt') else filename[:filename.rfind('.')]
        with open(file, 'r', encoding='utf-8') as fp:
            refs.append((key, '\n'.join([line.strip() for line in fp])))
    # Group multi-references
    refs_dict = collections.defaultdict(list)
    for filename, text in refs:
        refs_dict[filename].append(text)
    refs = [texts for filename, texts in sorted(refs_dict.items(), key=lambda x: x[0], reverse=False)]

    return hyps, refs


def compare_score(s1, s2, e):
    return abs(s1 - s2) <= e


def get_hypothesis_references(test_case, all_hyps, all_refs):
    if test_case == 'one_summary_sentence':
        hyps = [all_hyps[0].split('\n')[0]]
        refs = [[all_refs[0][0].split('\n')[0]]]
    elif test_case == 'one_summary_document_one_ref':
        hyps = [all_hyps[0]]
        refs = [[all_refs[0][0]]]
    elif test_case == 'one_summary_document_multiple_refs':
        hyps = [all_hyps[0]]
        refs = [all_refs[0]]
    elif test_case == 'all_summaries_all_refs':
        hyps = all_hyps
        refs = all_refs
    elif test_case == 'all_input_output':
        hyps = []
        with open('all_input.txt', 'r', encoding='utf-8') as fp:
            for line in fp:
                hyps.append(line.strip())
            hyps = '\n'.join(hyps)
        refs = []
        with open('all_output.txt', 'r', encoding='utf-8') as fp:
            for line in fp:
                refs.append(line.strip())
            refs = ['\n'.join(refs)]
        hyps = [hyps]
        refs = [refs]
    else:
        raise ValueError('Error: does not recognize {}'.format(test_case))

    return hyps, refs


def get_rouge_args(rouge_dir, metrics, N, stemming, apply_avg, apply_best, alpha, length_limit_type, length_type, length_limit, weight_factor=1.2):
    rouge_args = ' -e {}'.format(rouge_dir + '/data')
    if "rouge-n" in metrics:
        rouge_args +=' -n {}'.format(N)
    if "rouge-l" not in metrics:
        rouge_args += ' -x'
    if "rouge-w" in metrics:
        rouge_args += ' -w {}'.format(weight_factor)
    if stemming:
        rouge_args +=' -m'
    rouge_args +=' -c 95 -r 50000'
    rouge_args +=' -f {}'.format('A' if apply_avg else ('B' if apply_best else ''))
    rouge_args +=' -p {}'.format(alpha)
    rouge_args +=' -t 0'
    if length_limit_type:
        rouge_args +=' -{} {}'.format('l' if length_type == 'words' else ('b' if length_type == 'bytes' else ''), length_limit)
    rouge_args +=' -v -a'
    return rouge_args


def run_perl_rouge_script(hyps, refs, rouge_dir, metrics, N, stemming, apply_avg, apply_best, alpha, limit_length, length_limit_type, length_limit, tmp_folder='/Users/diego/Github/py-rouge/tests/tmp'):
    rouge_args = get_rouge_args(rouge_dir, metrics, N, stemming, apply_avg, apply_best, alpha, limit_length, length_limit_type, length_limit)

    rouge_perl_time = time.time()

    model_folder = os.path.join(tmp_folder, 'model')
    peer_folder = os.path.join(tmp_folder, 'peer')

    if not os.path.exists(tmp_folder):
        os.makedirs(tmp_folder)
    if os.path.exists(model_folder):
        shutil.rmtree(model_folder)
    os.makedirs(model_folder)
    if os.path.exists(peer_folder):
        shutil.rmtree(peer_folder)
    os.makedirs(peer_folder)

    annotators = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    offset_filename = 1000000  # to avoid having 10.txt before 2.txt
    for i, (hyp, ref) in enumerate(zip(hyps, refs)):
        assert len(ref) <= len(annotators)

        with open(os.path.join(peer_folder, '{}.txt'.format(offset_filename + i)), 'w', encoding='utf-8') as fp:
            fp.write(hyp + '\n')

        for j, ref_ in enumerate(ref):
            with open(os.path.join(model_folder, '{}.{}.txt'.format(offset_filename + i, annotators[j])), 'w',
                      encoding='utf-8') as fp:
                fp.write(ref_ + '\n')

    rouge_perl = setup_rouge_perl(rouge_dir, rouge_args, model_folder, peer_folder)
    rouge_output = rouge_perl.convert_and_evaluate(rouge_args=rouge_args)
    rouge_perl_scores = parse_rouge_for_results(rouge_output)
    rouge_perl_ngrams_count = parse_rouge_for_ngrams_counts(rouge_output)

    # Clean
    shutil.rmtree(tmp_folder)

    rouge_perl_time = time.time() - rouge_perl_time

    return rouge_perl_time, rouge_perl_scores, rouge_perl_ngrams_count


def run_python_rouge_script(hyps, refs, metrics, N, stemming, apply_avg, apply_best, alpha, limit_length, length_limit_type, length_limit, weight_factor=1.0):
    rouge_python_time = time.time()
    rouge_python = setup_rouge_python(metrics, N, stemming, apply_avg, apply_best, alpha, limit_length, length_limit_type, length_limit, weight_factor)
    rouge_python_scores = rouge_python.get_scores(hyps, refs)
    rouge_python_time = time.time() - rouge_python_time

    return rouge_python_time, rouge_python_scores


def run_a_single_t_est(rouge_dir, metrics, N, stemming, apply_avg, apply_best, alpha, limit_length, length_limit_type, length_limit, weight_factor=1.0, tmp_folder='/Users/diego/Github/py-rouge/tests/tmp'):
    results = {}
    for test_folder in ['summaries_1', 'summaries_2']:
        results[test_folder] = {}
        system_dir = os.path.join('./', test_folder, 'system')
        model_dir = os.path.join('./', test_folder, 'references')

        all_hyps, all_refs = get_peers_models(system_dir, model_dir)

        test_cases = ['one_summary_sentence', 'one_summary_document_one_ref', 'one_summary_document_multiple_refs', 'all_summaries_all_refs']
        if (length_limit_type == 'words' and 0 < length_limit < 300) or (length_limit_type == 'bytes' and 0 < length_limit < 1000):
            test_cases.append('all_input_output')
        for test_case in test_cases:
            hyps, refs = get_hypothesis_references(test_case, all_hyps, all_refs)

            rouge_python_time, rouge_python_scores = run_python_rouge_script(hyps, refs, metrics, N, stemming, apply_avg, apply_best, alpha, limit_length, length_limit_type, length_limit, weight_factor)
            rouge_perl_time, rouge_perl_scores, rouge_perl_ngrams_count = run_perl_rouge_script(hyps, refs, rouge_dir, metrics, N, stemming, apply_avg, apply_best, alpha, limit_length, length_limit_type, length_limit, tmp_folder)

            results[test_folder][test_case] = {'python':{'time':rouge_python_time,
                                                         'scores':rouge_python_scores},
                                               'perl':{'time':rouge_perl_time,
                                                       'scores':rouge_perl_scores,
                                                       'ngrams_count':rouge_perl_ngrams_count}
                                               }

    return results


def run_a_single_t_est_on_all_files_rouge_n(metrics, N, alpha, apply_avg, apply_best, length_limit, length_limit_type, limit_length, rouge_dir, stemming, epsilon_ngrams_count_and_hits, epsilon_avg_with_resampling):
    results = run_a_single_t_est(rouge_dir, metrics, N, stemming, apply_avg, apply_best, alpha, limit_length, length_limit_type, length_limit, tmp_folder='/Users/diego/Github/py-rouge/tests/tmp_rouge_n')

    all_asserts = []
    for test_folder in results.keys():
        for test_case in results[test_folder].keys():
            message = 'test_folder:{}, test_case:{}'.format(test_folder, test_case)
            # Compute scores based on output of ROUGE script for #n-grams hyp, ref and intersection
            len_hyps = int(len(results[test_folder][test_case]['perl']['ngrams_count']) / N)
            for n in range(1, N + 1):
                local_results = {'p': 0.0, 'r': 0.0, 'f': 0.0}
                for ngrams_counts in results[test_folder][test_case]['perl']['ngrams_count'][(n - 1) * len_hyps:n * len_hyps]:  # rouge_perl_ngrams_count return first 1-gram for all hyps, then 2-gram etc
                    evaluated_count = int(ngrams_counts['{}-gram'.format(n)]['peer'])
                    reference_count = int(ngrams_counts['{}-gram'.format(n)]['model'])
                    overlapping_count = int(ngrams_counts['{}-gram'.format(n)]['hit'])
                    score = rouge.Rouge._compute_p_r_f_score(evaluated_count, reference_count, overlapping_count, alpha=alpha)

                    for stat, val in score.items():
                        local_results[stat] += round(val, 5)  # Perl script as only 5 digits

                for stat in local_results.keys():
                    local_results[stat] /= len_hyps
                    perl_score_val = local_results[stat]
                    python_score_val = results[test_folder][test_case]['python']['scores']['rouge-{}'.format(n)][stat]
                    comparison = compare_score(perl_score_val, python_score_val, epsilon_ngrams_count_and_hits)
                    all_asserts.append((comparison, 'For ROUGE-{} {}, Perl: {:.5f} Python: {:.5f}'.format(n, stat, perl_score_val, python_score_val) + ' Scores based on n-grams count & hit. ' + message))

            # Compare real scores. As sampling is high, they should be more or less the same.
            for n in range(1, N + 1):
                perl_scores = results[test_folder][test_case]['perl']['scores']['ROUGE-{}'.format(n)]
                python_scores = results[test_folder][test_case]['python']['scores']['rouge-{}'.format(n)]

                for stat in perl_scores.keys():
                    perl_score_val = round(perl_scores[stat]['val'], 5)
                    python_score_val = round(python_scores[stat.lower()], 5)
                    comparison = compare_score(perl_score_val, python_score_val, epsilon_avg_with_resampling)
                    all_asserts.append((comparison, 'For ROUGE-{} {}, Perl: {:.5f} Python: {:.5f}'.format(n, stat, perl_score_val, python_score_val) + ' Scores based on final ROUGE scores. ' + message))
    return all_asserts


def run_a_single_t_est_on_all_files_rouge_l(metrics, N, alpha, apply_avg, apply_best, length_limit, length_limit_type, limit_length, rouge_dir, stemming, epsilon_ngrams_count_and_hits, epsilon_avg_with_resampling):
    results = run_a_single_t_est(rouge_dir, metrics, N, stemming, apply_avg, apply_best, alpha, limit_length, length_limit_type, length_limit, tmp_folder='/Users/diego/Github/py-rouge/tests/tmp_rouge_l')

    all_asserts = []
    for test_folder in results.keys():
        for test_case in results[test_folder].keys():
            message = 'test_folder:{}, test_case:{}'.format(test_folder, test_case)
            # Compute scores based on output of ROUGE script for #n-grams hyp, ref and intersection
            local_results = {'p': 0.0, 'r': 0.0, 'f': 0.0}
            len_hyps = len(results[test_folder][test_case]['perl']['ngrams_count'])
            for ngrams_counts in results[test_folder][test_case]['perl']['ngrams_count']:
                evaluated_count = int(ngrams_counts['ROUGE-L']['peer'])
                reference_count = int(ngrams_counts['ROUGE-L']['model'])
                overlapping_count = int(ngrams_counts['ROUGE-L']['hit'])
                score = rouge.Rouge._compute_p_r_f_score(evaluated_count, reference_count, overlapping_count, alpha=alpha)

                for stat, val in score.items():
                    local_results[stat] += round(val, 5)  # Perl script as only 5 digits

            for stat in local_results.keys():
                local_results[stat] /= len_hyps
                perl_score_val = local_results[stat]
                python_score_val = results[test_folder][test_case]['python']['scores']['rouge-l'][stat]
                comparison = compare_score(perl_score_val, python_score_val, epsilon_ngrams_count_and_hits)
                all_asserts.append((comparison, 'For ROUGE-L {}, Perl: {:.5f} Python: {:.5f}'.format(stat, perl_score_val, python_score_val) + ' Scores based on n-grams count & hit. ' + message))

            # Compare real scores. As sampling is high, they should be more or less the same.
            perl_scores = results[test_folder][test_case]['perl']['scores']['ROUGE-L']
            python_scores = results[test_folder][test_case]['python']['scores']['rouge-l']

            for stat in perl_scores.keys():
                perl_score_val = round(perl_scores[stat]['val'], 5)
                python_score_val = round(python_scores[stat.lower()], 5)
                comparison = compare_score(perl_score_val, python_score_val, epsilon_avg_with_resampling)
                all_asserts.append((comparison, 'For ROUGE-L {}, Perl: {:.5f} Python: {:.5f}'.format(stat, perl_score_val, python_score_val) + ' Scores based on final ROUGE scores. ' + message))
    return all_asserts


def run_a_single_t_est_on_all_files_rouge_w(metrics, N, alpha, apply_avg, apply_best, length_limit, length_limit_type, limit_length, rouge_dir, stemming, weight_factor, epsilon_ngrams_count_and_hits, epsilon_avg_with_resampling):
    results = run_a_single_t_est(rouge_dir, metrics, N, stemming, apply_avg, apply_best, alpha, limit_length, length_limit_type, length_limit, weight_factor, tmp_folder='/Users/diego/Github/py-rouge/tests/tmp_rouge_w')

    all_asserts = []
    for test_folder in results.keys():
        for test_case in results[test_folder].keys():
            message = 'test_folder:{}, test_case:{}'.format(test_folder, test_case)
            # Compute scores based on output of ROUGE script for #n-grams hyp, ref and intersection
            local_results = {'p': 0.0, 'r': 0.0, 'f': 0.0}
            len_hyps = len(results[test_folder][test_case]['perl']['ngrams_count'])
            for ngrams_counts in results[test_folder][test_case]['perl']['ngrams_count']:
                evaluated_count = float(ngrams_counts['ROUGE-W']['peer'])
                reference_count = float(ngrams_counts['ROUGE-W']['model'])
                overlapping_count = float(ngrams_counts['ROUGE-W']['hit'])
                score = rouge.Rouge._compute_p_r_f_score(evaluated_count, reference_count, overlapping_count, alpha=alpha, weight_factor=weight_factor)

                for stat, val in score.items():
                    local_results[stat] += round(val, 5)  # Perl script as only 5 digits

            for stat in local_results.keys():
                local_results[stat] /= len_hyps
                perl_score_val = local_results[stat]
                python_score_val = results[test_folder][test_case]['python']['scores']['rouge-w'][stat]
                comparison = compare_score(perl_score_val, python_score_val, epsilon_ngrams_count_and_hits)
                all_asserts.append((comparison, 'For ROUGE-W {}, Perl: {:.5f} Python: {:.5f}'.format(stat, perl_score_val, python_score_val) + ' Scores based on n-grams count & hit. ' + message))

            # Compare real scores. As sampling is high, they should be more or less the same.
            perl_scores = results[test_folder][test_case]['perl']['scores']['ROUGE-W']
            python_scores = results[test_folder][test_case]['python']['scores']['rouge-w']

            for stat in perl_scores.keys():
                perl_score_val = round(perl_scores[stat]['val'], 5)
                python_score_val = round(python_scores[stat.lower()], 5)
                comparison = compare_score(perl_score_val, python_score_val, epsilon_avg_with_resampling)
                all_asserts.append((comparison, 'For ROUGE-W {}, Perl: {:.5f} Python: {:.5f}'.format(stat, perl_score_val, python_score_val) + ' Scores based on final ROUGE scores. ' + message))
    return all_asserts