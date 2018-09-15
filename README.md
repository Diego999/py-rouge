# Py-rouge
A full Python implementation of the ROUGE metric, producing same results as in the official perl implementation.  

# Important remarks
- The original Porter stemmer in NLTK is slightly different than the one use in the official ROUGE perl script as it has been written by end. Therefore, there might be slightly different stems for certain words. For DUC2004 dataset, I have identified these words and this script produces same stems.
- The official ROUGE perl script use resampling strategy to compute the average with confidence intervals. Therefore, we might have a difference `<3e-5` for ROUGE-L as well as ROUGE-W and `<4e-5` for ROUGE-N.
- Finally, ROUGE-1.5.5. has a bug: should have $tmpTextLen += $sLen at line 2101. Here, the last sentence, $limitBytes is taken instead of $limitBytes-$tmpTextLen (as $tmpTextLen is never updated with bytes length limit). It has been fixed in this code. This bug does not have a consequence for the default evaluation `-b 665`.

In case of doubts, please see all the implemented tests to compare outputs between the official ROUGE-1.5.5 and this script.

## Installation

Package is uploaded on `PyPI <https://pypi.org/project/py-rouge>`_.

You can install it with pip:
```shell
pip install py-rouge
```

or do it manually:
```shell
git clone https://github.com/Diego999/py-rouge
cd py-rouge
python setup.py install
```

# Issues/Pull Requests/Feedbacks
Don't hesitate to contact for any feedback or create issues/pull requests (especially if you want to rewrite the stemmer implemented in ROUGE-1.5.5 in python ;)).

# Example 
```python
import rouge


def prepare_results(p, r, f):
    return '\t{}:\t{}: {:5.2f}\t{}: {:5.2f}\t{}: {:5.2f}'.format(metric, 'P', 100.0 * p, 'R', 100.0 * r, 'F1', 100.0 * f)


for aggregator in ['Avg', 'Best', 'Individual']:
    print('Evaluation with {}'.format(aggregator))
    apply_avg = aggregator == 'Avg'
    apply_best = aggregator == 'Best'

    evaluator = rouge.Rouge(metrics=['rouge-n', 'rouge-l', 'rouge-w'],
                           max_n=4,
                           limit_length=True,
                           length_limit=100,
                           length_limit_type='words',
                           apply_avg=apply_avg,
                           apply_best=apply_best,
                           alpha=0.5, # Default F1_score
                           weight_factor=1.2,
                           stemming=True)


    hypothesis_1 = "King Norodom Sihanouk has declined requests to chair a summit of Cambodia 's top political leaders , saying the meeting would not bring any progress in deadlocked negotiations to form a government .\nGovernment and opposition parties have asked King Norodom Sihanouk to host a summit meeting after a series of post-election negotiations between the two opposition groups and Hun Sen 's party to form a new government failed .\nHun Sen 's ruling party narrowly won a majority in elections in July , but the opposition _ claiming widespread intimidation and fraud _ has denied Hun Sen the two-thirds vote in parliament required to approve the next government .\n"
    references_1 = ["Prospects were dim for resolution of the political crisis in Cambodia in October 1998.\nPrime Minister Hun Sen insisted that talks take place in Cambodia while opposition leaders Ranariddh and Sam Rainsy, fearing arrest at home, wanted them abroad.\nKing Sihanouk declined to chair talks in either place.\nA U.S. House resolution criticized Hun Sen's regime while the opposition tried to cut off his access to loans.\nBut in November the King announced a coalition government with Hun Sen heading the executive and Ranariddh leading the parliament.\nLeft out, Sam Rainsy sought the King's assurance of Hun Sen's promise of safety and freedom for all politicians.",
                    "Cambodian prime minister Hun Sen rejects demands of 2 opposition parties for talks in Beijing after failing to win a 2/3 majority in recent elections.\nSihanouk refuses to host talks in Beijing.\nOpposition parties ask the Asian Development Bank to stop loans to Hun Sen's government.\nCCP defends Hun Sen to the US Senate.\nFUNCINPEC refuses to share the presidency.\nHun Sen and Ranariddh eventually form a coalition at summit convened by Sihanouk.\nHun Sen remains prime minister, Ranariddh is president of the national assembly, and a new senate will be formed.\nOpposition leader Rainsy left out.\nHe seeks strong assurance of safety should he return to Cambodia.\n",
                    ]

    hypothesis_2 = "China 's government said Thursday that two prominent dissidents arrested this week are suspected of endangering national security _ the clearest sign yet Chinese leaders plan to quash a would-be opposition party .\nOne leader of a suppressed new political party will be tried on Dec. 17 on a charge of colluding with foreign enemies of China '' to incite the subversion of state power , '' according to court documents given to his wife on Monday .\nWith attorneys locked up , harassed or plain scared , two prominent dissidents will defend themselves against charges of subversion Thursday in China 's highest-profile dissident trials in two years .\n"
    references_2 = "Hurricane Mitch, category 5 hurricane, brought widespread death and destruction to Central American.\nEspecially hard hit was Honduras where an estimated 6,076 people lost their lives.\nThe hurricane, which lingered off the coast of Honduras for 3 days before moving off, flooded large areas, destroying crops and property.\nThe U.S. and European Union were joined by Pope John Paul II in a call for money and workers to help the stricken area.\nPresident Clinton sent Tipper Gore, wife of Vice President Gore to the area to deliver much needed supplies to the area, demonstrating U.S. commitment to the recovery of the region.\n"

    all_hypothesis = [hypothesis_1, hypothesis_2]
    all_references = [references_1, references_2]

    scores = evaluator.get_scores(all_hypothesis, all_references)

    for metric, results in sorted(scores.items(), key=lambda x: x[0]):
        if not apply_avg and not apply_best: # value is a type of list as we evaluate each summary vs each reference
            for hypothesis_id, results_per_ref in enumerate(results):
                nb_references = len(results_per_ref['p'])
                for reference_id in range(nb_references):
                    print('\tHypothesis #{} & Reference #{}: '.format(hypothesis_id, reference_id))
                    print('\t' + prepare_results(results_per_ref['p'][reference_id], results_per_ref['r'][reference_id], results_per_ref['f'][reference_id]))
            print()
        else:
            print(prepare_results(results['p'], results['r'], results['f']))
    print()
```

It produces the following output:
```
Evaluation with Avg
	rouge-1:	P: 28.62	R: 26.46	F1: 27.49
	rouge-2:	P:  4.21	R:  3.92	F1:  4.06
	rouge-3:	P:  0.80	R:  0.74	F1:  0.77
	rouge-4:	P:  0.00	R:  0.00	F1:  0.00
	rouge-l:	P: 30.52	R: 28.57	F1: 29.51
	rouge-w:	P: 15.85	R:  8.28	F1: 10.87

Evaluation with Best
	rouge-1:	P: 30.44	R: 28.36	F1: 29.37
	rouge-2:	P:  4.74	R:  4.46	F1:  4.59
	rouge-3:	P:  1.06	R:  0.98	F1:  1.02
	rouge-4:	P:  0.00	R:  0.00	F1:  0.00
	rouge-l:	P: 31.54	R: 29.71	F1: 30.60
	rouge-w:	P: 16.42	R:  8.82	F1: 11.47

Evaluation with Individual
	Hypothesis #0 & Reference #0: 
		rouge-1:	P: 38.54	R: 35.58	F1: 37.00
	Hypothesis #0 & Reference #1: 
		rouge-1:	P: 45.83	R: 43.14	F1: 44.44
	Hypothesis #1 & Reference #0: 
		rouge-1:	P: 15.05	R: 13.59	F1: 14.29

	Hypothesis #0 & Reference #0: 
		rouge-2:	P:  7.37	R:  6.80	F1:  7.07
	Hypothesis #0 & Reference #1: 
		rouge-2:	P:  9.47	R:  8.91	F1:  9.18
	Hypothesis #1 & Reference #0: 
		rouge-2:	P:  0.00	R:  0.00	F1:  0.00

	Hypothesis #0 & Reference #0: 
		rouge-3:	P:  2.13	R:  1.96	F1:  2.04
	Hypothesis #0 & Reference #1: 
		rouge-3:	P:  1.06	R:  1.00	F1:  1.03
	Hypothesis #1 & Reference #0: 
		rouge-3:	P:  0.00	R:  0.00	F1:  0.00

	Hypothesis #0 & Reference #0: 
		rouge-4:	P:  0.00	R:  0.00	F1:  0.00
	Hypothesis #0 & Reference #1: 
		rouge-4:	P:  0.00	R:  0.00	F1:  0.00
	Hypothesis #1 & Reference #0: 
		rouge-4:	P:  0.00	R:  0.00	F1:  0.00

	Hypothesis #0 & Reference #0: 
		rouge-l:	P: 42.11	R: 39.39	F1: 40.70
	Hypothesis #0 & Reference #1: 
		rouge-l:	P: 46.19	R: 43.92	F1: 45.03
	Hypothesis #1 & Reference #0: 
		rouge-l:	P: 16.88	R: 15.50	F1: 16.16

	Hypothesis #0 & Reference #0: 
		rouge-w:	P: 22.27	R: 11.49	F1: 15.16
	Hypothesis #0 & Reference #1: 
		rouge-w:	P: 24.56	R: 13.60	F1: 17.51
	Hypothesis #1 & Reference #0: 
		rouge-w:	P:  8.29	R:  4.04	F1:  5.43
```    