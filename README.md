# py-rouge
A full Python implementation of the ROUGE metric, producing same results as in the official perl implementation.  

# Talk about stemming
# Talk about resampling strategy, which might not give the mean as we might compute it by default. With a high resampling rate, the problem is solve
# Use my repo for pyrouge because stemming is forced
# Official ROUGE-1.,5.5. has a bug: should have $tmpTextLen += $sLen at line 2101. Here, the last sentence, $limitBytes is taken instead of $limitBytes-$tmpTextLen (as $tmpTextLen is never updated with bytes length limit)