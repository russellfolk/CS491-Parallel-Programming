Russell Folk
CS 491/CS 521 Parallel Programming
Homework 1 Report

# Data Locality and Fine-Grained Parallelism

### Disclaimer
For these tests I changed the size of N to be 128 up from 64, this was to more readily see the effects of the various optimizations. Furthermore, all tests were run on the computer named koss in the grad lab. It is uncertain if I was the only user when these tests were run so results may vary.

## Optimize by Loop Permutation
The best results are determined by ordering the loops l, k, i, j which gives an average of 512.2 MFLOPS, up from the original implementation which was on average of 94 MFLOPS when running the default i, j, k, l ordering. The second best ordering is k, i, l, j which gives an average of 459.133 MFLOPS. The performance increase obtained with these orderings is 444.89% and 388.27% respectively. By far, as will be noticed throughout this report, correct ordering of the loop permutation gives the single best optimization available to be obtained.

These tests were run in a very manual way by changing the order of the loops in emacs and executing all 24 permutations 3 times each and recording the results. I chose to use megaflops as the metric by which to judge improvement as runtimes I saw had more variation. The percentage improvement calculation is obtained by (newValue - oldValue)/oldValue*100.

I was mildly surprised that none of the permutations was significantly worse than the default configuration.

