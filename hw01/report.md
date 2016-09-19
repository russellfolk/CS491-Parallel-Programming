Russell Folk
CS 491/CS 521 Parallel Programming
Homework 1 Report

# Data Locality and Fine-Grained Parallelism

### Disclaimer
For these tests I changed the size of N to be 128 up from 64, this was to more readily see the effects of the various optimizations. Furthermore, all tests were run on the computer named koss in the grad lab. It is uncertain if I was the only user when these tests were run so results may vary.

## Optimize by Loop Permutation
The best results are determined by ordering the loops l, k, i, j which gives an average of 512.2 MFLOPS, up from the original implementation which was on average of 94 MFLOPS when running the default i, j, k, l ordering. The second best ordering is k, i, l, j which gives an average of 459.133 MFLOPS. The performance increase obtained with these orderings is 444.89% and 388.27% respectively. By far, as will be noticed throughout this report, correct ordering of the loop permutation gives the single best optimization available to be obtained. As we can see in figure 1, the differences in the 2nd through the 6th best results is not very large, yet the difference between them and the unoptimized code is very significant. This leads me to believe that this is the single most important optimization.

These tests were run in a very manual way by changing the order of the loops in emacs and executing all 24 permutations 3 times each and recording the results. I chose to use megaflops as the metric by which to judge improvement as runtimes I saw had more variation. The percentage improvement calculation is obtained by (newValue - oldValue)/oldValue*100. For all values, refer to table 1 in the appendix.

I was mildly surprised that none of the permutations was significantly worse than the default configuration.

## Optimize by Loop Unrolling
### Disclaimer: loop unrolling permutations
>I spent a lot of time testing loop unrolling including using various permutations of mixed unrolling. However, I did not observe significantly better unrolling combinations than what was achieved by only unrolling a single loop at once. E.g. unrolling the i-loop by 2 and unrolling the j-loop by 4 did not significantly improve performance. Thus, due to time constraints and lack of true differentiation, I did not include the results of these trials in this report.

The results obtained by optimizing via loop unrolling were very time intensive and not very useful over all. In actuality, only unrolling the j loop gave any improvements. *Please note that at this stage the original i, j, k, l ordering was maintained.* However, as referenced in figure 2, the gains to be had here are limited compared to loop permutation. We will investigate further how mixing the two works in favor. For all values, refer to table 2 in the appendix.

