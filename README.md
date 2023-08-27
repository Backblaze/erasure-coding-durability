# erasure-coding-durability

## Overview

This is a simple statistical model for calculating the probability of losing
data that is stored using an erasure coding system, such as Reed-Solomon.

In erasure coding, each file stored is divided into D shards of the same length.
Then, P parity shards are computed using erasure coding, resulting in D+P shards.
Even if shards are lost, the original file can be recomputed from any D of the
D+P shards stored.  In other words, you can lose any P of the shards and still 
reconstruct the original file.

What we would like to compute is the durability of data stored with erasure coding
based on the durability of the individual shards.
The durability is the probability of not losing the data over a period of time.
The period of time we use here is one year, resulting in annual durability.
  
Systems that use erasure coding to store data will replace shards that are lost.
Once a shard is replaced, the data is fully intact again.  Data is lost only when
P+1 shards are all lost at the same time, before they are replaced.

## Assumptions

To calculate the probability of loss, we need to make some assumptions:

1. Data is stored using *D* data shards and *P* parity shards, and is lost when *P+1* shards are lost.
1. The annual failure rate of each shard is *shard_annual_failure_rate*.
1. The number of days it takes to replace a failed shard is *shard_failure_days*.
1. The failures of individual shards are independent.

## Calculation

The details of the calculations are in [calculation.ipynb](https://github.com/Backblaze/erasure-coding-durability/blob/master/calculation.ipynb).

## Python code

The python code in 
[durability.py](https://github.com/Backblaze/erasure-coding-durability/blob/master/durability.py)
does the calculations above, with a few tweaks
to maintain precision when dealing with tiny numbers, and prints out the results
for a given set of assumptions:

```
$ python3 durability.py
usage: durability.py [-h]
                     data_shards parity_shards annual_shard_failure_rate
                     shard_replacement_days
durability.py: error: too few arguments
$ python3 durability.py 17 3 0.00405 6.5

#
# total shards: 20
# replacement period (days): 6.5000
# annual shard failure rate: 0.0040
#

|===================================================================================================================================|
| failure_threshold | individual_prob | cumulative_prob | annual_loss_rate |         annual_odds |        durability |        nines | 
|-----------------------------------------------------------------------------------------------------------------------------------|
|                20 |       1.449e-83 |       1.449e-83 |        8.117e-82 |               NEVER | 1.000000000000000 |     81 nines | 
|                19 |       4.019e-78 |       4.019e-78 |        2.251e-76 |               NEVER | 1.000000000000000 |     75 nines | 
|                18 |       5.294e-73 |       5.294e-73 |        2.965e-71 |               NEVER | 1.000000000000000 |     70 nines | 
|                17 |       4.404e-68 |       4.404e-68 |        2.466e-66 |               NEVER | 1.000000000000000 |     65 nines | 
|                16 |       2.595e-63 |       2.595e-63 |        1.453e-61 |               NEVER | 1.000000000000000 |     60 nines | 
|                15 |       1.151e-58 |       1.151e-58 |        6.447e-57 |               NEVER | 1.000000000000000 |     56 nines | 
|                14 |       3.991e-54 |       3.991e-54 |        2.235e-52 |               NEVER | 1.000000000000000 |     51 nines | 
|                13 |       1.107e-49 |       1.107e-49 |        6.197e-48 |               NEVER | 1.000000000000000 |     47 nines | 
|                12 |       2.493e-45 |       2.493e-45 |        1.396e-43 |               NEVER | 1.000000000000000 |     42 nines | 
|                11 |       4.609e-41 |       4.609e-41 |        2.581e-39 |               NEVER | 1.000000000000000 |     38 nines | 
|                10 |       7.029e-37 |       7.029e-37 |        3.936e-35 |               NEVER | 1.000000000000000 |     34 nines | 
|                 9 |       8.859e-33 |       8.860e-33 |        4.962e-31 |               NEVER | 1.000000000000000 |     30 nines | 
|                 8 |       9.212e-29 |       9.213e-29 |        5.159e-27 |   5 in an octillion | 1.000000000000000 |     26 nines | 
|                 7 |       7.860e-25 |       7.861e-25 |        4.402e-23 |  44 in a septillion | 1.000000000000000 |     22 nines | 
|                 6 |       5.449e-21 |       5.450e-21 |        3.052e-19 | 305 in a sextillion | 1.000000000000000 |     18 nines | 
|                 5 |       3.022e-17 |       3.022e-17 |        1.693e-15 |  2 in a quadrillion | 0.999999999999998 |     14 nines | 
|                 4 |       1.309e-13 |       1.310e-13 |        7.354e-12 |     7 in a trillion | 0.999999999992646 | --> 11 nines | 
|                 3 |       4.271e-10 |       4.273e-10 |        2.399e-08 |     24 in a billion | 0.999999976008104 |      7 nines | 
|                 2 |       9.870e-07 |       9.874e-07 |        5.545e-05 |     55 in a million | 0.999944554648366 |      4 nines | 
|                 1 |       1.440e-03 |       1.441e-03 |        7.781e-02 |      8 in a hundred | 0.922193691444580 |      1 nines | 
|                 0 |       9.986e-01 |       1.000e+00 |        1.000e+00 |              always | 0.000000000000000 |      0 nines | 
|===================================================================================================================================|
```


