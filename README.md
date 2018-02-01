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

The durability of data is simply the inverse of the probability of losing the data
over than same period of time:

    durability = 1 - failure_rate
  
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

The python code in `durability.py` does the calculations above, with a few tweaks
to maintain precision when dealing with tiny numbers, and prints out the results
for a given set of assumptions:

```
$ python durability.py
usage: durability.py [-h]
                     data_shards parity_shards annual_shard_failure_rate
                     shard_replacement_days
durability.py: error: too few arguments
$ python durability.py 4 2 0.10 1

#
# total shards: 6
# replacement period (days):  1.0
# annual shard failure rate: 0.10
#

|==================================================================================================================================|
| failure_threshold | individual_prob | cumulative_prob | annual_loss_rate |         annual_odds |        durability |       nines | 
|----------------------------------------------------------------------------------------------------------------------------------|
|                 6 |       4.229e-22 |       4.229e-22 |        1.544e-19 | 154 in a sextillion | 1.000000000000000 |    18 nines | 
|                 5 |       9.259e-18 |       9.260e-18 |        3.380e-15 |  3 in a quadrillion | 0.999999999999997 |    14 nines | 
|                 4 |       8.447e-14 |       8.448e-14 |        3.083e-11 |    31 in a trillion | 0.999999999969167 |    10 nines | 
|                 3 |       4.110e-10 |       4.110e-10 |        1.500e-07 |    150 in a billion | 0.999999849970558 | --> 6 nines | 
|                 2 |       1.125e-06 |       1.125e-06 |        4.106e-04 |    411 in a million | 0.999589425325156 |     3 nines | 
|                 1 |       1.642e-03 |       1.643e-03 |        4.512e-01 |            5 in ten | 0.548766521902091 |     0 nines | 
|                 0 |       9.984e-01 |       1.000e+00 |        1.000e+00 |              always | 0.000000000000000 |     0 nines | 
|==================================================================================================================================|
```


