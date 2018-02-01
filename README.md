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
1. The annual failure rate of each shard is *shard_failure_rate*.
1. The number of days it takes to replace a failed shard is *shard_failure_days*.
1. The failures of individual shards are independent.

## Calculation

Let look at one period of *shard_failure_days*.  What are the chances of losing
data in that period?

