#!/usr/bin/env python2
######################################################################
# 
# File: durability.py
# 
# Copyright 2018 Backblaze Inc. All Rights Reserved.
# 
######################################################################

import argparse
import math
import sys
import unittest


class Table(object):

    """
    Knows how to display a table of data.

    The data is in the form of a list of dicts:

        [ { 'a' : 4, 'b' : 8 },
          { 'a' : 5, 'b' : 9 } ]

    And is displayed like this:

        |=======|
        | a | b |
        |-------|
        | 4 | 8 |
        | 5 | 9 |
        |=======|
    """

    def __init__(self, data, column_names):
        self.data = data
        self.column_titles = column_names
        self.column_widths = [
            max(len(col), max(len(item[col]) for item in data))
            for col in column_names
        ]

    def __str__(self):
        result = []

        # Title row
        total_width = 1 + sum(3 + w for w in self.column_widths)
        result.append('|')
        result.append('=' * (total_width - 2))
        result.append('|')
        result.append('\n')
        result.append('| ')
        for (col, w) in zip(self.column_titles, self.column_widths):
            result.append(self.pad(col, w))
            result.append(' | ')
        result.append('\n')
        result.append('|')
        result.append('-' * (total_width - 2))
        result.append('|')
        result.append('\n')

        # Data rows
        for item in self.data:
            result.append('| ')
            for (col, w) in zip(self.column_titles, self.column_widths):
                result.append(self.pad(item[col], w))
                result.append(' | ')
            result.append('\n')
        result.append('|')
        result.append('=' * (total_width - 2))
        result.append('|')
        result.append('\n')

        return ''.join(result)

    def pad(self, s, width):
        if len(s) < width:
            return (' ' * (width - len(s))) + s
        else:
            return s[:width]


def print_markdown_table(data, column_names):
    print
    print ' | '.join(column_names)
    print ' | '.join(['---'] * len(column_names))
    for item in data:
        print ' | '.join(item[cn] for cn in column_names)
    print


def factorial(n):
    if n == 0:
        return 1
    else:
        return n * factorial(n - 1)


def choose(n, r):
    """
    Returns: How many ways there are to choose a subset of n things from a set of r things.

    Computes n! / (r! (n-r)!) exactly. Returns a python long int.

    From: http://stackoverflow.com/questions/3025162/statistics-combinations-in-python
    """
    assert n >= 0
    assert 0 <= r <= n

    c = 1L
    for num, denom in zip(xrange(n, n-r, -1), xrange(1, r+1, 1)):
        c = (c * num) // denom
    return c


def binomial_probability(k, n, p):
    """
    Returns: The probability of exactly k of n things happening, when the
             probability of each one (independently) is p.

    See: https://en.wikipedia.org/wiki/Binomial_distribution#Probability_mass_function
    """
    return choose(n, k) * (p ** k) * ((1 - p) ** (n - k))


class TestBinomialProbability(unittest.TestCase):

    def test_binomial_probability(self):
        # these test cases are from the Wikipedia page
        self.assertAlmostEqual(0.117649, binomial_probability(0, 6, 0.3))
        self.assertAlmostEqual(0.302526, binomial_probability(1, 6, 0.3))
        self.assertAlmostEqual(0.324135, binomial_probability(2, 6, 0.3))

        # Wolfram Alpha: (1 - 1e-6)^800
        self.assertAlmostEqual(0.9992003, binomial_probability(0, 800, 1.0e-6))


def probability_of_failure_for_failure_rate(f):
    """
    Given a failure rate f, what's the probability of at least one failure?
    """
    probability_of_no_failures = math.exp(-f)
    return 1.0 - probability_of_no_failures


def probability_of_failure_in_any_period(p, n):
    """
    Returns the probability that a failure (of probability p in one period)
    happens once or more in n periods.

    The probability of failure in one period is p, so the probability
    of not failing is (1 - p).  So the probability of not
    failing over n periods is (1 - p) ** n, and the probability
    of one or more failures in n periods is:

        1 - (1 - p) ** n

    Doing the math without losing precision is tricky.
    After the binomial expansion, you get (for even n):

        a = 1 - (1 - choose(n, 1) * p + choose(n, 2) p**2 - p**3 + p**4 ... + choose(n, n) p**n)

    For odd n, the last term is negative.

    To avoid precision loss, we don't want to to (1 - p) if p is
    really tiny, so we'll cancel out the 1 and get:
    you get:

        a = choose(n, 1) * p - choose(n, 2) * p**2 ...
    """
    if p < 0.01:
        # For tiny numbers, (1 - p) can lose precision.
        # First, compute the result for the integer part
        n_int = int(n)
        result = 0.0
        sign = 1
        for i in xrange(1, n_int + 1):
            p_exp_i = p ** i
            if p_exp_i != 0:
                result += sign * choose(n_int, i) * (p ** i)
            sign = -sign
        # Adjust the result to include the fractional part
        # What we want is: 1.0 - (1.0 - result) * ((1.0 - p) ** (n - n_int))
        # Which gives this when refactored:
        result = 1.0 - ((1.0 - p) ** (n - n_int)) + result * ((1.0 - p) ** (n - n_int))
        return result
    else:
        # For high probabilities of loss, the powers of p don't
        # get small faster than the coefficients get big, and weird
        # things happen
        return 1.0 - (1.0 - p) ** n


class TestProbabilityOfFailureAnyPeriod(unittest.TestCase):

    def test_probability_of_failure(self):
        # Easy to check
        self.assertAlmostEqual(0.25, probability_of_failure_in_any_period(0.25, 1))
        self.assertAlmostEqual(0.4375, probability_of_failure_in_any_period(0.25, 2))
        self.assertAlmostEqual(0.0199, probability_of_failure_in_any_period(0.01, 2))

        # From Wolfram Alpha, some tests with tiny probabilities:
        self.assertAlmostEqual(2.0, probability_of_failure_in_any_period(1e-10, 200) * 1e8)
        self.assertAlmostEqual(2.0, probability_of_failure_in_any_period(1e-30, 200) * 1e28)
        self.assertAlmostEqual(7.60690480739, probability_of_failure_in_any_period(3.47347251479e-103, 2190) * 1e100)

        # Check fractional exponents
        self.assertAlmostEqual(0.1339746, probability_of_failure_in_any_period(0.25, 0.5))
        self.assertAlmostEqual(0.0345647, probability_of_failure_in_any_period(0.01, 3.5))


SCALE_TABLE = [
    (1, 'ten'),
    (2, 'a hundred'),
    (3, 'a thousand'),
    (6, 'a million'),
    (9, 'a billion'),
    (12, 'a trillion'),
    (15, 'a quadrillion'),
    (18, 'a quintillion'),
    (21, 'a sextillion'),
    (24, 'a septillion'),
    (27, 'an octillion')
    ]


def pretty_probability(p):
    """
    Takes a number between 0 and 1 and prints it as a probability in
    the form "5 in a million"
    """
    if abs(p - 1.0) < 0.01:
        return 'always'
    for (power, name) in SCALE_TABLE:
        count = p * (10.0 ** power)
        if count >= 0.90:
            return '%d in %s' % (round(count), name)
    return 'NEVER'


def count_nines(loss_rate):
    """
    Returns the number of nines after the decimal point before some other digit happens.
    """
    nines = 0
    power_of_ten = 0.1
    while True:
        if power_of_ten < loss_rate:
            return nines
        power_of_ten /= 10.0
        nines += 1
        if power_of_ten == 0.0:
            return 0


def do_scenario(total_shards, min_shards, annual_shard_failure_rate, shard_replacement_days):
    """
    Calculates the cumulative failure rates for different numbers of
    failures, starting with the most possible, down to 0.

    The first probability in the table will be extremely improbable,
    because it is the case where ALL of the shards fail.  The next
    line in the table is the case where either all of the shards fail,
    or all but one fail.  The final row in the table is the case where
    somewhere between all fail and none fail, which always happens, so
    the probability is one.
    """

    num_periods = 365.0 / shard_replacement_days
    failure_rate_per_period = annual_shard_failure_rate / num_periods

    print
    print '#'
    print '# total shards:', total_shards
    print '# replacement period (days): %6.4f' % (shard_replacement_days)
    print '# annual shard failure rate: %6.4f' % (annual_shard_failure_rate)
    print '#'
    print

    failure_probability_per_period = 1.0 - math.exp(-failure_rate_per_period)
    data = []
    period_cumulative_prob = 0.0
    for failed_shards in xrange(total_shards, -1, -1):
        period_failure_prob = binomial_probability(failed_shards, total_shards, failure_probability_per_period)
        period_cumulative_prob += period_failure_prob
        annual_loss_prob = probability_of_failure_in_any_period(period_cumulative_prob, num_periods)
        nines = '%d nines' % count_nines(annual_loss_prob)
        if failed_shards == total_shards - min_shards + 1:
            nines = "--> " + nines
        data.append({
            'individual_prob' : ('%10.3e' % period_failure_prob),
            'failure_threshold' : str(failed_shards),
            'cumulative_prob' : ('%10.3e' % period_cumulative_prob),
            'cumulative_odds' : pretty_probability(period_cumulative_prob),
            'annual_loss_rate' : ('%10.3e' % annual_loss_prob),
            'annual_odds' : pretty_probability(annual_loss_prob),
            'durability' : '%17.15f' % (1.0 - annual_loss_prob),
            'nines' : nines
            })

    print Table(data, ['failure_threshold',
                       'individual_prob',
                       'cumulative_prob',
                       'annual_loss_rate',
                       'annual_odds',
                       'durability',
                       'nines'
                       ])
    print

    return dict(
        (item['failure_threshold'], item)
        for item in data
        )


def example():
    """
    This is the example in the explanation.
    """
    # Make the table of probabilities of k failures with a failure rate of 2.0:
    p = 2.0
    data = [
        { 'k': str(k), 'p': '%6.4f' % (math.exp(-p) * p**k / factorial(k),) }
        for k in xrange(7)
    ]
    print_markdown_table(data, ['k', 'p'])

    print 'Probability of n Failing'
    annual_rate = 0.25
    p_one_failing = probability_of_failure_for_failure_rate(annual_rate)
    print 'probability of one failing: %6.4f' % p_one_failing
    print 'probability of none failing: %6.4f' % (1 - p_one_failing)
    print 'probability of three not failing: %6.4f' % (1 - p_one_failing) ** 3
    print 'probability of two or more failing: %6.4f' % (binomial_probability(2, 3, p_one_failing) + binomial_probability(3, 3, p_one_failing))
    print
    probs = {'ok': (1 - p_one_failing), 'FAIL': p_one_failing}
    data = []
    total_prob = 0.0
    for a in ['ok', 'FAIL']:
        for b in ['ok', 'FAIL']:
            for c in ['ok', 'FAIL']:
                data.append({
                    'A': a,
                    'A prob': '%6.4f' % probs[a],
                    'B': b,
                    'B prob': '%6.4f' % probs[b],
                    'C': c,
                    'C prob': '%6.4f' % probs[c],
                    'Probability': '%6.4f' % (probs[a] * probs[b] * probs[c])
                })
                total_prob += probs[a] * probs[b] * probs[c]
    print_markdown_table(data, ['A', 'A prob', 'B', 'B prob', 'C', 'C prob', 'Probability'])
    print 'sum of probabilities: %6.4f' % total_prob
    print

    data = [
        {'Number of Failures': str(k), 'Probability': '%6.4f' % binomial_probability(k, 3, p_one_failing)}
        for k in xrange(4)
    ]
    print_markdown_table(data, ['Number of Failures', 'Probability'])


def main():
    if sys.argv[1:] == ['test']:
        del sys.argv[1]
        unittest.main()
    elif sys.argv[1:] == ['example']:
        example()
    else:
        parser = argparse.ArgumentParser()
        parser.add_argument('data_shards', type=int),
        parser.add_argument('parity_shards', type=int),
        parser.add_argument('annual_shard_failure_rate', type=float),
        parser.add_argument('shard_replacement_days', type=float)
        args = parser.parse_args()
        total_shards = args.data_shards + args.parity_shards
        min_shards = args.data_shards
        do_scenario(total_shards, min_shards, args.annual_shard_failure_rate, args.shard_replacement_days)


if __name__ == '__main__':
    main()
