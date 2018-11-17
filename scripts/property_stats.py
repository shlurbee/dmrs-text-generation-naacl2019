#!/usr/bin/env python3

import argparse

from delphin import itsdb
from delphin.mrs import simplemrs
from delphin.mrs.components import var_sort

E_PROPERTIES = [
    ('PERF', ['bool', '+', '-']),
    ('PROG', ['bool', '+', '-']),
    ('MOOD', ['mood', 'indicative', 'subjunctive']),
    ('TENSE', ['tense', 'tensed', 'past', 'pres', 'fut', 'untensed']),
    ('SF', ['sf', 'prop-or-ques', 'prop', 'ques', 'comm'])
]
X_PROPERTIES = [
    ('PERS', ['person', '1', '2', '3']),
    ('NUM', ['number', 'sg', 'pl']),
    ('GEND', ['gender', 'm-or-f', 'm', 'f', 'n']),
    ('IND', ['bool', '+', '-']),
    ('PT', ['pt', 'refl', 'std', 'zero']),
]


def _make_counters():
    make = lambda d: {prop: {val: 0 for val in vals} for prop, vals in d}
    return {
        'e': make(E_PROPERTIES),
        'x': make(X_PROPERTIES),
        'i': make(X_PROPERTIES)
    }


def report(sums):
    for vartype, props in sums.items():
        for prop, vals in props.items():
            prop_count = sum(vals.values())
            if prop_count == 0:
                print('{:<4}\t{:<10}\t{:<12}\t{:>8}\t{:>10}'
                      .format(vartype, prop, '--', '--', '--'))
                continue
            for val, count in vals.items():
                prop_pct = count / float(prop_count)
                print('{:<4}\t{:<10}\t{:<12}\t{:8d}\t{:.8f}'
                      .format(vartype, prop, val, count, prop_pct))
    print()


def main(args):
    total_sums = _make_counters()
    total_record_count = 0

    for profile in args.PROFILE:
        ts = itsdb.TestSuite(profile)
        sums = _make_counters()
        record_count = 0

        for record in ts['result']:
            record_count += 1
            total_record_count += 1
            mrs = simplemrs.loads_one(record['mrs'])

            for var in mrs.variables():
                vartype = var_sort(var)

                for prop, val in mrs.properties(var).items():
                    sums[vartype][prop.upper()][val.lower()] += 1

        print('{} ({} MRSs):'.format(profile, record_count))
        report(sums)

        for vartype, props in sums.items():
            for prop, vals in props.items():
                for val, count in vals.items():
                    total_sums[vartype][prop][val] += count

    print('TOTAL ({} MRSs):'.format(total_record_count))
    report(total_sums)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Compute morphosemantic property statistics')
    parser.add_argument('PROFILE', nargs='+',
                        help='profile to compute statistics over')
    args = parser.parse_args()
    main(args)
