#!/usr/bin/env python3

import sys
import os, os.path
from optparse import OptionParser
import subprocess, shlex
import xml.etree.ElementTree as ET
import json


class ProgressBar(object):
    def __init__(self, total_ticks, message=''):
        self.total_ticks = total_ticks
        self.current_tick = 0
        self.bar_len = 60
        self.message = message

    def tick(self):
        if self.current_tick >= self.total_ticks:
            return

        self.current_tick += 1
        filled_len = (self.bar_len * self.current_tick) // self.total_ticks
        percents = round(100.0 * self.current_tick / self.total_ticks, 1)
        bar = '=' * filled_len + '-' * (self.bar_len - filled_len)
        sys.stdout.write('%s [%s] %s%s' % (self.message, bar, percents, '%'))
        if self.current_tick < self.total_ticks:
            sys.stdout.write('\r')
        else:
            sys.stdout.write('\n')
        sys.stdout.flush()


def build_tests():
    cmd = 'stack test'
    res = subprocess.run(shlex.split(cmd), check=True, stdout=subprocess.PIPE)


def get_project_root():
    cmd = 'stack path --project-root'
    res = subprocess.run(shlex.split(cmd), check=True, stdout=subprocess.PIPE)
    return res.stdout.decode('utf-8').strip()


def get_dist_dir():
    cmd = 'stack path --dist-dir'
    res = subprocess.run(shlex.split(cmd), check=True, stdout=subprocess.PIPE)
    dist_dir = res.stdout.decode('utf-8').strip()
    project_root = get_project_root()
    return os.path.normpath(os.path.join(project_root, dist_dir))


def make_unit_tests_command(dist_dir, name, jxml_path, test_args):
    return '{0}/build/{1}/{1} --jxml={2} {3}'.format(dist_dir, name, jxml_path, test_args)


def run_unit_tests(utcmd):
    args = shlex.split(utcmd)
    res = subprocess.run(args, stdout=subprocess.PIPE)
    return res.returncode


def parse_jxml(jxml_path):
    failures = []
    xml = ET.parse(jxml_path)
    for test_case in xml.getroot().findall('testcase'):
        for failure in test_case.findall('failure'):
            name = test_case.attrib['classname'] + '-' + test_case.attrib['name']
            failures.append((name, failure.text))
    return failures


def massive_test(utname, times, test_args):
    all_failures = []
    failure_freqs = {}
    dist_dir = get_dist_dir()
    jxml_path = '{0}.xml'.format(utname)
    utcmd = make_unit_tests_command(dist_dir, utname, jxml_path, test_args)
    print('running unit tests using command:', utcmd)
    pb = ProgressBar(times)
    for i in range(times):
        exit_code = run_unit_tests(utcmd)
        if exit_code == 1:
            failures = parse_jxml(jxml_path)
            for failure in failures:
                name = failure[0]
                freq = failure_freqs.get(name, 0)
                freq += 1
                failure_freqs[name] = freq
            all_failures += failures
        pb.tick()
    return failure_freqs, all_failures


def main():
    opt_parser = OptionParser('%prog -n 100 --tests-args="--select-tests=MachineLearning.Classification.OneVsAll*"')
    opt_parser.add_option('-n', '--times', dest='times', type='int', default=200)
    opt_parser.add_option('-t', '--tests-args', dest='test_args', default='')
    options, _ = opt_parser.parse_args()

    build_tests()

    freqs, failures = massive_test('mltool-test', options.times, options.test_args)
    for (name, freq) in sorted(freqs.items(), key=lambda p : p[1], reverse=True):
        print(name, freq)

    with open('result.json', 'w') as f:
        json.dump(failures, f, indent=2)


if __name__ == '__main__':
    main()
