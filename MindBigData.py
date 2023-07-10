#!/usr/bin/env python

import mmap
import sys
import re
import pandas as pd
import pickle
import os

def main():
    if len(sys.argv) != 2:
        print(f'Usage: ./{os.path.basename(__file__)} <data file>.txt')
        return 1

    column_regex = re.compile(r'[\t \r\n]+')

    with open(sys.argv[1], 'r') as f:
        # read a line from f
        line = f.readline() 
        while line:
            id, event, device, channel, code, _, data, *_ = column_regex.split(line)
            data = data.split(',')
            data = tuple(map(int, data))
            dir = f'./{device}/{code}/{event}'
            if not os.path.exists(dir):
                os.makedirs(dir)
            pickle.dump(data, open(f'{dir}/{channel}.pkl', 'wb'))
            line = f.readline()

    return 0

if __name__ == '__main__':
    exit(main())
