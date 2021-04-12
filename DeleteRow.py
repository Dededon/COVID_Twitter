import csv
import os
import argparse
import pandas as pd

if __name__ == '__main__':
    sourcefile = "data\\tweets\\out_10039.csv"
    outputfile = "data\\tweets\\out_10039_del.csv"
    inp = pd.read_csv(sourcefile, error_bad_lines=False)
    # inp = open(sourcefile, 'rt')
    # output = open(outputfile, 'wt')
    # writer = csv.writer(output)
    # for row in csv.reader(inp):
    #     if int(row[12]) == 0:
    #         writer.writerow(row)
    #
    # inp.close()
    # output.close()
