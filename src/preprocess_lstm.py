import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s','--supervised')
    parser.add_argument()
    return parser.parse_args()
