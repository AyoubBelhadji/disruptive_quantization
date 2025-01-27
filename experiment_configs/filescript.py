#!/usr/bin/env python3

import os
import argparse

# Read from a set of directories starting with a prefix
# For every directory, take all files in it and move any containing
# a certain string to a new directory. Prefix all these files with
# the suffix of the old directory name.
def main(prefix, new_directory, substrs):
    if not os.path.exists(new_directory):
        os.mkdir(new_directory)
    for dirname in os.listdir("."):
        if dirname.startswith(prefix):
            suffix = dirname[len(prefix):]
            for filename in os.listdir(dirname):
                # if all substrs in filename
                if all(substr in filename for substr in substrs):
                    newname = suffix + "_" + filename
                    # move the file
                    os.rename(os.path.join(dirname, filename), os.path.join(new_directory, newname))

# Parse arguments
parser = argparse.ArgumentParser(description='Move files containing certain substrings to a new directory.')
parser.add_argument('prefix', type=str, help='Prefix of directories to search for files.')
parser.add_argument('new_directory', type=str, help='Directory to move files to.')
parser.add_argument('substrs', type=str, nargs='+', help='Substrings to search for in filenames.')

if __name__ == '__main__':
    args = parser.parse_args()
    main(args.prefix, args.new_directory, args.substrs)