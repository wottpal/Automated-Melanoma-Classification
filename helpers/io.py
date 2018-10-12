# -*- coding: utf-8 -*-
"""Helper Functions for generating and retrieving file or directory names."""

import __main__ as main
import os
import sty
import shutil
import argparse
import time
import functools



def next_path(basedir, filename = None, create_dirs = False, append = False):
    """TOOD: Returns a path for given parameters.
    
    Arguments:
        basedir {string} -- Basepath (e.g. `./logs`)
    
    Keyword Arguments:
        filename {string} -- Filename (default: {None})
        append {bool} -- Set to True if dirname should not be iterated (default: {False})
    
    Returns:
        string -- path (filepath if `filename` given, dirpath otherwise)
    """


    try:
        dirs = list(os.walk(basedir))[0][1]
        dirs.sort()
    except:
        dirs = []

    timestr = time.strftime("%Y%m%d-%H%M%S")
    if append and len(dirs): timestr = dirs[-1]
    dir = os.path.join(basedir, timestr)
    # idx = max(1, len(dirs) + 1 if not append else 0)
    # dir += '-{0:03d}'.format(idx)

    if create_dirs and not os.path.exists(dir):
        os.makedirs(dir)

    dir = dir if not filename else os.path.join(dir, filename)
    return dir


def flatten(*args):
    """https://stackoverflow.com/a/6936753/1381666"""

    output = []
    for arg in args:
        if isinstance(arg, (list, tuple)):
            output.extend(flatten(*list(arg)))
        else:
            output.append(arg)
    return output


def clear_directories(*args):
    """Empties the given directory paths. 

    IMPORTANT: This is a destructive task and can't be undone.
    """

    dirs = flatten(args)
    for dir in dirs:
        if os.path.isdir(dir): shutil.rmtree(dir)
        if not os.path.exists(dir): os.makedirs(dir)


def scan_directory(dir):
    """Prints a hierarchical subtree of the given directory with the amount of files for each directory.
    
    Arguments:
        dir {string} -- Path to the directory which should be scanned
    """

    if not os.path.isdir(dir): 
        raise ValueError(f'Given path {dir} does not exist.')
    
    for root, _, files in os.walk(dir):
        level = root.replace(dir, '').count(os.sep)
        indent = ' ' * 3 * (level)
        files = [f for f in files if f not in ['.DS_Store']]
        path_str = os.path.basename(root) if level > 0 else root
        files_str = ""
        if len(files): files_str = sty.ef.dim + f'({len(files)})' + sty.rs.all
        print(f'{indent}{path_str}/ {files_str}')


def get_file_with_parents(filepath, levels=1):
    """Returns the filename and it's parent directory (https://www.saltycrane.com/blog/2011/12/how-get-filename-and-its-parent-directory-python/)"""
    common = filepath
    for i in range(levels + 1):
        common = os.path.dirname(common)
    return os.path.relpath(filepath, common)


def subdirs_and_files(path, verbose = 0):
    """Returns the the folder-names of all subdirs and an array of their files."""
    classes = []
    files = []

    if os.path.isdir(path): _, classes, _ = list(os.walk(path))[0]
    for class_name in classes:
        class_path = os.path.join(path, class_name)
        _, _, class_files = list(os.walk(class_path))[0]
        files.append(class_files)
        if verbose: print(f"Subdir '{class_name}' has {len(class_files)} files")
    
    return (classes, files)


def latest_files(search_dir, search_extensions=None, limit=1):
    """Returns a list of most recently modified files with given `search_extensions` under given `search_dir`"""
    print(f"\nDetermine latest files ending with '{search_extensions}'..")
    filepaths = []
    for root, _, files in os.walk(search_dir):
        filepaths += [os.path.join(root, x) for x in files]

    def compare_modified_date(f1, f2):
        return os.path.getmtime(f2) - os.path.getmtime(f1)
    filepaths = sorted(filepaths, key=functools.cmp_to_key(compare_modified_date))

    latest_files = []
    for filepath in filepaths:
        _, filename = os.path.split(filepath)
        if search_extensions is not None and not filename.lower().endswith(search_extensions): continue
        if len(latest_files) >= limit: break
        latest_files.append(filepath)

    if not latest_files:
        raise ValueError(f"Couldn't find any matching files under '{search_dir}'")

    return latest_files



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--scan', type=str)
    group.add_argument('--clear', nargs='+', type=str)
    args = parser.parse_args()

    if args.scan: scan_directory(args.scan)
    if args.clear: clear_directories(args.clear)
