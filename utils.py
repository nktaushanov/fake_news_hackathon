import io
import os
import pickle

from urlparse import urlparse

def print_unicode(obj):
    """Print unicode object represenation."""
    print repr(obj).decode('unicode-escape')

def get_project_file_path(*args):
    """Get filepath relative to the project home directory."""
    current_dir = os.path.dirname(os.path.realpath(__file__))
    return apply(os.path.join, [current_dir] + list(args))

def flatten(arr):
    result = []

    for x in arr:
        if isinstance(x, list):
            result.extend(flatten(x))
        else:
            result.append(x)

    return result

def get_domain(url):
    return urlparse(url).netloc

