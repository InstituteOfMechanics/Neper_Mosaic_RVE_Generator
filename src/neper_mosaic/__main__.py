"""
Entry point for command line usage of mosaic.

"""
import sys
from .mosaic import main
from .parser import parse_args

# argv[0] contains __main__.py path
args = parse_args(sys.argv[1:])

main(**args)
