"""
API Example 01: Standard mosaic call from python scripts.

This example demonstrates how to use neper_mosaic from a python script. The
function neper_mosaic.main() behaves exactly like the command line interface.
A typical use case would be the automated conversion of a larger number of
Neper-created .geo-files to rectilinear domains.

You can see the detailed parameter description by calling
>>> help(neper_mosaic.main)

"""
import neper_mosaic

# define input and output files
input_file = "n4-id1.geo"
output_files = ["output/api_example_01.msh"]


neper_mosaic.main(input_file,
                  output_files,
                  element_size=0.05,
                  verbose=True, # show information in the terminal
                  show_gui=True, # open the gmsh gui after import
                  )
