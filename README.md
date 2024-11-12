# Mosaic

## Publication
D. Güzel, T. Furlan, T. Kaiser, A. Menzel:
Neper-Mosaic: Seamless Generation of periodic representative volume elements on unit domains.
SoftwareX , 28, 101912, 2024,
DOI: [10.1016/j.softx.2024.101912](https://www.sciencedirect.com/science/article/pii/S2352711024002826), Open Access.

## Introduction

Mosaic is an open-source Python software tool designed to integrate non-rectilinear, periodic microstructures generated by Neper software into simulations that require periodic microstructures with periodic boundary conditions. It transforms complex microstructures into rectilinear periodic equivalents. This transformation ensures seamless integration with various simulation tools and workflows.


## Installation

The source code for Mosaic is available from our [github repository](https://www.github.com). After acceptance of the submission, it will also be made available through pip. Pull the repository and activate your conda environment, once your environment is active, switch to the repository root directory and run (type the dot and the space):

    pip install .


## Usage

Mosaic can be used from the command line or imported as a Python module to integrate it into your own projects.

### Command-line usage

To execute Mosaic from the command line, use the following syntax:

    python3 -m neper_mosaic inputfile.geo outputfile1 [outputfile2, ...]

Note that multiple output files can be specified for one input, but at least one must be given.


The following options are available:

| option                 | shorthand  | explanation                                        |
|------------------------|------------|----------------------------------------------------|
| `--help`               | `-h`       | show help message and options                      |
| `--verbose`            | `-v`       | increase output verbosity                          |
| `--show_gui`           |            | open gmsh GUI after importing the model            |
| `--show_gmsh`          |            | display gmsh output in the terminal                |
| `--ciGen`              |            | store .msh files in a format compatible with ciGen |
| `--box_x_position`     | `-x0`      | x-translation of the rectilinear domain            |
| `--box_y_position`     | `-y0`      | y-translation of the rectilinear domain            |
| `--box_z_position`     | `-z0`      | z-translation of the rectilinear domain            |
| `--element_size`       | `-cl`      | Characteristic element length                      |


### API usage

You can use all functionalities of Mosaic in your own python scripts through the mosaic API. The syntax looks as follows:

```python
import neper_mosaic as mosaic

mosaic.main(inputfile, outputfiles)
```

The `main` function has the same optional arguments as described in the table above but does not accept shorthands.

If you want more control over the imported model, you can also mix the usage of the mosaic and gmsh APIs:

```python
import gmsh
from neper_mosaic import load_rectilinear_geometry

# in this case, gmsh has to be initialized manually
gmsh.initialize()

# use mosaic to load a model created by Neper into OCC and create a rectilinear version
load_rectilinear_domain("input_file.geo")

# then we can use the gmsh API or even the GUI to manipulate the result
gmsh.fltk.run()

# in this case, we have to implement saving etc ourselves...

# at the end of the program, finalize gmsh
gmsh.finalize()

```

The examples folder of the Mosaic repository contains small examples for both API use cases.

### Examples

Mosaic comes in some examples to demonstrate the usage. Check them out in the examples folder!

## Tests

Mosaic comes with a few tests for pytest, which can be found in the directory `Test`.

## Support

If you encounter any bugs with Mosaic, please open an issue in our github repository!


## Contributing

If you implement any improvements to Mosaic, you are welcome to create a pull request in our repository.

## Authors

Mosaic was created at the Institute of Mechanics, TU Dortmund University, Germany, by Dilek Güzel, Tim Furlan, Tobias Kaiser, and Andreas Menzel.

## License

GPL-3.0 license

