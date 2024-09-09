"""
API Example 02: Custom meshing.

This example demonstrates how you can use neper_mosaic anf gmsh to control the
meshing of the rectilinear domain.


"""
import numpy as np
import gmsh
import neper_mosaic

# define input
input_file = "3d-n4-id1.geo"

# we have to initialize/finalize gmsh ourselves in this example!
gmsh.initialize()

# loads the Neper model, does the tiling and trimming
neper_mosaic.load_rectilinear_geometry(input_file)

# now we have the trimmed geometry in the gmsh occ kernel
print(f"Number of volumes: {len(gmsh.model.getEntities(dim=3))}")

# we create a mesh that is graded in x-direction
def mesh_size_callback(dim, tag, x, y, z, lc):
    # x ranges from 0 to 1, so sin(x*pi) ranges from 0 to 1
    return (np.sin(x*np.pi) + 0.2)*0.1

gmsh.model.mesh.setSizeCallback(mesh_size_callback)

gmsh.model.mesh.generate(3)

# open the gmsh gui to inspect the results
gmsh.option.setNumber("Mesh.SurfaceEdges", 0)
gmsh.option.setNumber("Mesh.SurfaceFaces", 0)

gmsh.option.setNumber("Mesh.VolumeEdges", 1)
gmsh.option.setNumber("Mesh.VolumeFaces", 1)

gmsh.fltk.run()

# remember, we have to finalize ourselves
gmsh.finalize()
