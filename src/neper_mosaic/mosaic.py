import sys
import logging
from collections import defaultdict
from dataclasses import dataclass
from itertools import product, chain
from pathlib import Path

import gmsh
import numpy as np

from typing import Sequence, List


LOGGER = logging.getLogger("mosaic")
LOGGER.setLevel(logging.DEBUG)


@dataclass
class ModelGeometry():
    """
    Store the geometry of a model.

    Attributes
    ----------
    index : int
        Counter of the instance.
    dim : int
        Dimensionality.
    point_tags : ndarray[int]
        Tags for all points in the model.
    point_coords: ndarray[float]
        Coordinates in the same order as point_tags.
    line_tags: ndarray[int]
        Tags for all lines in the model.
    line_boundaries: list[ndarray[int]]
        Start and end point tag for each line.
    surface_tags: ndarray[int]
        Tags for all surfaces in the model.
    surface_boundaries: list[ndarray[int]]
        Tags of the lines forming the surface boundary.
    volume_tags: ndarray[int]
        Tags of all volumes in the model.
    volume_boundaries: list[ndarray[int]]
        Tags of the surfaces forming the volume boundary.

    """
    index: int
    dim: int
    point_tags: Sequence[int]
    point_coords: Sequence[Sequence[float]]
    line_tags: Sequence[int]
    line_boundaries: List[Sequence[int]]
    surface_tags: Sequence[int]
    surface_boundaries: List[Sequence[int]]
    volume_tags: Sequence[int]
    volume_boundaries: List[Sequence[int]]

    @classmethod
    def from_current_model(cls):
        """
        Create a class instance from the current model geometry.

        Returns
        -------
        ModelGeometry

        """
        index = 0

        dim = gmsh.model.getDimension()

        point_dimtags = gmsh.model.getEntities(dim=0)
        point_tags = np.array([tag for (_, tag) in point_dimtags])
        point_coords = np.array(
            [gmsh.model.getValue(0, tag, []) for tag in point_tags])

        line_tags = np.array(
            [tag for (dim, tag) in gmsh.model.getEntities(dim=1)])
        line_boundaries = [cls._get_boundary_tags(1, tag) for tag in line_tags]

        surface_tags = np.array(
            [tag for (dim, tag) in gmsh.model.getEntities(dim=2)])
        surface_boundaries = [cls._get_boundary_tags(
            2, tag) for tag in surface_tags]

        volume_tags = np.array(
            [tag for (dim, tag) in gmsh.model.getEntities(dim=3)])
        volume_boundaries = [cls._get_boundary_tags(
            3, tag) for tag in volume_tags]

        ret = cls(index,
                  dim,
                  point_tags,
                  point_coords,
                  line_tags,
                  line_boundaries,
                  surface_tags,
                  surface_boundaries,
                  volume_tags,
                  volume_boundaries)

        LOGGER.info("Created ModelGeometry from gmsh model: "
                    f"{ret.num_points} points, {ret.num_lines} lines, "
                    f"{ret.num_surfaces} surfaces, {ret.num_volumes} volumes."
                    )

        return ret

    def copy(self, index, dx=0.0, dy=0.0, dz=0.0):
        """
        Make a copy of the geometry. Index should be unique.

        Parameters
        ----------
        index: int
            Index of the new instance of the ModelGeometry. Should be unique.
        dx, dy, dz: float, optional
            Translate the new instance by these values. Default: 0.0

        Returns
        -------
        ModelGeometry

        """
        dim = self.dim

        point_tags = self.point_tags + index*self.num_points
        point_coords = self.point_coords.copy()
        point_coords[:, 0] += dx
        point_coords[:, 1] += dy
        point_coords[:, 2] += dz

        line_tags = self.line_tags + index*self.num_lines
        line_boundaries = [bounds + index*self.num_points
                           for bounds in self.line_boundaries]

        surface_tags = self.surface_tags + index*self.num_surfaces
        # this assumes that the line tags on the surface boundary are positive!
        surface_boundaries = [bounds + index*self.num_lines
                              for bounds in self.surface_boundaries]

        volume_tags = self.volume_tags + index*self.num_volumes
        volume_boundaries = [bounds + np.sign(bounds)*index*self.num_surfaces
                             for bounds in self.volume_boundaries]

        return type(self)(index,
                          dim,
                          point_tags,
                          point_coords,
                          line_tags,
                          line_boundaries,
                          surface_tags,
                          surface_boundaries,
                          volume_tags,
                          volume_boundaries)

    def to_occ(self):
        """
        Load the ModelGeometry in the gmsh occ kernel.

        Returns
        -------
        None

        """

        for tag, (x, y, z) in zip(self.point_tags, self.point_coords):
            gmsh.model.occ.addPoint(x, y, z, tag=tag)

        for tag, bpoints in zip(self.line_tags, self.line_boundaries):
            gmsh.model.occ.addLine(*bpoints, tag=tag)

        for tag, blines in zip(self.surface_tags, self.surface_boundaries):
            sorted_blines = self.sort_curve_loop(
                blines, self.line_tags, self.line_boundaries)
            cloop_tag = gmsh.model.occ.addCurveLoop(sorted_blines, tag=tag)
            gmsh.model.occ.addPlaneSurface([cloop_tag], tag=tag)

        for tag, bsurfs in zip(self.volume_tags, self.volume_boundaries):
            vloop_tag = gmsh.model.occ.addSurfaceLoop(bsurfs, tag=tag)
            gmsh.model.occ.addVolume([vloop_tag], tag=tag)

        gmsh.model.occ.synchronize()

        highest_dim_tags = self.volume_tags if self.dim == 3 else self.surface_tags
        num_entities = len(highest_dim_tags)

        for tag in highest_dim_tags:
            group_tag = tag + self.index*num_entities

            existing_groups = [
                tag for (dim, tag) in gmsh.model.getPhysicalGroups(dim=self.dim)]

            if group_tag in existing_groups:
                tags = gmsh.model.getEntitiesForPhysicalGroup(
                    self.dim, group_tag)
                gmsh.model.removePhysicalGroups([(self.dim, group_tag)])
                updated_tags = np.hstack((tags, tag))
            else:
                updated_tags = [tag]

            gmsh.model.addPhysicalGroup(self.dim, updated_tags, group_tag)

        gmsh.model.occ.synchronize()

    @staticmethod
    def sort_curve_loop(line_tags, all_line_tags, all_line_boundaries):
        """
        Return line tags in a curve loop sorted and with direction sign.

        Parameters
        ----------
        line_tags: Sequence[int]
            Tags of the lines in the curve loop (not oriented).
        all_line_tags: Sequence[int]
            Tags of all lines in the model.
        all_line_boundaries: Sequence[Sequence[int]]
            Boundary points of all lines in the model, describing line
            orientation.

        Returns
        -------
        list[list[int]]
            Tags of the lines in the curve loop, with a minus when they need to
            change orientation.

        """
        # indices to reduce the global arrays
        indices = [i for i, tag in enumerate(
            all_line_tags) if tag in line_tags]

        red_line_tags = [all_line_tags[i] for i in indices]
        red_line_boundaries = [all_line_boundaries[i] for i in indices]

        # create mapping line tag -> point tags
        points_from_line = dict(zip(red_line_tags, red_line_boundaries))

        # create mapping point tag -> directed line tags
        lines_from_point = defaultdict(list)

        for ltag, (p1, p2) in zip(red_line_tags, red_line_boundaries):
            lines_from_point[p1].append(ltag)
            lines_from_point[p2].append(ltag)

        line = line_tags[0]

        sorted_lines = [line]
        (p1, p2) = points_from_line.pop(line)
        p0 = p2

        while len(points_from_line) > 0:
            line_candidates = lines_from_point[p0]

            for line in line_candidates:
                if line in points_from_line:
                    (p1, p2) = points_from_line.pop(line)
                    if p1 == p0:
                        sorted_lines.append(line)
                        p0 = p2
                    else:
                        sorted_lines.append(-line)
                        p0 = p1

                    break

        return sorted_lines

    @property
    def num_points(self):
        """Number of points in the model."""
        return len(self.point_tags)

    @property
    def num_lines(self):
        """Number of lines in the model."""
        return len(self.line_tags)

    @property
    def num_surfaces(self):
        """Number of surfaces in the model."""
        return len(self.surface_tags)

    @property
    def num_volumes(self):
        """Number of volumes in the model."""
        return len(self.volume_tags)

    @staticmethod
    def _get_boundary_tags(dim, tag):
        """
        Get the tags of dimension dim-1 for the boundary of an entity.

        For example, get the tags of all lines on the boundary of a surface.

        Parameters
        ----------
        dim: int
            Dimension of the entity.
        tag: int
            Tag of the entity.

        Returns
        -------
        np.ndarray
            Tags of the entities on the boundary.

        """
        boundary_dimtags = gmsh.model.getBoundary([(dim, tag)], oriented=False)
        return np.array([btag for (_, btag) in boundary_dimtags])

    def compute_domain_size(self, tol=1e-8):
        """
        Calculate the domain size in x,y,z directions, assuming periodicity.

        Parameters
        ----------
        tol: float, optional
            Tolerance to assume two points are on the same line. Default: 1e-8.

        Returns
        -------
        Tuple[float]
            Domain size in each coordinate direction.

        """
        x, y, z = self.point_coords.T

        rx = self._get_coordinate_width(x, y, z, tol=tol)
        ry = self._get_coordinate_width(y, z, x, tol=tol)
        rz = self._get_coordinate_width(z, x, y, tol=tol)

        return rx, ry, rz

    @staticmethod
    def _get_coordinate_width(x, y, z, tol=1e-8):
        """
        Find the point with the smallest x coordinate. Find all point with the
        same y and z coordinates and calculate the maximum distance in x
        direction.

        Parameters
        ----------
        x, y, z: np.ndarray
            Coordinates in all three directions.
        tol: float = 1e-8
            Tolerance to assume points have the same y and z coordinate.

        Returns
        -------
        float
            The maximum distance in x direction.

        """
        i_xmin = np.argmin(x)
        xmin = x[i_xmin]
        selector_x = (np.abs(y - y[i_xmin]) < tol) * \
            (np.abs(z - z[i_xmin]) < tol)
        xmax = x[selector_x].max()

        return xmax - xmin


def setup_logger(verbose):
    """
    Configure the logging to stdout/stderr.

    Parameters
    ----------
    verbose : bool
        Whether to output info level logging messages.

    Returns
    -------
    None

    """
    # we just update the module constant LOGGER to the correct settings here
    stdout_level = logging.INFO if verbose else logging.WARNING

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(stdout_level)

    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setLevel(logging.ERROR)

    # errors should be excluded from stdout
    stdout_handler.addFilter(lambda record: record.levelno < logging.ERROR)

    LOGGER.addHandler(stdout_handler)
    LOGGER.addHandler(stderr_handler)
    
    
def load_neper_geometry(input_file):
    """
    Load the geometry defined by Neper from a .geo file.
    
    Gmsh must be initialized before calling this function. The Neper geometry
    is created in a separate gmsh model, which is deleted after reading the
    geometry.
    
    Parameters
    ----------
    input_file : str
        Path to the .geo file generated by Neper.

    Returns
    -------
    ModelGeometry

    """
    if not Path(input_file).is_file():
        raise FileNotFoundError(f"Input file {input_file} not found.")

    gmsh.model.add("neper_import")
    gmsh.model.setCurrent("neper_import")
    
    gmsh.open(input_file)
    LOGGER.info(f"Neper file '{input_file}' imported in geo kernel.")

    mg = ModelGeometry.from_current_model()
    LOGGER.info("Extracted ModelGeometry from geo kernel.")
    
    gmsh.model.remove()
    
    return mg
    

def create_geometry_tiling(mg):
    """
    Create 3x3 or 3x3x3 tiling from a periodic ModelGeometry.
    
    Gmsh must be initialized before calling this function. The tiling is
    performed in the current gmsh model using the occ kernel.

    Parameters
    ----------
    mg : ModelGeometry
        Periodic geometry loaded from a Neper-generated .geo file.

    Returns
    -------
    None.

    """
    # size of the periodic domain in x, y and z direction
    rx, ry, rz = mg.compute_domain_size()

    # create list of copy placements
    if mg.dim == 3:
        translations = product((0, -rx, rx), (0, -ry, ry), (0, -rz, rz))
    else:
        translations = product((0, -rx, rx), (0, -ry, ry))

    # create copies in x, y (z) directions
    for index, translation in enumerate(translations):
        if mg.dim == 3:
            dx, dy, dz = translation
        else:
            dx, dy = translation
            dz = 0

        mg.copy(index=index, dx=dx, dy=dy, dz=dz).to_occ()
        LOGGER.info(f"Loaded model copy with index {index}.")
        

def trim_to_rectilinear_domain(dim, rx, ry, rz, x0=0.0, y0=0.0, z0=0.0):
    """
    Trim the model to a rectangle or cube.

    Parameters
    ----------
    dim : int
        Dimension of the model.
    rx, ry, rz : float
        Domain size in x, y, z direction.
    x0, y0, z0 : float, optional
        Origin of the rectangle/cube used to trim the domain. Default: 0.0.
        
    Returns
    -------
    None

    """
    if dim == 3:
        cut_geo = gmsh.model.occ.addBox(x0, y0, z0, rx, ry, rz)
    else:
        cut_geo = gmsh.model.occ.addRectangle(x0, y0, 0, rx, ry)

    gmsh.model.occ.synchronize()

    # store "before" info for physical groups:
    group_tags = [tag for (dim, tag) in gmsh.model.getPhysicalGroups(dim=dim)]
    group_entities = {tag: gmsh.model.getEntitiesForPhysicalGroup(dim, tag)
                      for tag in group_tags}

    # get a list of all entities that will be intersected with the unit box
    objects = [(dim, tag) for (dim, tag) in gmsh.model.getEntities(dim)
               if tag != cut_geo]

    # perform the intersection
    new_entities, entity_children = gmsh.model.occ.intersect(objects,
                                                             [(dim, cut_geo)],
                                                             removeTool=True)

    # deal with douplicates of boundaries etc
    gmsh.model.occ.removeAllDuplicates()
    
    gmsh.model.occ.synchronize()

    masses = [gmsh.model.occ.getMass(dim, tag)
              for (dim, tag) in gmsh.model.getEntities(dim=dim)]
    
    grain_mass = sum(masses)
    
    box_mass = rx*ry if dim == 2 else rx*ry*rz
    
    if not np.isclose(grain_mass, box_mass):
        if all([x0==0.0, y0==0.0, z0==0.0]):
            msg = "Model volume after trimming is wrong."
        else:
            msg = ("Model volume after trimming is wrong. "
                   "Box position might be too far from origin.")
            
        LOGGER.error(msg)
        raise RuntimeError(msg)
            

    # delete all the old physical groups (because we cant update them)
    gmsh.model.removePhysicalGroups()

    # mapping of original dimtags to children after intersection
    entity_mapping = dict(zip(objects, entity_children))

    # recreate each physical group, but with the new entity tags
    for group_tag, old_entity_dimtags in group_entities.items():
        # list of (new) dimtags for this physical group
        new_entity_dimtags = list(chain(*[entity_mapping[(dim, etag)]
                                          for etag in old_entity_dimtags]))

        # extract only the tags
        new_entity_tags = [tag for (_, tag) in new_entity_dimtags]

        # recreate the group
        gmsh.model.addPhysicalGroup(dim, new_entity_tags, tag=group_tag)

    edge_lengths = [rx, ry, rz][0:dim]
    LOGGER.info(f"Trimmed model to {dim}d domain with edge lengths "
                f"{' x '.join([str(l) for l in edge_lengths])}.")


def load_rectilinear_geometry(input_file, x0=0.0, y0=0.0, z0=0.0):
    """
    Create the rectilinear geometry for a Neper-created .geo-file.
    
    Gmsh must be initialized before calling this function. The geometry is
    created in the gmsh occ kernel.

    Parameters
    ----------
    input_file : str
        Path to the .geo file generated by Neper.
    x0, y0, z0 : float, optional
        Origin of the rectangle/cube used to trim the domain. Default: 0.0.

    Returns
    -------
    None.

    """
    model_geometry = load_neper_geometry(input_file)
    
    create_geometry_tiling(model_geometry)
    
    rx, ry, rz = model_geometry.compute_domain_size()
    trim_to_rectilinear_domain(model_geometry.dim,
                               rx,
                               ry,
                               rz,
                               x0=x0,
                               y0=y0,
                               z0=z0)
    

def main(input_file,
         output_files,
         element_size=0.1,
         box_x_position=0.0,
         box_y_position=0.0,
         box_z_position=0.0,
         verbose=False,
         show_gui=False,
         show_gmsh_output=False,
         ):
    """
    Run the main mosaic routine.

    This is what is executed when mosaic is called from the command line.

    Parameters
    ----------
    input_file : str
        The .geo file generated by Neper.
    output_files : str
        A list of output files. If only one output file is requested, it still
        needs to be passed in a list.
    element_size : float, optional
        Target characteristic length for elements. Default: 0.1.
    box_x_position : float, optional
        Origin of the rectilinear domain in x direction.
    box_y_position : float, optional
        Origin of the rectilinear domain in y direction.
    box_z_position : float, optional
        Origin of the rectilinear domain in z direction.
    verbose : bool, optional
        Whether to output additional logging information. Default: False.
    show_gui : bool, optional
        Whether to show the gmsh GUI after meshing. Default: False.
    show_gmsh_output : bool, optional
        Whether to show (extensive) logging information from gmsh.
        Default: False.

    Returns
    -------
    None

    """
    setup_logger(verbose)

    LOGGER.info("This is Mosaic.py :-)")

    LOGGER.info(f"input file: {input_file}")
    LOGGER.info(f"output files: {output_files}")
    
    box_pos = (box_x_position, box_y_position, box_z_position)
    if not box_pos == (0.0, 0.0, 0.0):
        LOGGER.info(f"position of rectilinear domain: ({box_pos})")
        
    LOGGER.info(f"target element size: {element_size}")

    if gmsh.isInitialized():
        gmsh.finalize()


    gmsh.initialize()

    gmsh.option.setNumber("General.Terminal", int(show_gmsh_output))

    load_rectilinear_geometry(input_file,
                              x0=box_x_position,
                              y0=box_y_position,
                              z0=box_z_position)

    dim = gmsh.model.getDimension()
    
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)

    gmsh.option.setNumber("Mesh.MeshSizeMin", element_size)
    gmsh.option.setNumber("Mesh.MeshSizeMax", element_size)
    
    # gmsh.option.setNumber("Mesh.Algorithm", 2)
    # gmsh.option.setNumber("Mesh.Algorithm3D", 1)

    gmsh.model.mesh.generate(dim)

    for file in output_files:
        # make sure the output directory exists
        file_path = Path(file)
        file_path.resolve().parent.mkdir(exist_ok=True)
        
        gmsh.write(str(file))

    if show_gui:
        if dim == 3:
            gmsh.option.setNumber("Mesh.SurfaceEdges", 0)
            gmsh.option.setNumber("Mesh.SurfaceFaces", 0)

            gmsh.option.setNumber("Mesh.VolumeEdges", 1)
            gmsh.option.setNumber("Mesh.VolumeFaces", 1)
        else:
            gmsh.option.setNumber("Mesh.SurfaceEdges", 1)
            gmsh.option.setNumber("Mesh.SurfaceFaces", 1)

            gmsh.option.setNumber("Mesh.VolumeEdges", 0)
            gmsh.option.setNumber("Mesh.VolumeFaces", 0)

        gmsh.option.setNumber("Mesh.ColorCarousel", 2)

        gmsh.fltk.run()

    # HACK: not finalizing avoids python crash...
    gmsh.finalize()
