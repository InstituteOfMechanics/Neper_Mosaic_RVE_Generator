#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
#   Mosaic - Periodic RVE Generator on Unit Domain
#
#   Copyright (C) 2024, Dilek Güzel, Tim Furlan, Tobias Kaiser and Andreas Menzel, Institute of Mechanics, TU Dortmund University, Germany.
#
#   This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
#
#   You should have received a copy of the GNU General Public License along with this program. If not, see https://www.gnu.org/licenses/.

import sys
import argparse
import logging
from collections import defaultdict
from dataclasses import dataclass
from itertools import product, chain

from typing import Sequence, List, Tuple, Dict
from numpy.typing import ArrayLike

import gmsh
import numpy as np

LOGGER = logging.getLogger("mosaic")
LOGGER.setLevel(logging.DEBUG)

@dataclass
class ModelGeometry():
    """Store the geometry of a model."""
    index: int
    dim: int
    point_tags: Sequence[int]
    point_coords: ArrayLike
    line_tags: Sequence[int]
    line_boundaries: ArrayLike
    surface_tags: Sequence[int]
    surface_boundaries: ArrayLike
    volume_tags: Sequence[int]
    volume_boundaries: ArrayLike

    @classmethod
    def from_current_model(cls) -> "ModelGeometry":
        """Create a class instance from the current model geometry."""
        index = 0

        dim = gmsh.model.getDimension()

        point_dimtags = gmsh.model.getEntities(dim=0)
        point_tags = np.array([tag for (_, tag) in point_dimtags])
        point_coords = np.array([gmsh.model.getValue(0, tag, []) for tag in point_tags])

        line_tags = np.array([tag for (dim, tag) in gmsh.model.getEntities(dim=1)])
        line_boundaries = [cls._get_boundary_tags(1, tag) for tag in line_tags]

        surface_tags = np.array([tag for (dim, tag) in gmsh.model.getEntities(dim=2)])
        surface_boundaries = [cls._get_boundary_tags(2, tag) for tag in surface_tags]

        volume_tags = np.array([tag for (dim, tag) in gmsh.model.getEntities(dim=3)])
        volume_boundaries = [cls._get_boundary_tags(3, tag) for tag in volume_tags]

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


    def copy(self,
             index: int,
             dx: float=0.0,
             dy: float=0.0,
             dz: float=0.0,
             ) -> "ModelGeometry":
        """Make a copy of the geometry. Index should be unique."""
        dim = self.dim

        point_tags = self.point_tags + index*self.num_points
        point_coords = self.point_coords.copy()
        point_coords[:,0] += dx
        point_coords[:,1] += dy
        point_coords[:,2] += dz

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


    def to_occ(self) -> None:
        """Load the model geometry in the gmsh occ kernel."""

        for tag, (x,y,z) in zip(self.point_tags, self.point_coords):
            gmsh.model.occ.addPoint(x, y, z, tag=tag)

        for tag, bpoints in zip(self.line_tags, self.line_boundaries):
            gmsh.model.occ.addLine(*bpoints, tag=tag)

        for tag, blines in zip(self.surface_tags, self.surface_boundaries):
            sorted_blines = self.sort_curve_loop(blines, self.line_tags, self.line_boundaries)
            cloop_tag = gmsh.model.occ.addCurveLoop(sorted_blines, tag=tag)
            gmsh.model.occ.addPlaneSurface([cloop_tag], tag=tag)

        for tag, bsurfs in zip(self.volume_tags, self.volume_boundaries):
            vloop_tag = gmsh.model.occ.addSurfaceLoop(bsurfs, tag=tag)
            gmsh.model.occ.addVolume([vloop_tag], tag=tag)
            
        gmsh.model.occ.synchronize()
            
        highest_dim_tags = self.volume_tags if self.dim == 3 else self.surface_tags
        num_entities = len(highest_dim_tags)
        
        for tag in highest_dim_tags:
            group_tag = tag - self.index*num_entities
            
            existing_groups = [tag for (dim, tag) in gmsh.model.getPhysicalGroups(dim=self.dim)]
            
            if group_tag in existing_groups:
                tags = gmsh.model.getEntitiesForPhysicalGroup(self.dim, group_tag)
                gmsh.model.removePhysicalGroups([(self.dim, group_tag)])            
                updated_tags = np.hstack((tags, tag))
            else:
                updated_tags = [tag]
            
            gmsh.model.addPhysicalGroup(self.dim, updated_tags, group_tag)

        gmsh.model.occ.synchronize()


    @staticmethod
    def sort_curve_loop(line_tags: Sequence[int],
                        all_line_tags: Sequence[int],
                        all_line_boundaries: Sequence[Sequence[int]],
                        ) -> List[List[int]]:
        """Return line tags in a curve loop sorted and with direction sign."""
        # indices to reduce the global arrays
        indices = [i for i, tag in enumerate(all_line_tags) if tag in line_tags]

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
    def num_points(self) -> int:
        return len(self.point_tags)

    @property
    def num_lines(self) -> int:
        return len(self.line_tags)

    @property
    def num_surfaces(self) -> int:
        return len(self.surface_tags)

    @property
    def num_volumes(self) -> int:
        return len(self.volume_tags)

    @staticmethod
    def _get_boundary_tags(dim: int, tag: int) -> np.ndarray:
        """Get the tags of dimension dim-1 for the boundary of an entity."""
        boundary_dimtags = gmsh.model.getBoundary([(dim, tag)], oriented=False)
        return np.array([btag for (_, btag) in boundary_dimtags])

    def compute_domain_size(self, tol: float = 1e-8) -> Tuple[float]:
        """
        Calculate the domain size in x,y,z directions, assuming periodicity.
        
        """
        x, y, z = self.point_coords.T

        rx = self._get_coordinate_width(x, y, z, tol=tol)
        ry = self._get_coordinate_width(y, z, x, tol=tol)
        rz = self._get_coordinate_width(z, x, y, tol=tol)

        return (rx, ry, rz)


    @staticmethod
    def _get_coordinate_width(x: float,
                              y: float,
                              z: float,
                              tol: float = 1e-8,
                              ) -> float:
        """
        Find points on opposing x sides of the rve and calculate their distance.
        """
        i_xmin = np.argmin(x)
        xmin = x[i_xmin]
        selector_x = (np.abs(y - y[i_xmin]) < tol) * (np.abs(z - z[i_xmin]) < tol)
        xmax = x[selector_x].max()

        return xmax - xmin


def parse_args(argstr: str = None) -> Dict[str, str]:
    """Parse command line args as a dict."""
    parser = argparse.ArgumentParser()

    parser.add_argument("input_file",
                        help="Path to a .geo file generated by Neper.")

    parser.add_argument("output_files",
                        nargs="+",
                        help=("Path of the output file(s). Type is determined "
                              "based on type (.geo, .msh, .inp).")
                        )

    parser.add_argument("-v", "--verbose",
                        action="store_true",
                        default=False,
                        help="Increase output verbosity.")
    
    parser.add_argument("--show_gui",
                        action="store_true",
                        default=False,
                        help="Open the gmsh GUI after the model was imported.")
    
    parser.add_argument("--show_gmsh",
                        action="store_true",
                        default=False,
                        help="Show gmsh output on stdout.")

    parser.add_argument("--ciGen",
                        action="store_true",
                        default=False,
                        help="Generate ciGen output for cohesive elements.")
    if argstr:
        args = parser.parse_args(argstr)
    else:
        args = parser.parse_args()

    return vars(args)


def setup_logger(verbose: bool) -> None:
    """Configure the logging to stdout/stderr."""
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


def import_neper_model_to_occ(input_file: str) -> None:
    """Load a gmsh model from a geo-file generated by Neper into occ kernel."""
    gmsh.model.add("geo")
    gmsh.model.setCurrent("geo")

    gmsh.open(input_file)
    LOGGER.info("Neper file '{input_file}' imported in geo kernel.")

    mg = ModelGeometry.from_current_model()
    LOGGER.info("Extracted ModelGeometry from geo kernel.")

    gmsh.model.add("occ")
    gmsh.model.setCurrent("occ")

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

    # cut everything outside of a rectangular prism
    trim_to_unit_domain(mg.dim, rx, ry, rz)


def trim_to_unit_domain(dim: int, rx: float, ry: float, rz: float) -> None:
    if dim == 3:
        cut_geo = gmsh.model.occ.addBox(0, 0, 0, rx, ry, rz)
    else:
        cut_geo = gmsh.model.occ.addRectangle(0, 0, 0, rx, ry)

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


    # delete all the old physical groups (because we cant update them)
    gmsh.model.removePhysicalGroups(gmsh.model.getPhysicalGroups())
    
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


def main(input_file: str, 
         output_files: str,
         verbose: bool = False,
         show_gui: bool = False, 
         show_gmsh: bool = False,
         ciGen: bool = False,
         ) -> None:
    setup_logger(verbose)

    LOGGER.info("This is Mosaic.py :-)")

    LOGGER.info(f"input file: {input_file}")
    LOGGER.info(f"output files: {output_files}")

    if gmsh.isInitialized():
        gmsh.finalize()

    gmsh.initialize()
    
    gmsh.option.setNumber("General.Terminal", int(show_gmsh))

    import_neper_model_to_occ(input_file)
    
    dim = gmsh.model.getDimension()
    
    if ciGen:
        gmsh.option.setNumber("Mesh.Format", 2) 
        gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
    
    gmsh.model.mesh.generate(dim)

    for file in output_files:
        gmsh.write(file)
        
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
        
    #gmsh.finalize()


if __name__ == "__main__":

    # debug_args = [
    #               # "examples/n3-id1.geo",
    #               "examples/n5-id1.geo",
    #               "testout.geo_unrolled",
    #               "testout.msh",
    #               "-v",
    #               "--show_gui",
    #               "--show_gmsh",
    #               ]

    # args = parse_args(debug_args)
    args = parse_args()

    main(**args)



