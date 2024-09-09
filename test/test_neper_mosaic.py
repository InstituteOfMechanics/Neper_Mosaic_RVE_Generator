"""
Tests for neper_mosaic.

"""
import pytest
import numpy as np
import gmsh
from neper_mosaic import main, load_rectilinear_geometry


@pytest.fixture
def gmsh_instance():
    """Initialize and finalize gmsh for a test."""
    gmsh.initialize()
    yield
    # tests running here...
    gmsh.finalize()

def test_main_2d(tmp_path):
    """The Mosaic main() routine works for the 2d case."""
    input_file = "2d-n5-id1.geo"
    output_files = [tmp_path / "out.geo_unrolled",
                    tmp_path / "out.msh",
                    ]

    main(input_file, output_files)

    for file in output_files:
        assert file.is_file()

def test_main_3d(tmp_path):
    """The Mosaic main() routine works for the 3d case."""
    input_file = "3d-n4-id1.geo"
    output_files = [tmp_path / "out.geo_unrolled",
                    tmp_path / "out.msh",
                    ]

    main(input_file, output_files)

    for file in output_files:
        assert file.is_file()

def test_domain_position_2d(tmp_path):
    """Mosaic still works in 2d with a change in the domain."""
    input_file = "2d-n5-id1.geo"
    output_files = [tmp_path / "out.msh"]

    main(input_file,
         output_files,
         box_x_position=0.1,
         box_y_position=0.1)

    for file in output_files:
        assert file.is_file()

def test_exception_domain_position_wrong_2d(tmp_path):
    """
    Domain position change raises exception when volume is incomplete in 2d.

    """
    input_file = "2d-n5-id1.geo"
    output_files = [tmp_path / "out.msh"]

    with pytest.raises(RuntimeError) as e:
        main(input_file,
             output_files,
             box_x_position=0.8,
             box_y_position=0.8)

    assert e.match("Model volume after trimming is wrong.")

def test_domain_position_3d(tmp_path):
    """Mosaic still works in 3d with a change in the domain."""
    input_file = "3d-n4-id1.geo"
    output_files = [tmp_path / "out.msh"]

    main(input_file,
         output_files,
         box_x_position=0.1,
         box_y_position=0.1)

    for file in output_files:
        assert file.is_file()

def test_exception_domain_position_wrong_3d(tmp_path):
    """
    Domain position change raises exception when volume is incomplete in 3d.

    """
    input_file = "3d-n4-id1.geo"
    output_files = [tmp_path / "out.msh"]

    with pytest.raises(RuntimeError) as e:
        main(input_file,
             output_files,
             box_x_position=0.8,
             box_y_position=0.8)

    assert e.match("Model volume after trimming is wrong.")

def test_ciGen(tmp_path):
    """The ciGen option does not break Mosaic (ciGen is not tested)."""
    input_file = "2d-n5-id1.geo"
    output_files = [tmp_path / "out.msh"]

    main(input_file, output_files, ciGen=True)

    for file in output_files:
        assert file.is_file()

def test_loading_2d(gmsh_instance, tmp_path):
    """Load_rectilinear_geometry() is working for 2d."""
    input_file = "2d-n5-id1.geo"

    assert len(gmsh.model.getEntities()) == 0

    load_rectilinear_geometry(input_file)
    assert len(gmsh.model.getEntities()) > 0

def test_loading_3d(gmsh_instance):
    """Load_rectilinear_geometry() is working in 3d."""
    input_file = "3d-n4-id1.geo"

    assert len(gmsh.model.getEntities()) == 0

    load_rectilinear_geometry(input_file)
    assert len(gmsh.model.getEntities()) > 0

def test_rve_position_2d(gmsh_instance):
    """Geometry should have the same bounding box independent of trim position (2d)."""
    input_file = "2d-n5-id1.geo"

    load_rectilinear_geometry(input_file)
    bounds_before = gmsh.model.getBoundingBox(-1, -1)

    gmsh.clear()

    load_rectilinear_geometry(input_file,
                              box_x_position=0.1,
                              box_y_position=0.1)

    bounds_after = gmsh.model.getBoundingBox(-1, -1)

    assert np.allclose(bounds_before, bounds_after)

def test_rve_position_3d(gmsh_instance):
    """Geometry should have the same bounding box independent of trim position (3d)."""
    input_file = "3d-n4-id1.geo"

    load_rectilinear_geometry(input_file)
    bounds_before = gmsh.model.getBoundingBox(-1, -1)

    gmsh.clear()

    load_rectilinear_geometry(input_file,
                              box_x_position=0.1,
                              box_y_position=0.1,
                              box_z_position=0.1)

    bounds_after = gmsh.model.getBoundingBox(-1, -1)

    assert np.allclose(bounds_before, bounds_after, atol=1e-6)

def test_exception_rve_z_coordinate_2d(gmsh_instance):
    """Specifying z0 != 0 for 2d does not change the result."""
    input_file = "2d-n5-id1.geo"

    load_rectilinear_geometry(input_file)
    bounds_before = gmsh.model.getBoundingBox(-1, -1)

    gmsh.clear()

    load_rectilinear_geometry(input_file,
                              box_z_position=0.1)

    bounds_after = gmsh.model.getBoundingBox(-1, -1)

    assert np.allclose(bounds_before, bounds_after)

if __name__ == "__main__":
    pytest.main()


