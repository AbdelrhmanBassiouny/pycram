import numpy as np
import pathlib
from typing_extensions import Optional, List

from pycram.datastructures.dataclasses import VisualShapeUnion
from pycram.datastructures.enums import Shape
from pycram.ros import logwarn

try:
    from multiverse_parser import Configuration, Factory, InertiaSource, GeomBuilder
    from multiverse_parser import (WorldBuilder,
                                   GeomType, GeomProperty,
                                   MeshProperty,
                                   MaterialProperty)
    from multiverse_parser import MjcfExporter
    from pxr import Usd, UsdGeom
except ImportError:
    # do not import this module if multiverse is not found
    multiverse_parser = None
    pxr = None
    logwarn("Multiverse Parser not found. Some features may not be available.")
    raise ImportError("Multiverse not found.")


class ObjectFactory(Factory):
    """
    Create MJCF object descriptions from mesh files.
    """

    def __init__(self, file_path: str, config: Configuration):
        super().__init__(file_path, config)

    def from_mesh_file(self, object_name: str, texture_type: str = "png"):

        self._world_builder = WorldBuilder(usd_file_path=self.tmp_usd_file_path)

        body_builder = self._world_builder.add_body(body_name=object_name)

        tmp_usd_mesh_file_path, tmp_origin_mesh_file_path = self.import_mesh(
            mesh_file_path=self.source_file_path, merge_mesh=True)
        mesh_stage = Usd.Stage.Open(tmp_usd_mesh_file_path)
        for idx, mesh_prim in enumerate([prim for prim in mesh_stage.Traverse() if prim.IsA(UsdGeom.Mesh)]):
            mesh_name = mesh_prim.GetName()
            mesh_path = mesh_prim.GetPath()
            mesh_property = MeshProperty.from_mesh_file_path(mesh_file_path=tmp_usd_mesh_file_path,
                                                             mesh_path=mesh_path)
            # mesh_property._texture_coordinates = None # TODO: See if needed otherwise remove it.
            geom_property = GeomProperty(geom_type=GeomType.MESH,
                                         is_visible=False,
                                         is_collidable=True)
            geom_builder = body_builder.add_geom(geom_name=f"SM_{object_name}_mesh_{idx}",
                                                 geom_property=geom_property)
            geom_builder.add_mesh(mesh_name=mesh_name, mesh_property=mesh_property)

            # Add texture if available
            texture_file_path = self.source_file_path.replace(pathlib.Path(self.source_file_path).suffix,
                                                              f".{texture_type}")
            if pathlib.Path(texture_file_path).exists():
                self.add_material_with_texture(geom_builder=geom_builder, material_name=f"M_{object_name}_{idx}",
                                               texture_file_path=texture_file_path)

            geom_builder.build()

        body_builder.compute_and_set_inertial(inertia_source=InertiaSource.FROM_COLLISION_MESH)

    @staticmethod
    def add_material_with_texture(geom_builder: GeomBuilder, material_name: str, texture_file_path: str):
        """
        Add a material with a texture to the geom builder.

        :param geom_builder: The geom builder to add the material to.
        :param material_name: The name of the material.
        :param texture_file_path: The path to the texture file.
        """
        material_property = MaterialProperty(diffuse_color=texture_file_path,
                                             opacity=None,
                                             emissive_color=None,
                                             specular_color=None)
        geom_builder.add_material(material_name=material_name,
                                  material_property=material_property)

    def export_to_mjcf(self, output_file_path: str):
        """
        Export the object to a MJCF file.

        :param output_file_path: The path to the output file.
        """
        exporter = MjcfExporter(self, output_file_path)
        exporter.build()
        exporter.export(keep_usd=False)


class PrimitiveObjectFactory(ObjectFactory):

    def __init__(self, object_name: str, shape_data: VisualShapeUnion, save_path: str,
                 orientation: Optional[List[float]] = None):
        """
        Create an MJCF object description from a primitive shape.

        :param object_name: The name of the object.
        :param shape_data: The shape data of the object.
        :param save_path: The path to save the MJCF file.
        :param orientation: The orientation of the object.
        """
        self.shape_data: VisualShapeUnion = shape_data
        self.orientation: List[float] = [0, 0, 0, 1] if orientation is None else orientation
        config = Configuration(model_name=object_name,
                               fixed_base=False,
                               with_visual=True,
                               with_collision=True,
                               default_rgba=np.array(shape_data.rgba_color.get_rgba()))
        super().__init__(save_path, config)

    def build_shape(self):

        self._world_builder = WorldBuilder(usd_file_path=self.tmp_usd_file_path)

        body_builder = self._world_builder.add_body(body_name=self.config.model_name)

        geom_type_map = {
            Shape.SPHERE: GeomType.SPHERE,
            Shape.BOX: GeomType.CUBE,
            Shape.CYLINDER: GeomType.CYLINDER,
            Shape.PLANE: GeomType.PLANE,
            Shape.CAPSULE: GeomType.CAPSULE,
        }
        geom_property = GeomProperty(geom_type=geom_type_map[self.shape_data.visual_geometry_type],
                                     is_visible=self.config.with_visual,
                                     is_collidable=self.config.with_collision,
                                     rgba=self.config.default_rgba)

        geom_builder = body_builder.add_geom(
            geom_name=f"{self.config.model_name}_Shape",
            geom_property=geom_property
        )
        geom_pos = self.shape_data.visual_frame_position.to_list()
        geom_quat = np.array(self.orientation)
        if self.shape_data.visual_geometry_type == Shape.PLANE:
            geom_builder.set_transform(pos=geom_pos, quat=geom_quat, scale=np.array([50, 50, 1]))
        elif self.shape_data.visual_geometry_type == Shape.BOX:
            geom_builder.set_transform(pos=geom_pos, quat=geom_quat,
                                       scale=np.array(self.shape_data.half_extents) * 2)
        elif self.shape_data.visual_geometry_type == Shape.SPHERE:
            geom_builder.set_transform(pos=geom_pos, quat=geom_quat)
            geom_builder.set_attribute(radius=self.shape_data.radius)
        elif self.shape_data.visual_geometry_type in [Shape.CYLINDER, Shape.CAPSULE]:
            geom_builder.set_transform(pos=geom_pos, quat=geom_quat)
            geom_builder.set_attribute(radius=self.shape_data.radius, height=self.shape_data.length)
        geom_builder.build()