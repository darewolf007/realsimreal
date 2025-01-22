import os
from robosuite.models.objects import MujocoXMLObject
from robosuite.utils.mjcf_utils import xml_path_completion

class CanObject(MujocoXMLObject):
    def __init__(self, name):
        base_path = os.path.dirname(os.path.realpath(__file__))
        obj_xml = os.path.join(base_path, "../asset/know/can.xml")
        super().__init__(
            xml_path_completion(obj_xml),
            name=name,
            joints=[dict(type="free", damping="0.01")],
            obj_type="all",
            duplicate_collision_geoms=True,
        )

class KettleObject(MujocoXMLObject):
    def __init__(self, name):
        base_path = os.path.dirname(os.path.realpath(__file__))
        obj_xml = os.path.join(base_path, "../asset/know/kettle.xml")
        super().__init__(
            xml_path_completion(obj_xml),
            name=name,
            joints=[dict(type="free", damping="0.01")],
            obj_type="all",
            duplicate_collision_geoms=True,
        )

# class CupObject(MujocoXMLObject):
#     def __init__(self, name):
#         base_path = os.path.dirname(os.path.realpath(__file__))
#         obj_xml = os.path.join(base_path, "../asset/know/cup.xml")
#         super().__init__(
#             xml_path_completion(obj_xml),
#             name=name,
#             joints=[dict(type="free", damping="0.01")],
#             obj_type="all",
#             duplicate_collision_geoms=True,
#         )

class CupObject(MujocoXMLObject):
    def __init__(self, name):
        base_path = os.path.dirname(os.path.realpath(__file__))
        obj_xml = os.path.join(base_path, "../asset/know/cup.xml")
        super().__init__(
            xml_path_completion(obj_xml),
            name=name,
            joints=None,
            obj_type="visual",
            duplicate_collision_geoms=True,
        )

class BananaObject(MujocoXMLObject):
    def __init__(self, name):
        base_path = os.path.dirname(os.path.realpath(__file__))
        obj_xml = os.path.join(base_path, "../asset/know/banana.xml")
        super().__init__(
            xml_path_completion(obj_xml),
            name=name,
            joints=[dict(type="free", damping="0.01")],
            obj_type="all",
            duplicate_collision_geoms=True,
        )

class AppleObject(MujocoXMLObject):
    def __init__(self, name):
        base_path = os.path.dirname(os.path.realpath(__file__))
        obj_xml = os.path.join(base_path, "../asset/know/apple.xml")
        super().__init__(
            xml_path_completion(obj_xml),
            name=name,
            joints=[dict(type="free", damping="0.01")],
            obj_type="all",
            duplicate_collision_geoms=True,
        )

class BowlObject(MujocoXMLObject):
    def __init__(self, name):
        base_path = os.path.dirname(os.path.realpath(__file__))
        obj_xml = os.path.join(base_path, "../asset/know/bowl.xml")
        super().__init__(
            xml_path_completion(obj_xml),
            name=name,
            joints=[dict(type="free", damping="0.01")],
            obj_type="all",
            duplicate_collision_geoms=True,
        )

class MarkerObject(MujocoXMLObject):
    def __init__(self, name):
        base_path = os.path.dirname(os.path.realpath(__file__))
        obj_xml = os.path.join(base_path, "../asset/know/marker.xml")
        super().__init__(
            xml_path_completion(obj_xml),
            name=name,
            joints=[dict(type="free", damping="0.01")],
            obj_type="all",
            duplicate_collision_geoms=True,
        )