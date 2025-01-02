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