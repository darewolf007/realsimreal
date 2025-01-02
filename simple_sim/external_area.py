from robosuite.models.base import MujocoXML
from robosuite.utils.mjcf_utils import (
    ENVIRONMENT_COLLISION_COLOR,
    array_to_string,
    find_elements,
    new_body,
    new_element,
    new_geom,
    new_joint,
    recolor_collision_geoms,
    string_to_array,
)


class ExternalArea(MujocoXML):
    def __init__(self, fname):
        super().__init__(fname)

    def set_camera(self, camera_name, pos, quat, camera_attribs=None):
        camera = find_elements(root=self.worldbody, tags="camera", attribs={"name": camera_name}, return_first=True)
        if camera_attribs is None:
            camera_attribs = {}
        camera_attribs["pos"] = array_to_string(pos)
        camera_attribs["quat"] = array_to_string(quat)

        if camera is None:
            self.worldbody.append(new_element(tag="camera", name=camera_name, **camera_attribs))
        else:
            for attrib, value in camera_attribs.items():
                camera.set(attrib, value)