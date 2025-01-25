import cv2
import numpy as np
import trimesh
from PIL import Image, ImageDraw

def load_obj_and_mtl(obj_file, mtl_file):
    """
    解析 .obj 和 .mtl 文件，返回顶点、UV、面和材质信息。
    """
    vertices = []
    uvs = []
    faces = []
    materials = {}
    current_material = None

    # 解析 MTL 文件
    with open(mtl_file, 'r') as mtl:
        lines = mtl.readlines()
        for line in lines:
            line = line.strip()
            if line.startswith("newmtl"):
                current_material = line.split()[1]
                materials[current_material] = {"Kd": (1.0, 1.0, 1.0)}  # 默认白色
            elif line.startswith("Kd") and current_material:
                materials[current_material]["Kd"] = tuple(map(float, line.split()[1:]))

    # 解析 OBJ 文件
    with open(obj_file, 'r') as obj:
        lines = obj.readlines()
        for line in lines:
            line = line.strip()
            if line.startswith("v "):  # 顶点
                vertices.append(list(map(float, line.split()[1:])))
            elif line.startswith("vt "):  # UV 坐标
                uvs.append(list(map(float, line.split()[1:])))
            elif line.startswith("usemtl"):  # 当前材质
                current_material = line.split()[1]
            elif line.startswith("f "):  # 面
                face = line.split()[1:]
                face_data = []
                for vertex in face:
                    v_data = vertex.split("/")
                    face_data.append((int(v_data[0]) - 1,  # 顶点索引
                                      int(v_data[1]) - 1 if len(v_data) > 1 and v_data[1] else None,  # UV 索引
                                      current_material))  # 材质
                faces.append(face_data)

    return vertices, uvs, faces, materials


def generate_texture(uvs, faces, materials, texture_size=(512, 512)):
    """
    根据 UV 坐标、面和材质生成纹理图像。
    """
    texture = np.ones((texture_size[1], texture_size[0], 3), dtype=np.uint8) * 255  # 默认白色纹理

    for face in faces:
        color = materials[face[0][2]]["Kd"]  # 获取材质颜色
        color = tuple(int(c * 255) for c in color)  # 转换为 RGB
        uv_coords = [uvs[vertex[1]] for vertex in face if vertex[1] is not None]  # 获取 UV 坐标
        if len(uv_coords) == 3:  # 三角形面
            # 将 UV 坐标映射到纹理空间
            uv_coords = [(int(u * texture_size[0]), int((1 - v) * texture_size[1])) for u, v in uv_coords]
            # 填充三角形（简化实现，可以优化为抗锯齿）
            ImageDraw.Draw(texture).polygon(uv_coords, fill=color)

    return Image.fromarray(texture)

# def generate_texture(texture_type="noise", size=(4096, 4096), save_path = None):
#     height, width = size

#     if texture_type == "noise":
#         texture = np.random.randint(250, 256, (height, width, 3), dtype=np.uint8)
    
#     elif texture_type == "stripes":
#         texture = np.zeros((height, width, 3), dtype=np.uint8)
#         stripe_width = 20
#         for i in range(0, width, stripe_width * 2):
#             texture[:, i:i + stripe_width] = (255, 255, 255)  # 白色条纹

#     elif texture_type == "gradient":
#         # 渐变纹理
#         x = np.linspace(0, 255, width, dtype=np.uint8)
#         gradient = np.tile(x, (height, 1))
#         texture = cv2.merge([gradient, gradient, gradient])  # 灰度渐变

#     else:
#         raise ValueError("Unsupported texture_type. Use 'noise', 'stripes', or 'gradient'.")
    
#     cv2.imwrite(save_path, texture)

def convert_red_to_blue(image_path, save_path):
    image = cv2.imread(image_path)
    
    if image is None:
        raise ValueError(f"无法加载图片: {image_path}")
    
    if image.shape[2] != 3:
        raise ValueError("图片不是三通道，无法处理！")
    
    blue_image = image.copy()
    blue_image[:, :, 0] = image[:, :, 2]
    blue_image[:, :, 2] = 0
    
    cv2.imwrite(save_path, blue_image)

if __name__ == "__main__":
    # texture_path = "/home/haowen/hw_mine/Real_Sim_Real/simple_sim/asset/know/meshes/can/texture_map.png"
    # convert_red_to_blue(texture_path, "blue_texture.png")
    # generate_texture("noise", save_path="stripes_texture.png")
    obj_file = "/home/haowen/hw_mine/Real_Sim_Real/simple_sim/asset/know/meshes/stop_button/textured.obj"
    mtl_file = "/home/haowen/hw_mine/Real_Sim_Real/simple_sim/asset/know/meshes/stop_button/textured.mtl"
    output_texture_path = "output_texture.png"

    vertices, uvs, faces, materials = load_obj_and_mtl(obj_file, mtl_file)
    texture = generate_texture(uvs, faces, materials)
    texture.save(output_texture_path)