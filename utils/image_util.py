import os
import cv2
import pickle
import numpy as np

def resize_image(image, scale_factor=0.5):                
    height, width = image.shape[:2]
    new_size = (int(width * scale_factor), int(height * scale_factor))
    resized_image = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
    return resized_image

def save_image_pkl(image_dict, path, save_ori_image = False):
    if not os.path.exists(path):
         os.makedirs(path)
    pkl_num = sum(1 for name in os.listdir(path) if name.endswith('.pkl'))
    pkl_path = os.path.join(path, f"{pkl_num + 1}.pkl")
    with open(pkl_path, 'wb') as file:
        pickle.dump(image_dict, file)
    if save_ori_image:
        for key, image in image_dict.items():
            if 'view' in key:
                save_path = os.path.join(path, f"{key}_{pkl_num + 1}.png")
                cv2.imwrite(save_path, image)

def show_image(image):
    cv2.imshow("image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # image_dict = {
    #     "sceneview": np.random.rand(224, 224, 3).astype(np.uint8),
    #     "birdview": np.random.rand(224, 224, 3).astype(np.uint8),
    #     "frontview": np.random.rand(224, 224, 3).astype(np.uint8),
    #     "rightview": np.random.rand(224, 224, 3).astype(np.uint8),
    #     "result": 1
    # }
    # save_dir = "/home/haowen/hw_mine/Real_Sim_Real/experiments/simulation_online_result/Pour_can_into_cup/test"
    # save_image_pkl(image_dict, save_dir, True)
    with open("/home/haowen/hw_mine/Real_Sim_Real/experiments/Pour_can_into_cup/promot_data/done/1.pkl", 'rb') as file:
        data = pickle.load(file)
        image = data['obs_view']
        import matplotlib.pyplot as plt
        plt.imshow(image)
        plt.show()
        # show_image(image)
