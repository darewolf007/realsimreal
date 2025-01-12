import cv2
import numpy as np

def overlay_images(image1, image2, alpha=0.3):
    img1 = cv2.imread(image1)
    img2 = cv2.imread(image2)
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    if len(img1.shape) != len(img2.shape):
        if len(img1.shape) == 2:
            img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
        if len(img2.shape) == 2:
            img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
    beta = 1.0 - alpha
    overlaid = cv2.addWeighted(img1, alpha, img2, beta, 0)
    cv2.imshow('Overlaid Image', overlaid)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    overlay_images("/home/haowen/hw_mine/Real_Sim_Real/data/can_pour_1/3/scene_rgb_image/scene_150.jpg",
                   "/home/haowen/hw_mine/Real_Sim_Real/scene_view_image.png")
