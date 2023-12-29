import cv2
import numpy as np
import os

def binary2edge(mask_path):
    """
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    ret, mask_binary = cv2.threshold(mask, 10, 255, cv2.THRESH_BINARY)  # if <0, pixel=0 else >0, pixel=255
    mask_edge = cv2.Canny(mask_binary, 10, 150)

    return mask_edge
def test_1edge():
    file = "../dataset"
    for f in os.listdir(file):
        os.makedirs(os.path.join(file, f, "train", "seg1_Edge"), exist_ok=True)
        for name in os.listdir(os.path.join(file, f, "train", 'seg1')):
            edge = binary2edge(os.path.join(file, f, "train", 'seg1', name))
            cv2.imwrite(os.path.join(file, f, "train", "seg1_Edge", name), edge)
def test_2edge():
    file = "../dataset"
    for f in os.listdir(file):
        os.makedirs(os.path.join(file, f, "train", "seg2_Edge"), exist_ok=True)
        for name in os.listdir(os.path.join(file, f, "train", 'seg2')):
            edge = binary2edge(os.path.join(file, f, "train", 'seg2', name))
            cv2.imwrite(os.path.join(file, f, "train", "seg2_Edge", name), edge)
