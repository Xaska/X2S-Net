import os
import random
import shutil
def dataset_seg(input_path, shuffle=True):
    for file in os.listdir(input_path):
        os.makedirs("../dataset/" + file + "/train/x1/", exist_ok=True)
        os.makedirs("../dataset/" + file + "/train/x2/", exist_ok=True)
        os.makedirs("../dataset/" + file + "/train/seg1/", exist_ok=True)
        os.makedirs("../dataset/" + file + "/train/seg2/", exist_ok=True)
        os.makedirs("../dataset/" + file + "/train/mat/", exist_ok=True)
        os.makedirs("../dataset/" + file + "/validation/x1/", exist_ok=True)
        os.makedirs("../dataset/" + file + "/validation/x2/", exist_ok=True)
        os.makedirs("../dataset/" + file + "/validation/seg1/", exist_ok=True)
        os.makedirs("../dataset/" + file + "/validation/seg2/", exist_ok=True)
        os.makedirs("../dataset/" + file + "/validation/mat/", exist_ok=True)
        os.makedirs("../dataset/" + file + "/test/x1/", exist_ok=True)
        os.makedirs("../dataset/" + file + "/test/x2/", exist_ok=True)
        os.makedirs("../dataset/" + file + "/test/seg1/", exist_ok=True)
        os.makedirs("../dataset/" + file + "/test/seg2/", exist_ok=True)
        os.makedirs("../dataset/" + file + "/test/mat/", exist_ok=True)
        file_list = list(range(1, 1001, 1))
        if shuffle:
            random.shuffle(file_list)
        train_list = file_list[0:800]
        val_list = file_list[800:900]
        test_list = file_list[900:1000]
        if file == "9L1":
            sde = "T9_L1_"
        elif file == "59":
            sde = "T5_9_"
        elif file == "L1L5":
            sde = "L15_"
        elif file == "S9L1":
            sde = "T9_L1_"
        elif file == "S15":
            sde = "T1_5_"
        elif file == "S59":
            sde = "T5_9_"
        elif file == "SL1L5":
            sde = "L_"
        for img in train_list:
            shutil.copy(os.path.join(input_path, file, "generator_0", str(img) + ".jpg"),
                        "../dataset/" + file + "/train/x1/" + str(img) + ".jpg")
            shutil.copy(os.path.join(input_path, file, "generator_90", str(img) + ".jpg"),
                        "../dataset/" + file + "/train/x2/" + str(img) + ".jpg")
            shutil.copy(os.path.join(input_path, file, "thresh_0", str(img) + ".jpg"),
                        "../dataset/" + file + "/train/seg1/" + str(img) + ".jpg")
            shutil.copy(os.path.join(input_path, file, "thresh_90", str(img) + ".jpg"),
                        "../dataset/" + file + "/train/seg2/" + str(img) + ".jpg")
            shutil.copy(os.path.join(input_path, file, "off_mat/spine/train", sde + str(img) + ".mat"),
                        "../dataset/" + file + "/train/mat/" + sde + str(img) + ".mat")
        for img in val_list:
            shutil.copy(os.path.join(input_path, file, "generator_0", str(img) + ".jpg"),
                        "../dataset/" + file + "/validation/x1/" + str(img) + ".jpg")
            shutil.copy(os.path.join(input_path, file, "generator_90", str(img) + ".jpg"),
                        "../dataset/" + file + "/validation/x2/" + str(img) + ".jpg")
            shutil.copy(os.path.join(input_path, file, "thresh_0", str(img) + ".jpg"),
                        "../dataset/" + file + "/validation/seg1/" + str(img) + ".jpg")
            shutil.copy(os.path.join(input_path, file, "thresh_90", str(img) + ".jpg"),
                        "../dataset/" + file + "/validation/seg2/" + str(img) + ".jpg")
            shutil.copy(os.path.join(input_path, file, "off_mat/spine/train", sde + str(img) + ".mat"),
                        "../dataset/" + file + "/validation/mat/" + sde + str(img) + ".mat")
        for img in test_list:
            shutil.copy(os.path.join(input_path, file, "generator_0", str(img) + ".jpg"),
                        "../dataset/" + file + "/test/x1/" + str(img) + ".jpg")
            shutil.copy(os.path.join(input_path, file, "generator_90", str(img) + ".jpg"),
                        "../dataset/" + file + "/test/x2/" + str(img) + ".jpg")
            shutil.copy(os.path.join(input_path, file, "thresh_0", str(img) + ".jpg"),
                        "../dataset/" + file + "/test/seg1/" + str(img) + ".jpg")
            shutil.copy(os.path.join(input_path, file, "thresh_90", str(img) + ".jpg"),
                        "../dataset/" + file + "/test/seg2/" + str(img) + ".jpg")
            shutil.copy(os.path.join(input_path, file, "off_mat/spine/train", sde + str(img) + ".mat"),
                        "../dataset/" + file + "/test/mat/" + sde + str(img) + ".mat")

if __name__ == '__main__':
    dataset_seg("../dataset_orginal")
