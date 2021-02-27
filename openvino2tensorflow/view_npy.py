import numpy as np
from matplotlib import pyplot as plt
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--npy_file_path', type=str, default='calibration_data_img_sample.npy', help='Specify the path to the .npy file that contains the binaryized image file')
    args = parser.parse_args()
    npy_file_path = args.npy_file_path
    img_array = np.load(npy_file_path)

    for idx in range(img_array.shape[0]):
        print(img_array[idx].shape)
        plt.imshow(img_array[idx])
        plt.show()

if __name__ == '__main__':
    main()
