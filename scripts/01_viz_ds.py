import torch
from src.datasets.dataloader import GridDataset
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
ds = GridDataset(
    csv_name="train_indices.csv",
    root_dir="/Users/florian/data/nrcan/data_samples_approach_1",
    feature_names_list=['ignition_grid', 'fuel_grid', 'elevation_grid'],
    filename_col="filename",
    out_norm="min_max",
    fuel_feats_encoding="ordinal",
    normalize_fuel_feats_ordinal=True,
    modelling_approach="1",
    valid_mask_threshold=0.0,
    transform=None,
)

def tnp(x):
    return (torch.permute(x, (1,2,0)).float().numpy() * 255).astype(np.uint8)

for x in ds:
    inputs, targets, mask = x
    # print ("inputs[:,:,1]", inputs[:,:,1].min(), inputs[:,:,1].max())
    # continue
    inputs = tnp(inputs)
    mask = tnp(mask)
    targets = tnp(targets)
    print ("inputs", inputs.shape, inputs.dtype, inputs.min(), inputs.max())
    print ("mask", mask.shape, mask.dtype, mask.min(), mask.max())
    print ("targets", targets.shape, targets.dtype, targets.min(), targets.max())
    plt.subplot(2, 5, 1)
    plt.imshow(inputs[:,:,0])
    plt.title("Ignition Grid")
    plt.subplot(2, 5, 2)
    plt.imshow(inputs[:,:,1])
    plt.title("Fuel Grid")
    plt.subplot(2, 5, 6)
    plt.imshow(inputs[:,:,1]>0)
    plt.title("Fuel Grid MASK >0")
    plt.subplot(2, 5, 7)
    plt.imshow(inputs[:,:,1]>18)
    plt.title("Fuel Grid MASK > '1'")
    
    plt.subplot(2, 5, 8)
    img = (inputs[:,:,1]>18).astype(np.uint8)
    erosion_size = 1
    erosion_element = cv.getStructuringElement(cv.MORPH_ELLIPSE, (2 * erosion_size + 1, 2 * erosion_size + 1),
                                       (erosion_size, erosion_size))
    erosion_dst = cv.erode(img, erosion_element)
    
    plt.imshow(erosion_dst)
    plt.title("Fuel Grid MASK > '1'\n + erode-1")

    
    
    plt.subplot(2, 5, 9)
    dilation_size = 6
    dilation_element = cv.getStructuringElement(cv.MORPH_ELLIPSE, (2 * dilation_size + 1, 2 * dilation_size + 1),
                                       (dilation_size, dilation_size))
    dilation_dst = cv.dilate(erosion_dst, dilation_element)
    
    logical_and_dst = cv.bitwise_and((inputs[:,:,1]>0).astype(np.uint8), dilation_dst)
    plt.imshow(logical_and_dst)
    plt.title("Fuel Grid MASK > '1'\n + erode-1 + dilate-6")
    plt.subplot(2, 5, 3)
    plt.imshow(inputs[:,:,2])
    plt.title("Elevation Grid")
    plt.subplot(2, 5, 4)
    plt.imshow(mask)
    plt.title("Mask")
    plt.subplot(2, 5, 5)
    plt.imshow(targets)
    plt.title("Targets")
    plt.subplot(2, 5, 10)
    plt.imshow(targets>0)
    plt.title("Targets MASK")
    plt.show()
    

