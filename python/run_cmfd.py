from oversegmentation import *
from matching_postprocessing import generated_object_match_map
from glob import glob
from tqdm.contrib.concurrent import process_map
from pathlib import Path
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import cv2
import os

#files = glob("/work/jcardenuto/translate-forgery-algs/testset/**/figure_v?.png", recursive=True)
#files += glob("/work/jcardenuto/translate-forgery-algs/testset/**/figure.png", recursive=True)
files = glob("*.png")

def run_method(fname):
    exist_name = fname.split('testset/')[-1]
    exist_name = exist_name[:-4]
    if os.path.isfile(f"result/{exist_name}_map.png")  \
            and os.path.isfile(f"result/{exist_name}_final_map.png") \
            and os.path.isfile(f'result/{exist_name}_final_map.png') \
            and os.path.isfile(f'result/{exist_name}_matched_image.png'):
        return

    #img = cv2.imread(fname)
    #img_rgb = img[...,::-1]
    img_rgb = Image.open(fname)
    img_rgb = np.array(img_rgb)

    # run method 
    forgery_map , matches, Locations = oversegmentation(fname)

    # Fix locations
    p1 = []
    p2 = []
    for match in matches:
        point1, point2 = match
        p1.append( tuple( int(p) for p in Locations[point1][1::-1]))
        p2.append( tuple( int(p) for p in Locations[point2][1::-1]))
    p1 = np.array(p1).T
    p2 = np.array(p2).T

    final_map = generated_object_match_map(forgery_map,p1, p2)

    fname = fname.split('testset/')[-1]
    fname = fname[:-4]
    os.makedirs(f'result/{fname}',exist_ok=True)

    forgery_map = forgery_map.astype("uint8")
    final_map = final_map.astype("uint8")
    cv2.imwrite(f"result/{fname}_map.png",forgery_map)
    cv2.imwrite(f"result/{fname}_final_map.png", final_map)
    plt.imsave(f'result/{fname}_final_map.png', final_map)

process_map(run_method, files, chunksize = 10, max_workers = 30)

