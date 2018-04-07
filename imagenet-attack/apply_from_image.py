from attack_util import write_img, read_img, load_all_pngjpg_in_dir, imresize
import numpy as np
import sys
import os

if __name__ == "__main__":
    if len(sys.argv) < 4: 
        print("Usage: python apply_from_image.py <from_image> <to_folder> <mask>")
        sys.exit(0)
    
    from_img_name = sys.argv[1]
    from_img = read_img(from_img_name)
    shape = from_img.shape
    from_img_name = from_img_name.split(".")[0]

    mask = read_img(sys.argv[3])/255.0
    inv_mask = 1.0 - mask

    to_fldr = sys.argv[2]
    save_folder = os.path.join(to_fldr, from_img_name)
    if not os.path.isdir(save_folder):
        os.makedirs(save_folder)

    fnames, to_imgs = load_all_pngjpg_in_dir(to_fldr)

    
    for f, img in zip(fnames, to_imgs):
        new_img = imresize(img, shape[0:2]) * inv_mask + from_img * mask
        write_img(os.path.join(save_folder, f), new_img)
    
