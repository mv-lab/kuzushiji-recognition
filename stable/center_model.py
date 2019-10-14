import numpy as np
import PIL
from PIL import ImageDraw, ImageFont
from tqdm import tqdm_notebook as tqdm
from fastai.vision import *
from fastai import layers
from math import ceil


#DATA = Path("../data")
#train_fnames = (DATA/"train_fnames.txt").read_text().split("\n")
#valid_fnames = (DATA/"valid_fnames.txt").read_text().split("\n")
#df_train = pd.read_csv("data/train.csv")
#unicode_map = {codepoint: char for codepoint, char in pd.read_csv('data/unicode_translation.csv').values}


def create_heatmap(image_fn, labels, intensity_res=50):
    """Creates heatmap showing center points as ellipses (radii proportional to height/width)"""
    # Some code from anokas' kernel
    
    # Convert annotation string to array
    if type(labels) == float:
        labels = np.array([])
    else:
        labels = np.array(labels.split(' ')).reshape(-1, 5)
    
    # Read image
    imsource = PIL.Image.open(image_fn).convert('RGB')
    bbox_canvas = PIL.Image.new('RGB', imsource.size)
    bbox_draw = ImageDraw.Draw(bbox_canvas) # Separate canvases for boxes and chars so a box doesn't cut off a character

    for codepoint, x, y, w, h in labels:
        x, y, w, h = int(x), int(y), int(w), int(h)
        x_center = x + w // 2
        y_center = y + h // 2

        #intensities = np.array(np.array([255, 255, 255], dtype=np.uint8) * (np.linspace([0,0,0], [1,1,1], intensity_res) ** 2), dtype=np.uint8)
        intensities = np.array(np.array([255, 255, 255], dtype=np.uint8) * np.exp(np.linspace([-5,-5,-5], [0,0,0], intensity_res)), dtype=np.uint8)
        x_radii = np.linspace(w//3, 1, intensity_res, dtype=np.uint8)
        y_radii = np.linspace(h//3, 1, intensity_res, dtype=np.uint8)
        
        for i in range(intensity_res):
            x_r = x_radii[i]
            y_r = y_radii[i]
            bbox_draw.ellipse([x_center - x_r, y_center - y_r, x_center + x_r, y_center + y_r], fill=tuple(intensities[i]))

    return np.asarray(bbox_canvas)


def create_heatmap_circular(image_fn, labels, intensity_res=25, radius=10, resize_to=None, save_resized_im_dir=None):
    """Creates heatmap showing center points as circles (fixed radius)"""
    # Some code from anokas' kernel
    
    intensity_res = min(intensity_res, radius)
    
    # Convert annotation string to array
    if type(labels) == float:
        labels = np.array([])
    else:
        labels = np.array(labels.split(' ')).reshape(-1, 5)
    
    # Read image
    imsource = PIL.Image.open(image_fn).convert('RGB')
    width_orig, height_orig = imsource.size
    
    if resize_to is not None:
        imsource = imsource.resize([resize_to, resize_to])
        
        if save_resized_im_dir is not None:
            imsource.save(save_resized_im_dir/(image_fn.name))
        
    bbox_canvas = PIL.Image.new('RGB', imsource.size)
    bbox_draw = ImageDraw.Draw(bbox_canvas)

    for codepoint, x, y, w, h in labels:
        x, y, w, h = int(x), int(y), int(w), int(h)
        x_center = x + w // 2
        y_center = y + h // 2
        if resize_to is not None:
            x_center = round(x_center * (resize_to / width_orig))
            y_center = round(y_center * (resize_to / height_orig))
      

        #intensities = np.array(np.array([255, 255, 255], dtype=np.uint8) * (np.linspace([0,0,0], [1,1,1], intensity_res) ** 2), dtype=np.uint8)
        intensities = np.array(np.array([255, 255, 255], dtype=np.uint8) * np.square(np.linspace([0,0,0], [1,1,1], intensity_res)), dtype=np.uint8)
        
        radii = np.linspace(radius, 1, intensity_res, dtype=np.uint8)
        
        for i in range(intensity_res):
            r = radii[i]
            bbox_draw.ellipse([x_center - r, y_center - r, x_center + r, y_center + r], fill=tuple(intensities[i]))

    return np.asarray(bbox_canvas)


def generate_heatmap_crops(im_path, hm_save_path, im_crop_save_path, hm_crop_save_path, df, n_crops=25, crop_size=512):
    """Creates heatmaps for all train images and makes random crops"""
    if not hm_save_path.exists(): hm_save_path.mkdir()
    if not im_crop_save_path.exists(): im_crop_save_path.mkdir()
    if not hm_crop_save_path.exists(): hm_crop_save_path.mkdir()

    for row in tqdm(df.values):
        img_id, labels = row
        hm = create_heatmap(im_path/"{}.jpg".format(img_id), labels)
        PIL.Image.fromarray(hm).save(hm_save_path/"{}.png".format(img_id))
        im = np.array(PIL.Image.open(im_path/f"{img_id}.jpg"))
        height, width = im.shape[:2]
        for i in range(crops):
            x = np.random.randint(0, width - crop_size)
            y = np.random.randint(0, height - crop_size)
            cropped_im = im[y:y+crop_size, x:x+crop_size]
            PIL.Image.fromarray(cropped_im).save(im_crop_save_path/f"{img_id}___{str(i).zfill(6)}.png")
            cropped_hm = hm[y:y+crop_size, x:x+crop_size]
            PIL.Image.fromarray(cropped_hm).save(hm_crop_save_path/f"{img_id}___{str(i).zfill(6)}.png")
        

def generate_heatmap_crops_circular(im_path, hm_save_path, im_crop_save_path, hm_crop_save_path, df, overlap=256, crop_size=1024, resize_to=256, radius=25):
    """Creates heatmaps with circular centers for all train images and makes random crops.
NEW DETERMINISTIC CROPPING (GRID). overlap IS NUMBER OF PIXELS CROPPED REGIONS OVERLAP BY"""
    if not hm_save_path.exists(): hm_save_path.mkdir()
    if not im_crop_save_path.exists(): im_crop_save_path.mkdir()
    if not hm_crop_save_path.exists(): hm_crop_save_path.mkdir()

    for row in tqdm(df.values):
        img_id, labels = row
        hm = create_heatmap_circular(im_path/"{}.jpg".format(img_id), labels, radius=radius)
        PIL.Image.fromarray(hm).save(hm_save_path/"{}.png".format(img_id))
        im = np.array(PIL.Image.open(im_path/f"{img_id}.jpg"))
        height, width = im.shape[:2]
        
        for x_i in range(ceil((width - crop_size) / overlap)): # ceil because we want to include crops with the edge of the image
            for y_i in range(ceil((height - crop_size) / overlap)):
                # Note that the crops which include the right or bottom edges might be less than crop_size x crop_size
                x_offset = x_i * overlap
                y_offset = y_i * overlap
                cropped_im = im[y_offset:y_offset+crop_size, x_offset:x_offset+crop_size]
                fname = f"{img_id}___{str(x_i).zfill(3)}_{str(y_i).zfill(3)}.png"
                PIL.Image.fromarray(cropped_im).resize([resize_to, resize_to]).save(im_crop_save_path/fname)
                cropped_hm = hm[y_offset:y_offset+crop_size, x_offset:x_offset+crop_size]
                PIL.Image.fromarray(cropped_hm).resize([resize_to, resize_to]).save(hm_crop_save_path/fname)
               
            
def generate_heatmaps_circular_whole_page(im_path, im_save_path, hm_save_path, df, overlap=256, crop_size=1024, resize_to=512, radius=25, intensity_res=10):
    """Creates heatmaps with circular centers for all train images and makes random crops.
NEW DETERMINISTIC CROPPING (GRID). overlap IS NUMBER OF PIXELS CROPPED REGIONS OVERLAP BY"""
    if not hm_save_path.exists(): hm_save_path.mkdir()
    if not im_save_path.exists(): im_save_path.mkdir()

    for row in tqdm(df.values):
        img_id, labels = row
        hm = create_heatmap_circular(im_path/"{}.jpg".format(img_id), labels, radius=radius, resize_to=resize_to, intensity_res=intensity_res, save_resized_im_dir=im_save_path)
        PIL.Image.fromarray(hm).save(hm_save_path/"{}.png".format(img_id))
        
        
            
def generate_crops(im_path, hm_path, im_save_path, hm_save_path, df, n_crops=25, crop_size=512, resize_to=512):
    """Generates the crops from pre-generated heatmaps"""
    if not im_save_path.exists(): im_save_path.mkdir()
    if not hm_save_path.exists(): hm_save_path.mkdir()

    for row in tqdm(df.values):
        img_id, labels = row
        im = np.array(PIL.Image.open(im_path/f"{img_id}.jpg"))
        hm = np.array(PIL.Image.open(hm_path/f"{img_id}.png"))
        height, width = im.shape[:2]
        for i in range(n_crops):
            x = np.random.randint(0, width - crop_size)
            y = np.random.randint(0, height - crop_size)
            cropped_im = im[y:y+crop_size, x:x+crop_size]
            cropped_im = PIL.Image.fromarray(cropped_im).resize([resize_to, resize_to], resample=PIL.Image.BILINEAR)
            cropped_im.save(im_save_path/f"{img_id}___{str(i).zfill(6)}.png")
            cropped_hm = hm[y:y+crop_size, x:x+crop_size]
            cropped_hm = PIL.Image.fromarray(cropped_hm).resize([resize_to, resize_to], resample=PIL.Image.BILINEAR)
            cropped_hm.save(hm_save_path/f"{img_id}___{str(i).zfill(6)}.png")
            
            
def load_data(folder, size=256, bs=16, tfms=get_transforms(do_flip=False, p_lighting=0.0, max_zoom=1.5)):
    data = (ImageImageList.from_folder(folder)
           .split_by_valid_func(lambda x: x.name[:-13] + ".jpg" in valid_fnames)
           .label_from_func(lambda x: folder/(x.name))
           .transform(tfms, tfm_y=True, size=size, resize_method=ResizeMethod.SQUISH)
           .databunch(bs=bs, num_workers=6)
           .normalize(imagenet_stats))
    data.c = 3
    return data