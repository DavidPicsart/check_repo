import os, multiprocessing, csv
from urllib.request import urlopen
from PIL import Image
# from io import StringIO
from io import BytesIO
import argparse
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def parse_data(data_file, id_colname, url_colname):
    csv_file = open(data_file, 'r')
    csv_reader = csv.reader(csv_file)
    cols = next(csv_reader)
    id_idx, url_idx = cols.index(id_colname), cols.index(url_colname)
    key_url_list = [[line[id_idx], line[url_idx]] for line in csv_reader]
    return key_url_list

def download_image(key_url, out_dir = '/home/david/Documents/Recommendations_DL/search_retrieval/check/', timeout = 50, resize = False):
    (key, url) = key_url
    filename = os.path.join(out_dir, '%s.jpg' % key)

    if os.path.exists(filename):
        print('Image %s already exists. Skipping download.' % filename)
        return

    try:
        response = urlopen(url, timeout=timeout)
        image_data = response.read()
    except:
        print('Warning: Could not download image %s from %s' % (key, url))
        return

    try:
        pil_image = Image.open(BytesIO(image_data))
    except:
        print('Warning: Failed to parse image %s' % key)
        return

    try:
        pil_image_rgb = pil_image.convert('RGB')
        if resize:
            if pil_image_rgb.size[0] != 299 or pil_image_rgb.size[1] != 299:
                 pil_image_rgb = pil_image_rgb.resize((299, 299))
    except:
        print('Warning: Failed to convert image %s to RGB' % key)
        return

    try:
        pil_image_rgb.save(filename, format='JPEG', quality=90)
    except:
        print('Warning: Failed to save image %s' % filename)
        return

def run(id_colname, url_colname, data_file, out_dir, num_proc = 50):
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    key_url_list = parse_data(data_file, id_colname, url_colname)
    pool = multiprocessing.Pool(processes=num_proc)
    pool.map(download_image, key_url_list)

run(id_colname = 'item_id', url_colname = 'url', data_file = '/home/david/Documents/Recommendations_DL/search_retrieval/data/urls_check.csv', out_dir = '/home/david/Documents/Recommendations_DL/search_retrieval/check/')