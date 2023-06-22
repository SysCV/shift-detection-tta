import json

ann_file = 'data/shift/continuous/videos/1x/val/front/det_2d_cocoformat.json'
dest_file = 'data/shift/continuous/videos/1x/val/front/det_2d_cocoformat_tmp.json'

with open(ann_file, 'r') as fp: 
    anns = json.load(fp)

for im in anns['images']:
    im['file_name'] = im['file_name'].replace('_img_front.jpg', '.jpg')

with open(dest_file, 'w') as fp:
    json.dump(anns, fp)

