import json

ann_file = 'data/shift/continuous/videos/1x/test/front/det_2d_cocoformat.json'
dest_file = 'data/shift/continuous/videos/1x/test/front/det_2d_cocoformat_tmp.json'

with open(ann_file, 'r') as fp: 
    anns = json.load(fp)

print('Starting conversion')
for im in anns['images']:
    im['file_name'] = im['file_name'].replace(str(im['frame_id']).zfill(8), str(im['frame_id']//10).zfill(8))
    im['frame_id'] = im['frame_id'] // 10
print('Conversion done')

with open(dest_file, 'w') as fp:
    json.dump(anns, fp)

