import json
import os.path as osp
import os

import collections

frame_privacy_dict = collections.defaultdict(dict)
attr_mapping_dict = {'gender':'a4_gender', 'face':'a10_face_partial', 'nudity':'a12_semi_nudity',
                     'skin_color': 'a17_color', 'relationship':'a64_rel_personal'}
json_files = os.listdir('raw_pa_hmdb51_annos')
vid_lst = []
for jf in json_files:
    action = jf[:-5]
    with open(osp.join('raw_pa_hmdb51_annos', jf)) as handle:
        anno = json.load(handle)
        for vid in anno.keys():
            print(vid)
            vid_lst.append(vid)
            for attr in anno[vid].keys():
                if attr == "review" or attr == "note":
                    continue
                for split in anno[vid][attr]:
                    start, end, value = split[0], split[1], split[2]
                    for i in range(start, end+1):
                        frame_name = '{}_{}_{}'.format(action,vid[:-4],i)
                        frame_privacy_dict[frame_name][attr_mapping_dict[attr]] = value
                        #print("{}: {}, {}, {}".format(frame_name, attr, i, value))

import json
if not os.path.exists('pa_hmdb51_annos'):
    os.makedirs('pa_hmdb51_annos')
for frame in frame_privacy_dict.keys():
    print(frame)
    privacy_attrs = []
    for attr in frame_privacy_dict[frame].keys():
        if frame_privacy_dict[frame][attr] != 0:
            privacy_attrs.append(attr)
    with open(osp.join('pa_hmdb51_annos', '{}.json'.format(frame)), 'w') as fp:
        json.dump({"image_path":os.path.join('pa_hmdb51_frames', frame+'.png'), "labels":privacy_attrs}, fp, indent=4)