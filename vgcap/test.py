import os
import json
import numpy as np

sg_data_dir = "/home/henry/Datasets/coco/coco_cmb_vrg_final/"
input_json = "/home/henry/Datasets/coco/cocotalk.json"

vrg_vocab = {v: k for k, v in json.load(open(input_json))['ix_to_word'].items()}

if __name__ == "__main__":
    current_max = 0
    for i, file in enumerate(os.listdir(sg_data_dir)):
        sg_use = np.load(sg_data_dir + file)
        if sg_use['prela'].shape[0] == 0:
            triplet_p = np.array([[0, 0, vrg_vocab['near']]], dtype=sg_use['prela'].dtype)
        else:
            triplet_p = sg_use['prela']

        local_max = triplet_p[:,2:].max()
        if local_max > current_max:
            current_max = local_max
        print(f"{i}/{len(os.listdir(sg_data_dir))}\t", local_max, "\t", current_max)

    print(f"\n\n{current_max}")

