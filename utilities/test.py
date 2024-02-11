import numpy as np
import os
import sys

def main(datadir):

    print(np.__version__)


    labels = []

    idcs = [label_file for label_file in os.listdir(data_dir) if 'label' in label_file]

    print(idcs[:10])

    sizelim = 1.
    sizelim2 = 30
    sizelim3 = 40

    for idx in idcs:
        l = np.load(os.path.join(data_dir, '%s' %idx))
        if np.all(l==0):
            l=np.array([])
        labels.append(l)

    sample_bboxes = labels
    bboxes = []
    for i, l in enumerate(labels):

        if len(l) > 0 :
            for t in l:
                if t[3]>sizelim:
                    bboxes.append([np.concatenate([[i],t])])
                if t[3]>sizelim2:
                    bboxes+=[[np.concatenate([[i],t])]]*2
                if t[3]>sizelim3:
                    bboxes+=[[np.concatenate([[i],t])]]*4

    bboxes = np.concatenate(bboxes,axis = 0)

    print(bboxes[:10])

if __name__ == '__main__':

    data_dir = sys.argv[1]
    main(data_dir)