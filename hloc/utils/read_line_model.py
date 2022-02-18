
import os
import collections
import numpy as np
import struct
import argparse
from dataclasses import dataclass


@dataclass
class LineReconstruction:
    images: dict
    lines3D: dict

    def get_id(self, name):
        line_im_id = -1
        for (im_id, im) in self.images.items():
            if im['name'].endswith(name):
                line_im_id = im_id
                break
        return line_im_id


def read_images(path):
    images = {}
    with open(path, "r") as fid:
        num_images = int(fid.readline().strip())
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            (img_id, name) = line.split(' ')
            images[int(img_id)] = {'name': name}

    return images

def read_detections(path, images):
    with open(path, "r") as fid:
        num_images = int(fid.readline())
        for img_k in range(num_images):
            line = fid.readline().strip()
            (img_id, n_segs) = [int(x) for x in line.split(' ')]
            image = images[img_id]
            image['lines2D'] = []
            image['lines3D_ids'] = []
            for k in range(n_segs):
                line = fid.readline().strip()
                (x1, y1, x2, y2) = [float(x) for x in line.split(' ')]
                image['lines2D'].append([x1,y1,x2,y2])
                image['lines3D_ids'].append(-1)
            image['lines2D'] = np.array(image['lines2D'])

    return images

def read_tracks(path, images):
    lines3D = {}
    if not os.path.exists(path):
        return lines3D, images

    with open(path, "r") as fid:
        num_tracks = int(fid.readline().strip())
        for track_k in range(num_tracks):
            line = fid.readline().strip()
            (track_id, n_supporting_segs, n_supporting_images) = [int(x) for x in line.split(' ')]

            line = fid.readline().strip()
            (x1,y1,z1) = [float(x) for x in line.split(' ')]

            line = fid.readline().strip()
            (x2,y2,z2) = [float(x) for x in line.split(' ')]

            line = fid.readline().strip()
            image_ids = [int(x) for x in line.split(' ')]
            line = fid.readline().strip()
            line_ids = [int(x) for x in line.split(' ')]

            lines3D[track_id] = {
                'xyz': np.array([x1,y1,z1,x2,y2,z2]),
                'track': []
            }

            for (im_id, l_id) in zip(image_ids, line_ids):
                images[im_id]['lines3D_ids'][l_id] = track_id
                lines3D[track_id]['track'].append((im_id, l_id))

    return lines3D, images


def read_neighbours(path, images):
    if not os.path.exists(path):
        for _, im in images.items():
            im['neighbours'] = []
        return images

    with open(path, "r") as fid:
        num_images = int(fid.readline().strip())
        for im_k in range(num_images):
            line = fid.readline().strip()
            ids = [int(x) for x in line.split(' ')]

            image = images[ids[0]]
            image['neighbours'] = ids[1:]

    return images

def read_line_model(path, tag=''):
    images = read_images(path + '/' + tag + 'image_list.txt')

    images = read_detections(path + '/' + tag + 'detections.txt', images)

    lines3D, images = read_tracks(path + '/' + tag + 'alltracks.txt', images)

    images = read_neighbours(path + '/' + tag + 'neighbors.txt', images)

    return LineReconstruction(images, lines3D)