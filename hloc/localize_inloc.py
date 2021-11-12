import argparse
from pathlib import Path
import numpy as np
import h5py
from scipy.io import loadmat
import torch
from tqdm import tqdm
import logging
import pickle
import cv2
import pycolmap
import poselib

from .utils.parsers import parse_retrieval, names_to_pair
from .utils.read_write_model import rotmat2qvec
from .utils.lines import *

def interpolate_scan(scan, kp):
    h, w, c = scan.shape
    kp = kp / np.array([[w-1, h-1]]) * 2 - 1
    assert np.all(kp > -1) and np.all(kp < 1)
    scan = torch.from_numpy(scan).permute(2, 0, 1)[None]
    kp = torch.from_numpy(kp)[None, None]
    grid_sample = torch.nn.functional.grid_sample

    # To maximize the number of points that have depth:
    # do bilinear interpolation first and then nearest for the remaining points
    interp_lin = grid_sample(
        scan, kp, align_corners=True, mode='bilinear')[0, :, 0]
    interp_nn = torch.nn.functional.grid_sample(
        scan, kp, align_corners=True, mode='nearest')[0, :, 0]
    interp = torch.where(torch.isnan(interp_lin), interp_nn, interp_lin)
    valid = ~torch.any(torch.isnan(interp), 0)

    kp3d = interp.T.numpy()
    valid = valid.numpy()
    return kp3d, valid

def line_interpolate_scan(scan, lins, samples=10, thr=0.1):
    h, w, c = scan.shape
    lins = lins.copy()
    lins[:,0:2] = lins[:,0:2] / np.array([[w-1, h-1]]) * 2 - 1
    lins[:,2:4] = lins[:,2:4] / np.array([[w-1, h-1]]) * 2 - 1

    assert np.all(lins >= -1) and np.all(lins <= 1)
    scan = torch.from_numpy(scan).permute(2, 0, 1)[None]
    grid_sample = torch.nn.functional.grid_sample

    kp3d = []
    valid = []

    for k, lin in enumerate(lins):
        x1 = lin[0:2]
        x2 = lin[2:4]

        # interpolate samples
        xx = np.array([x1 + (k/(samples-1))*(x2-x1) for k in range(samples)])

        # TODO: This code should probably refactored so that there is only a single
        #       call to grid_sample
        xx = torch.from_numpy(xx.astype(np.float))[None, None]

        # To maximize the number of points that have depth:
        # do bilinear interpolation first and then nearest for the remaining points
        interp_lin = grid_sample(
            scan, xx, align_corners=True, mode='bilinear')[0, :, 0]
        interp_nn = torch.nn.functional.grid_sample(
            scan, xx, align_corners=True, mode='nearest')[0, :, 0]
        interp = torch.where(torch.isnan(interp_lin), interp_nn, interp_lin)
        v = ~torch.any(torch.isnan(interp), 0)

        if v.sum() < 2:
            kp3d.append(np.zeros(6))
            valid.append(False)
        else:
            # We have at least 2 points, try to robustly fit a line
            interp = interp.numpy().T
            line = fit_line(interp[v], thr)
            if line is None:
                kp3d.append(np.zeros(6))
                valid.append(False)
            else:
                kp3d.append(line)
                valid.append(True)

    kp3d = np.array(kp3d)    
    return kp3d, np.array(valid)




def get_scan_pose(dataset_dir, rpath):
    split_image_rpath = rpath.split('/')
    floor_name = split_image_rpath[-3]
    scan_id = split_image_rpath[-2]
    image_name = split_image_rpath[-1]
    building_name = image_name[:3]

    path = Path(
        dataset_dir, 'database/alignments', floor_name,
        f'transformations/{building_name}_trans_{scan_id}.txt')
    with open(path) as f:
        raw_lines = f.readlines()

    P_after_GICP = np.array([
        np.fromstring(raw_lines[7], sep=' '),
        np.fromstring(raw_lines[8], sep=' '),
        np.fromstring(raw_lines[9], sep=' '),
        np.fromstring(raw_lines[10], sep=' ')
    ])

    return P_after_GICP


def pose_from_cluster(dataset_dir, q, retrieved, feature_file, match_file,
                      skip=None, line_feature_file=None, line_match_file=None):
    height, width = cv2.imread(str(dataset_dir / q)).shape[:2]
    cx = .5 * width
    cy = .5 * height
    focal_length = 4032. * 28. / 36.

    all_mkpq = []
    all_mkpr = []
    all_mkp3d = []
    all_indices = []

    # Lines are represented as [x1 y1 x2 y2] or [x1 y1 z1 x2 y2 z2] for 3D
    all_mlinq = []
    all_mlinr = []
    all_ml3d = []
    all_line_indices = []

    kpq = feature_file[q]['keypoints'].__array__()
    num_matches = 0
    num_matches_lines = 0
    if not line_feature_file is None:
        lines_q = line_feature_file[q]['line_segments'].__array__()

    for i, r in enumerate(retrieved):
        kpr = feature_file[r]['keypoints'].__array__()
        pair = names_to_pair(q, r)
        m = match_file[pair]['matches0'].__array__()
        v = (m > -1)

        if skip and (np.count_nonzero(v) < skip):
            continue

        mkpq, mkpr = kpq[v], kpr[m[v]]
        num_matches += len(mkpq)

        scan_r = loadmat(Path(dataset_dir, r + '.mat'))["XYZcut"]
        mkp3d, valid = interpolate_scan(scan_r, mkpr)
        Tr = get_scan_pose(dataset_dir, r)
        mkp3d = (Tr[:3, :3] @ mkp3d.T + Tr[:3, -1:]).T

        all_mkpq.append(mkpq[valid])
        all_mkpr.append(mkpr[valid])
        all_mkp3d.append(mkp3d[valid])
        all_indices.append(np.full(np.count_nonzero(valid), i))

        if line_feature_file is None:
            continue

        # Extract 2D-3D line matches
        lines_db = line_feature_file[r]['line_segments'].__array__()
        m = line_match_file[pair]['matches0'].__array__()
        v = (m > -1)
        mlinq, mlinr  = lines_q[v], lines_db[m[v]]
        num_matches_lines += len(mlinq)

        if len(mlinq) == 0:
            continue

        # TODO: Fix this in the feature extraction instead...
        #       (Sometimes LSD segments are outside the image? (+/- 1px))
        mlinr[:,0] = np.clip(mlinr[:,0], 0.0, scan_r.shape[1]-1.0)
        mlinr[:,1] = np.clip(mlinr[:,1], 0.0, scan_r.shape[0]-1.0)
        mlinr[:,2] = np.clip(mlinr[:,2], 0.0, scan_r.shape[1]-1.0)
        mlinr[:,3] = np.clip(mlinr[:,3], 0.0, scan_r.shape[0]-1.0)

        # Find 3D endpoints
        ml3d, valid = line_interpolate_scan(scan_r, mlinr)                
        ml3d[:,0:3] = (Tr[:3, :3] @ ml3d[:,0:3].T + Tr[:3, -1:]).T
        ml3d[:,3:6] = (Tr[:3, :3] @ ml3d[:,3:6].T + Tr[:3, -1:]).T        

        all_mlinq.append(mlinq[valid])
        all_mlinr.append(mlinr[valid])
        all_ml3d.append(ml3d[valid])
        all_line_indices.append(np.full(np.count_nonzero(valid), i))




    all_mkpq = np.concatenate(all_mkpq, 0)
    all_mkpr = np.concatenate(all_mkpr, 0)
    all_mkp3d = np.concatenate(all_mkp3d, 0)
    all_indices = np.concatenate(all_indices, 0)

    all_mlinq = np.concatenate(all_mlinq, 0)
    all_mlinr = np.concatenate(all_mlinr, 0)
    all_ml3d = np.concatenate(all_ml3d, 0)
    all_line_indices = np.concatenate(all_line_indices, 0)

    cfg = {
        'model': 'SIMPLE_PINHOLE',
        'width': width,
        'height': height,
        'params': [focal_length, cx, cy]
    }

    ret = poselib.estimate_absolute_pose_pnpl(all_mkpq, all_mkp3d, all_mlinq[:,0:2], all_mlinq[:,2:4], all_ml3d[:,0:3], all_ml3d[:,3:6], cfg, 48.00)
    if ret['success']:
        # make it compatible with pycolmap output
        ret['qvec'] = rotmat2qvec(ret['pose'].R)
        ret['tvec'] = ret['pose'].t
        ret.pop('pose')

    ret['cfg'] = cfg
    return ret, all_mkpq, all_mkpr, all_mkp3d, all_indices, num_matches, all_mlinq, all_mlinr, all_ml3d, all_line_indices, num_matches_lines


def main(dataset_dir, retrieval, features, matches, results,
         skip_matches=None, line_features=None, line_matches=None):

    assert retrieval.exists(), retrieval
    assert features.exists(), features
    assert matches.exists(), matches

    retrieval_dict = parse_retrieval(retrieval)
    queries = list(retrieval_dict.keys())

    feature_file = h5py.File(features, 'r')
    match_file = h5py.File(matches, 'r')

    if line_features is None:
        line_feature_file = None
        line_match_file = None
    else:
        assert line_features.exists()
        assert line_matches.exists()
        line_feature_file = h5py.File(line_features)
        line_match_file = h5py.File(line_matches)

    poses = {}
    logs = {
        'features': features,
        'matches': matches,
        'retrieval': retrieval,
        'loc': {},
    }
    logging.info('Starting localization...')
    for q in tqdm(queries):
        db = retrieval_dict[q]
        ret, mkpq, mkpr, mkp3d, indices, num_matches, mlinq, mlinr, ml3d, line_indices, num_line_matches = pose_from_cluster(
            dataset_dir, q, db, feature_file, match_file, skip_matches, line_feature_file, line_match_file)

        poses[q] = (ret['qvec'], ret['tvec'])
        logs['loc'][q] = {
            'db': db,
            'PnP_ret': ret,
            'keypoints_query': mkpq,
            'keypoints_db': mkpr,
            '3d_points': mkp3d,
            'indices_db': indices,
            'num_matches': num_matches
        }
        if not line_feature_file is None:
            logs['loc'][q].update({
                'lines_query': mlinq,
                'lines_db': mlinr,
                '3d_lines': ml3d,
                'line_indices_db': line_indices,
                'num_line_matches': num_line_matches
            })

    logging.info(f'Writing poses to {results}...')
    with open(results, 'w') as f:
        for q in queries:
            qvec, tvec = poses[q]
            qvec = ' '.join(map(str, qvec))
            tvec = ' '.join(map(str, tvec))
            name = q.split("/")[-1]
            f.write(f'{name} {qvec} {tvec}\n')

    logs_path = f'{results}_logs.pkl'
    logging.info(f'Writing logs to {logs_path}...')
    with open(logs_path, 'wb') as f:
        pickle.dump(logs, f)
    logging.info('Done!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=Path, required=True)
    parser.add_argument('--retrieval', type=Path, required=True)
    parser.add_argument('--features', type=Path, required=True)
    parser.add_argument('--matches', type=Path, required=True)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--skip_matches', type=int)
    args = parser.parse_args()
    main(**args.__dict__)
