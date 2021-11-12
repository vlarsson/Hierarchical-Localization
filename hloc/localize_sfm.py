import argparse
import logging
import numpy as np
from pathlib import Path
from collections import defaultdict
import h5py
from tqdm import tqdm
import pickle
import pycolmap
import torch

from .utils.read_write_model import read_model
from .utils.parsers import parse_image_lists, parse_retrieval, names_to_pair
from .utils.lines import *


def line_interpolate_depth(depth_map, lins, camera, samples=10, thr=0.1):
    h, w = depth_map.shape
    lins = lins.copy()
    lins[:,0:2] = lins[:,0:2] / np.array([[w-1, h-1]]) * 2 - 1
    lins[:,2:4] = lins[:,2:4] / np.array([[w-1, h-1]]) * 2 - 1

    # TODO REWRITE THIS

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



def do_covisibility_clustering(frame_ids, all_images, points3D):
    clusters = []
    visited = set()

    for frame_id in frame_ids:
        # Check if already labeled
        if frame_id in visited:
            continue

        # New component
        clusters.append([])
        queue = {frame_id}
        while len(queue):
            exploration_frame = queue.pop()

            # Already part of the component
            if exploration_frame in visited:
                continue
            visited.add(exploration_frame)
            clusters[-1].append(exploration_frame)

            observed = all_images[exploration_frame].point3D_ids
            connected_frames = set(
                j for i in observed if i != -1 for j in points3D[i].image_ids)
            connected_frames &= set(frame_ids)
            connected_frames -= visited
            queue |= connected_frames

    clusters = sorted(clusters, key=len, reverse=True)
    return clusters


def pose_from_cluster(qname, qinfo, db_ids, db_cameras, db_images, points3D,
                      feature_file, match_file, thresh,
                      line_feature_file=None, line_match_file=None, depth_maps_path=None):
    kpq = feature_file[qname]['keypoints'].__array__()
    kp_idx_to_3D = defaultdict(list)
    kp_idx_to_3D_to_db = defaultdict(lambda: defaultdict(list))
    num_matches = 0
    num_matches_lines = 0

    if not line_feature_file is None:
        lines_q = line_feature_file[qname]['line_segments'].__array__()

    # Lines are represented as [x1 y1 x2 y2] or [x1 y1 z1 x2 y2 z2] for 3D
    all_mlinq = []
    all_mlinr = []
    all_ml3d = []
    all_line_indices = []

    for i, db_id in enumerate(db_ids):
        db_name = db_images[db_id].name
        db_camera = db_cameras[db_images[db_id].camera_id]
        points3D_ids = db_images[db_id].point3D_ids
        if len(points3D_ids) == 0:
            logging.debug(f'No 3D points found for {db_name}.')
            continue

        pair = names_to_pair(qname, db_name)
        matches = match_file[pair]['matches0'].__array__()
        valid = np.where(matches > -1)[0]
        valid = valid[points3D_ids[matches[valid]] != -1]
        num_matches += len(valid)

        for idx in valid:
            id_3D = points3D_ids[matches[idx]]
            kp_idx_to_3D_to_db[idx][id_3D].append(i)
            # avoid duplicate observations
            if id_3D not in kp_idx_to_3D[idx]:
                kp_idx_to_3D[idx].append(id_3D)

        if line_feature_file is None or depth_maps_path is None
            continue

        # Extract 2D-3D line matches
        lines_db = line_feature_file[db_name]['line_segments'].__array__()
        m = line_match_file[pair]['matches0'].__array__()
        v = (m > -1)
        mlinq, mlinr  = lines_q[v], lines_db[m[v]]
        num_matches_lines += len(mlinq)

        if len(mlinq) == 0:
            continue       

        # Read depth maps
        depth_filename = db_name.replace('/','_')
        depth_filename = depth_filename[:-3] + '_depth.npz'
        depth_map = np.load(depth_maps_path + depth_filename)['depth']

        # Find 3D endpoints
        ml3d, valid = line_interpolate_depth(depth_map, mlinr, db_camera)          
        
        all_mlinq.append(mlinq[valid])
        all_mlinr.append(mlinr[valid])
        all_ml3d.append(ml3d[valid])
        all_line_indices.append(np.full(np.count_nonzero(valid), i))

        

    idxs = list(kp_idx_to_3D.keys())
    mkp_idxs = [i for i in idxs for _ in kp_idx_to_3D[i]]
    mkpq = kpq[mkp_idxs]
    mkpq += 0.5  # COLMAP coordinates

    mp3d_ids = [j for i in idxs for j in kp_idx_to_3D[i]]
    mp3d = [points3D[j].xyz for j in mp3d_ids]
    mp3d = np.array(mp3d).reshape(-1, 3)

    # mostly for logging and post-processing
    mkp_to_3D_to_db = [(j, kp_idx_to_3D_to_db[i][j])
                       for i in idxs for j in kp_idx_to_3D[i]]

    camera_model, width, height, params = qinfo
    cfg = {
        'model': camera_model,
        'width': width,
        'height': height,
        'params': params,
    }
    ret = pycolmap.absolute_pose_estimation(mkpq, mp3d, cfg, thresh)
    ret['cfg'] = cfg
    return ret, mkpq, mp3d, mp3d_ids, num_matches, (mkp_idxs, mkp_to_3D_to_db)


def main(reference_sfm, queries, retrieval, features, matches, results,
         ransac_thresh=12, covisibility_clustering=False,
         prepend_camera_name=False, line_features=None, line_matches=None, depth_maps=None):

    assert reference_sfm.exists(), reference_sfm
    assert retrieval.exists(), retrieval
    assert features.exists(), features
    assert matches.exists(), matches

    queries = parse_image_lists(queries, with_intrinsics=True)
    retrieval_dict = parse_retrieval(retrieval)

    logging.info('Reading 3D model...')
    db_cameras, db_images, points3D = read_model(str(reference_sfm))
    db_name_to_id = {image.name: i for i, image in db_images.items()}

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
    for qname, qinfo in tqdm(queries):
        if qname not in retrieval_dict:
            logging.warning(f'No images retrieved for query image {qname}. Skipping...')
            continue
        db_names = retrieval_dict[qname]
        db_ids = []
        for n in db_names:
            if n not in db_name_to_id:
                logging.warning(f'Image {n} was retrieved but not in database')
                continue
            db_ids.append(db_name_to_id[n])

        if covisibility_clustering:
            clusters = do_covisibility_clustering(db_ids, db_images, points3D)
            best_inliers = 0
            best_cluster = None
            logs_clusters = []
            for i, cluster_ids in enumerate(clusters):
                ret, mkpq, mp3d, mp3d_ids, num_matches, map_ = (
                        pose_from_cluster(
                            qname, qinfo, cluster_ids, db_images, points3D,
                            feature_file, match_file, thresh=ransac_thresh))
                if ret['success'] and ret['num_inliers'] > best_inliers:
                    best_cluster = i
                    best_inliers = ret['num_inliers']
                logs_clusters.append({
                    'db': cluster_ids,
                    'PnP_ret': ret,
                    'keypoints_query': mkpq,
                    'points3D_xyz': mp3d,
                    'points3D_ids': mp3d_ids,
                    'num_matches': num_matches,
                    'keypoint_index_to_db': map_,
                })
            if best_cluster is not None:
                ret = logs_clusters[best_cluster]['PnP_ret']
                poses[qname] = (ret['qvec'], ret['tvec'])
            logs['loc'][qname] = {
                'db': db_ids,
                'best_cluster': best_cluster,
                'log_clusters': logs_clusters,
                'covisibility_clustering': covisibility_clustering,
            }
        else:
            ret, mkpq, mp3d, mp3d_ids, num_matches, map_ = pose_from_cluster(
                qname, qinfo, db_ids, db_cameras, db_images, points3D,
                feature_file, match_file, line_feature_file=line_feature_file,
                line_match_file=line_match_file, depth_maps_path=depth_maps,
                thresh=ransac_thresh)

            if ret['success']:
                poses[qname] = (ret['qvec'], ret['tvec'])
            else:
                closest = db_images[db_ids[0]]
                poses[qname] = (closest.qvec, closest.tvec)
            logs['loc'][qname] = {
                'db': db_ids,
                'PnP_ret': ret,
                'keypoints_query': mkpq,
                'points3D_xyz': mp3d,
                'points3D_ids': mp3d_ids,
                'num_matches': num_matches,
                'keypoint_index_to_db': map_,
                'covisibility_clustering': covisibility_clustering,
            }

    logging.info(f'Localized {len(poses)} / {len(queries)} images.')
    logging.info(f'Writing poses to {results}...')
    with open(results, 'w') as f:
        for q in poses:
            qvec, tvec = poses[q]
            qvec = ' '.join(map(str, qvec))
            tvec = ' '.join(map(str, tvec))
            name = q.split('/')[-1]
            if prepend_camera_name:
                name = q.split('/')[-2] + '/' + name
            f.write(f'{name} {qvec} {tvec}\n')

    logs_path = f'{results}_logs.pkl'
    logging.info(f'Writing logs to {logs_path}...')
    with open(logs_path, 'wb') as f:
        pickle.dump(logs, f)
    logging.info('Done!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--reference_sfm', type=Path, required=True)
    parser.add_argument('--queries', type=Path, required=True)
    parser.add_argument('--features', type=Path, required=True)
    parser.add_argument('--matches', type=Path, required=True)
    parser.add_argument('--retrieval', type=Path, required=True)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--ransac_thresh', type=float, default=12.0)
    parser.add_argument('--covisibility_clustering', action='store_true')
    parser.add_argument('--prepend_camera_name', action='store_true')
    args = parser.parse_args()
    main(**args.__dict__)
