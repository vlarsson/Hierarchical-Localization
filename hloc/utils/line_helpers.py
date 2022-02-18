import numpy as np
import poselib


def compute_epipolar_IoU(F,l1,l2):
    l1_1 = np.r_[l1[0:2], [1.0]]
    l1_2 = np.r_[l1[2:4], [1.0]]

    l2_1 = np.r_[l2[0:2], [1.0]]
    l2_2 = np.r_[l2[2:4], [1.0]]

    # compute infinite lines
    inf1 = np.cross(l1_1, l1_2)
    inf2 = np.cross(l2_1, l2_2)

    # line segment lengths
    len1 = np.linalg.norm(l1_1 - l1_2)
    len2 = np.linalg.norm(l2_1 - l2_2)

    # compute intersection points
    e1_1 = np.cross(inf1, F.T @ l2_1)
    e1_2 = np.cross(inf1, F.T @ l2_2)
    e2_1 = np.cross(inf2, F @ l1_1)
    e2_2 = np.cross(inf2, F @ l1_2)

    e1_1 = e1_1 / e1_1[2]
    e1_2 = e1_2 / e1_2[2]
    e2_1 = e2_1 / e2_1[2]
    e2_2 = e2_2 / e2_2[2]

    # direction vectors
    v1 = l1_2 - l1_1
    v2 = l2_2 - l2_1
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)

    # Now we want to compute iou of (e1_1,e1_2) vs. (l1_1, l1_2) etc.
    c1 = np.dot(v1, e1_1 - l1_1) / len1
    c2 = np.dot(v1, e1_2 - l1_1) / len1
    if(c1 > c2):
        c1, c2 = c2, c1

    if c2 < 0 or c1 > 1.0:
        iou1 = 0
    else:
        iou1 = (np.min((c2,1.0)) - np.max((c1,0.0))) / (np.max((c2,1.0)) - np.min((c1,0.0)))


    c1 = np.dot(v2, e2_1 - l2_1) / len2
    c2 = np.dot(v2, e2_2 - l2_1) / len2
    if(c1 > c2):
        c1, c2 = c2, c1

    if c2 < 0 or c1 > 1.0:
        iou2 = 0
    else:
        iou2 = (np.min((c2,1.0)) - np.max((c1,0.0))) / (np.max((c2,1.0)) - np.min((c1,0.0)))

    return (iou1, iou2)


def match_lines_given_pose(pose, db_pose, q_lin2d, db_lin2d, db_lin3d, db_lin3d_ids, qcam, dbcam, tol=100.0, tol_iou=0.1, tol_angle=10):
    tol_rad = np.deg2rad(tol_angle)

    # We start by projecting lines
    proj_lin2d_inf = []
    proj_ok = []
    for l_3d in db_lin3d:
        p1 = pose.R @ l_3d[0:3] + pose.t
        p2 = pose.R @ l_3d[3:6] + pose.t
        z1 = p1[0:2] / p1[2]
        z2 = p2[0:2] / p2[2]
        z1 = qcam.world_to_image(z1)
        z2 = qcam.world_to_image(z2)

        l_inf = np.cross(np.r_[z1, [1.0]], np.r_[z2, [1.0]])
        l_inf = l_inf / np.linalg.norm(l_inf[0:2])

        proj_lin2d_inf.append(l_inf)

        if p1[2] < 0 or p2[2] < 0:
            proj_ok.append(False)
        else:
            proj_ok.append(True)

    # Compute essential and fundamental matrix
    q_K = qcam.calibration_matrix()
    db_K = dbcam.calibration_matrix()
    R = db_pose.R @ pose.R.T
    t = db_pose.t - R @ pose.t

    st = np.array([[0.0, -t[2], t[1]],
                   [t[2], 0.0, -t[0]],
                   [-t[1], t[0], 0.0]])
    E = st @ R
    F = np.linalg.inv(db_K.T) @ E @ np.linalg.inv(q_K)

    matches = []
    scores = []
    # Go through each line segment in the 2D image and find the closest projection
    for q_id, l2d in enumerate(q_lin2d):
        e1 = np.r_[l2d[0:2], [1.0]]
        e2 = np.r_[l2d[2:4], [1.0]]
        v = e2 - e1
        v = v / np.linalg.norm(v)
        # Endpoint-to-line distance
        d_inf = np.array([np.abs(l.dot(e1)) + np.abs(l.dot(e2)) for l in proj_lin2d_inf])
        ind = np.argsort(d_inf)

        for i in ind:
            if d_inf[i] > tol:
                break
            if not proj_ok[i]:
                continue
            epi_IoU = np.min(compute_epipolar_IoU(F, l2d, db_lin2d[i]))
            if epi_IoU < tol_iou:
                continue

            # compute angle
            theta = np.pi/2 - np.arccos(np.abs(v.dot(proj_lin2d_inf[i])))
            if theta > tol_rad:
                continue

            # We have the smallest element which have sufficient IoU
            matches.append([q_id, i])
            scores.append(d_inf[i])
            break


    return np.array(matches), np.array(scores)