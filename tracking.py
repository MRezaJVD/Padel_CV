import numpy as np
import pandas as pd
import cv2
from shapely.geometry import Polygon
from courtplayers import CornersPoints, DiscardOutPersons

from shapely.geometry import Point, Polygon

############################ draw labeling ##########################
def WritePrevBox(r, img, color):
    for i in range(0, len(r)):
        bbox = r[i]['bbox']
        cv2.rectangle(img, (int(bbox[0]), int(bbox[1])),
                      (int(bbox[2]), int(bbox[3])), color)
    return img

######################### Fcn for tracking: IOU of polygon ##################################
def IOU_POLY(pol1_xy, pol2_xy):   # proportion of rectangle inside the trapezoid
    # Define each polygon
    polygon1_shape = Polygon(pol1_xy)
    polygon2_shape = Polygon(pol2_xy)

    # Calculate intersection and union, and the IOUimport pandas as pd
    polygon_intersection = polygon1_shape.intersection(polygon2_shape).area
    # polygon_union = polygon1_shape.union(polygon2_shape).area
    return polygon_intersection / polygon1_shape.area

###################### Fcn for tracking: box format changing #################################
def XYXY_BBOX(xyxy_bbox):
    d_x = xyxy_bbox[2] - xyxy_bbox[0]
    d_y = xyxy_bbox[3] - xyxy_bbox[1]
    p1 = [xyxy_bbox[0], xyxy_bbox[1]]
    p2 = [xyxy_bbox[0] + d_x, xyxy_bbox[1]]
    p3 = [xyxy_bbox[0], xyxy_bbox[1] + d_y]
    p4 = [xyxy_bbox[0] + d_x, xyxy_bbox[1] + d_y]
    bbox_poly = [p1, p2, p4, p3]
    return bbox_poly

########################### perspective matrix ###############################
def getPerspective(court_path):
    f = open(court_path, 'r')
    data = f.read()
    f.close()
    values = data.split(";")
    onImage = []
    for i in range(4):
        onImage.append([int(values[2*i]), int(values[2*i+1])])
    out_width = 10
    out_height = 20
    presp = np.float32(onImage)
    real = np.float32([[0, out_height], [out_width, out_height], [out_width, 0], [0, 0]])
    m = cv2.getPerspectiveTransform(presp, real)
    return m

########################### compose trajectory for occluded object ###############################
def ComposeTrajectory(Occ, court_path):
    G_0 =  Occ['init_est']# initial position of occluded player
    G_n =  Occ['final_est']# final position of occluded player
    frame_0 = Occ['init_occ_frame'] # initial frame of occluded player
    frame_n = Occ['end_occ_frame'] # final frame of occluded player
    player = Occ['who_is_occ']
    d_x = G_n[0] - G_0[0]
    d_y = G_n[1] - G_0[1]
    s = frame_n - frame_0
    D_composed = {"Frame": [], player+"_x":[], player+"_y":[], player+"_xm":[], player+"_ym":[]}
    for frame in range(frame_0, frame_n):
        D_composed['Frame'].append(str(frame))
        G_x = G_0[0] + (d_x/s) * (frame - frame_0)
        G_y = G_0[1] + (d_y/s) * (frame - frame_0)
        D_composed[player+"_x"].append(str(G_x))
        D_composed[player+"_y"].append(str(G_y))
        m = getPerspective(court_path)
        points = cv2.perspectiveTransform(np.float32([[[G_x, G_y]]]), m)[0][0]
        D_composed[player+"_xm"].append(str(points[0]))
        D_composed[player+"_ym"].append(str(points[1]))
    return D_composed

########################### track player function ###############################
def TrackPlayers(img, r1, r2, court_path, frame, Occ):  # r1:actual, r2:previous
    D_composed = None
    print(f'lock sitiation is: {Occ}')
    if Occ["is_locked"] == False:
        print(f'number of detected: {len(r1)}')
        IOU = np.zeros((len(r1), len(r2)))
        print('len(r1):', f'{len(r1)}', ', len(r2):', f'{len(r2)}')
        poser = 0
        # Objects with IOU=0 elimination
        for i in range(0, len(r1)):
            bbox_1 = [int(r1[i-poser]['bbox'][k]) for k in range(0, 4)]
            for j in range(0, len(r2)):
                bbox_2 = [int(r2[j]['bbox'][k]) for k in range(0, 4)]
                IOU[i-poser][j] = IOU_POLY(XYXY_BBOX(bbox_1), XYXY_BBOX(bbox_2))
            # or (r1[i-poser]['bbox'][4] < 0.6):
            print(f' unlocked IOU[{i}-{poser}] = \n{IOU[i-poser]}')
            if all([v == 0 for v in IOU[i-poser]]):
                print('an object with iou=0 detected: ', f'poser:{poser+1}')
                # print(r1[i-poser])
                r1.pop(i-poser)
                IOU = np.delete(IOU, (i-poser), axis=0)
                poser += 1
                # print(r1[i-poser]['bbox'][4])

    ############ in case of being in locked situation ############
    if Occ["is_locked"] == True:
        img, r1, _ = DiscardOutPersons(img, r1, court_path)
        IOU = np.zeros((len(r1), len(r2)))
        for i in range(0, len(r1)):
            bbox_1 = [int(r1[i]['bbox'][k]) for k in range(0, 4)]
            for j in range(0, len(r2)):
                bbox_2 = [int(r2[j]['bbox'][k]) for k in range(0, 4)]
                IOU[i][j] = IOU_POLY(XYXY_BBOX(bbox_1), XYXY_BBOX(bbox_2))
        print(f'locked IOU is ready: \n{IOU}')
        partner_id = [i for i, d in enumerate(r2) if d.get('player') == Occ["who_is_occ_partner"]] # partner player id with prev. frame
        print(f'partner_id in occlusion is found: {partner_id}')
        print(f'len of non-zero vector: {len(np.nonzero(IOU[:, partner_id])[0])}')
        if (len(np.nonzero(IOU[:, partner_id])[0]) == 1) and (len(r1) >= 4): # check if there is no bbox have iou with previous partner, except the current partner
            print('situation is locked, and separation is done')
            ZeroIoUs_TF = np.all(IOU == 0, axis=1) # all the detected zero-iou bboxes
            zero_ious_idx = np.where(ZeroIoUs_TF)[0] # index of all the detected zero-iou bboxes
            zero_ious_Conf = [r1[k]['bbox'][4] for k in zero_ious_idx] # all the conf. values of non-iou detected bboxes
            print(f'zero_ious_Conf is: {zero_ious_Conf}')
            if max(zero_ious_Conf) > 0.55: # The real detected person should have a high conf. value, unlike FPs
                print(f'situation is locked, and separation is done, and the occ person is detected, conf:{zero_ious_Conf}')
                occ_idx = zero_ious_idx[zero_ious_Conf.index(max(zero_ious_Conf))] # index of maximum conf. value for all non-iou detected bboxes
                print(f'occ_idx is: {occ_idx}')
                FPs_idx = [value for index, value in enumerate(zero_ious_idx) if value not in [occ_idx]]
                print(f'FPs_idx len: {len(FPs_idx)}')
                IOU = np.delete(IOU, FPs_idx, axis=0)
                r1 = [value for index, value in enumerate(r1) if index not in FPs_idx] # remove all non-maximum items
                r1[occ_idx]['player'] = Occ['who_is_occ']  # Now is time to put lable on occluded player
                print(f'the appeared occluded person is: {r1[occ_idx]}')
                print(f'Occ situation before is: {Occ["is_locked"]}')
                Occ["is_locked"] = False
                print(f'Occ situation after is: {Occ["is_locked"]}')
                Occ["end_occ_frame"] =  frame
                G_x = 1/2 * (r1[occ_idx]['keypoints'][11][0] + r1[occ_idx]['keypoints'][12][0])  # mean(L_hip.x, R_hip.x)
                G_y = 1/2 * (r1[occ_idx]['keypoints'][15][1] + r1[occ_idx]['keypoints'][16][1])  # mean(L_ankle.y, R_ankle.y)
                occ_player_final_pos = (G_x, G_y)
                Occ["final_est"] = occ_player_final_pos
                D_composed = ComposeTrajectory(Occ, court_path)
        else:  # we are in lock situation but still the Occ is not separated from the partner
            print('situation is locked, but still no separation')
            iou_with_partner = np.nonzero(IOU[:, partner_id])[0]
            partner_id_updated = np.argmax(IOU[partner_id]) # if the detected person is swaped
            occ_idx = [x for x in iou_with_partner if x != partner_id_updated]
            IOU = np.delete(IOU, occ_idx, axis=0)
            r1 = [value for index, value in enumerate(r1) if index not in occ_idx]

            ZeroIoUs_TF = np.all(IOU == 0, axis=1) # all the detected zero-iou bboxes
            zero_ious_idx = np.where(ZeroIoUs_TF)[0] # index of all the detected zero-iou bboxes
            IOU = np.delete(IOU, zero_ious_idx, axis=0)
            r1 = [value for index, value in enumerate(r1) if index not in zero_ious_idx]

    # Duplicated & glass reflecion bbox elimination
    ArgMax = np.argmax(IOU, axis=1)
    if (len(set(ArgMax)) != len(ArgMax)) and (len(r1) > 4): # (len(r1) > 4) is for avoiding elimination of new separated person
        print(f'Duplicated bbox detected: {ArgMax}')
        unique = np.unique(ArgMax)
        for val in unique:
            idx_maxs = np.where(ArgMax == val)[0]
            if len(idx_maxs) > 1:
                print(f'{idx_maxs} are duplicated idxs')
                for c in idx_maxs:   # remove glass reflecion bbox
                    G_x = 1/2 * (r1[c]['keypoints'][11][0] + r1[c]['keypoints'][12][0])  # mean(L_hip.x, R_hip.x)
                    G_y = 1/2 * (r1[c]['keypoints'][15][1] + r1[c]['keypoints'][16][1])  # mean(L_ankle.y, R_ankle.y)
                    point = Point(G_x, G_y)
                    court_corners, _ = CornersPoints(court_path)
                    if Polygon(court_corners).contains(point) == False:
                        r1.pop(c)
                        IOU = np.delete(IOU, c, axis=0)
                Maxs_Conf = [r1[k]['bbox'][4] for k in idx_maxs]  # remove all Duplicated bboxes
                no_Max_idx = idx_maxs[Maxs_Conf.index(min(Maxs_Conf))]
                r1.pop(no_Max_idx)
                IOU = np.delete(IOU, no_Max_idx, axis=0)
    # FP with IOU!=0 elimination
    poser = 0
    for j in range(0, len(r2)):
        # [i for i, e in enumerate(IOU[:,j]) if e!=0]
        ReverseIOU = np.nonzero(IOU[:, j])[0]
        if len(ReverseIOU) > 1:
            FPs_Conf = [r1[k]['bbox'][4] for k in ReverseIOU]
            no_FP = ReverseIOU[FPs_Conf.index(max(FPs_Conf))]
            for c in ReverseIOU:  # remove all non-maximum items
                if (c != no_FP) and (len(np.nonzero(IOU[c-poser, :])[0]) == 1) and (len(r1) > 4):
                    r1.pop(c-poser)
                    IOU = np.delete(IOU, c-poser, axis=0)
                    poser += 1

    # Missing detection due to motion blur
    if len(r1) < len(r2):
        print('Missing detection due to motion blur')
        for i in range(0, len(r2)):
            if len(np.nonzero(IOU[:, i])[0]) == 0:
                r1.append(r2[i])
                miss_row = np.zeros((1, len(r2)))
                miss_row[0, i] = 1
                IOU = np.append(IOU, miss_row, axis=0)
    # Computation of maximum IOU vector
    print(f'IOU after processing: \n{IOU}')
    MAXIOU = []
    L =  len(r1) #min(len(r1), len(r2)) # in case of appearance after occlusion, or missdetection
    for i in range(0, L):
        if not np.all(IOU[i] == 0): # after separation, we have a zero row that should be ignored during association
            maxiouarg = [n for n, m in enumerate(IOU[i]) if m == max(IOU[i])]
            MAXIOU.append(maxiouarg)
    print(MAXIOU)
    r1_swaped = []
    k = 0 # used for appreance after occlusion when the conf. value is high
    for s in range(0, L):
        if not np.all(IOU[s] == 0):
            # swapping r2 values based on r1
            r1_swaped.append(r1[MAXIOU.index([s-k]) + k])
            r1_swaped[s]['player'] = r2[s-k]['player']  # put the label based on r2
        else:
            r1_swaped.append(r1[s])
            k += 1

    print(f' len r1 = {len(r1)}')
    return r1_swaped, D_composed
