import math
import numpy as np
import cv2
import torch
from copy import deepcopy

from utils.datos import transformar_a_coordenadas_hip, deshacer_normalizacion
from shapely.geometry import Point, Polygon    # for calculating Trapezoid iou


def CornersPoints(court_path):
    ############# reading the court coordinates ###############
    with open(court_path, 'r') as f:
        corners = f.read().split(';')
    for C in range(0, len(corners) - 1):  # the last item is empty
        globals()['corner_%s_%s' % ((math.floor(C / 2)), ('x' if C % 2 ==0 else 'y'))] = int(corners[C])  # corner_x_0...7 corner_y_0 ...7
    court_corners = [[corner_0_x, corner_0_y], [corner_1_x, corner_1_y], [corner_2_x, corner_2_y], [corner_3_x, corner_3_y]]
    court_all_points = [[corner_0_x, corner_0_y], [corner_1_x, corner_1_y], [corner_2_x, corner_2_y], [corner_3_x, corner_3_y], 
        [corner_4_x, corner_4_y], [corner_5_x, corner_5_y], [corner_6_x, corner_6_y], [corner_7_x, corner_7_y]]
    return court_corners, court_all_points

def DiscardOutPersons(img, r, court_path):
    court_corners, court_all_points = CornersPoints(court_path)
    pts = np.array(court_corners, np.int32)
    img = cv2.polylines(img, [pts], True, (0, 0, 255), 3)  # draw courte border
    ############# excluding of out-court persons #############
    poser = 0  # for shift the index after discarding some non-useful detections
    for p in range(0, len(r)):
        G_x = 1/2 * (r[p-poser]['keypoints'][11][0] + r[p-poser]['keypoints'][12][0])  # mean(L_hip.x, R_hip.x)
        G_y = 1/2 * (r[p-poser]['keypoints'][15][1] + r[p-poser]['keypoints'][16][1])  # mean(L_ankle.y, R_ankle.y)
        point = Point(G_x, G_y)
        if (Polygon(court_corners).contains(point) == False) or (r[p-poser]['bbox'][4] < 0.5):
            r.pop(p-poser)
            poser += 1  
    return img, r, court_all_points


def PlayerPosition(r, court_all_points, frame):
    # Horizental center line
    Net_y = 1/2 * (court_all_points[6][1] + court_all_points[7][1])
    # Vertical center line
    C_x = 1/2 * (court_all_points[0][0] + court_all_points[1][0])
    Upper_Line = 1/2 * (court_all_points[0][1] + court_all_points[1][1])
    Lower_Line = 1/2 * (court_all_points[2][1] + court_all_points[3][1])
    for num in range(0, len(r)):
        G_x = 1/2 * (r[num]['keypoints'][11][0] + r[num]['keypoints'][12][0])  # mean(L_hip.x, R_hip.x)
        G_y = 1/2 * (r[num]['keypoints'][15][1] + r[num]['keypoints'][16][1])  # mean(L_ankle.y, R_ankle.y)
        if G_y < Net_y:
            if G_x < C_x:
                r[num]['player'] = 'TL'
            else:
                r[num]['player'] = 'TR'
        else:
            if G_x < C_x:
                r[num]['player'] = 'BL'
            else:
                r[num]['player'] = 'BR'
                
    ###### in case both players in 1 side ######
    G_x = []
    PL_POS = []
    aus_serve = False
    occ_player = None
    occ_partner = None
    occ_player_id = None
    occ_player_init_pos = None
    
    for pl in range(len(r)):
        PL_POS.append(r[pl]['player'])
        G_x.append(1/2 * (r[pl]['keypoints'][11][0] + r[pl]['keypoints'][12][0]))
    print(G_x, PL_POS)
    A = [pl_ind for pl_ind, pl_label in enumerate(PL_POS) if pl_label == 'TL']
    B = [pl_ind for pl_ind, pl_label in enumerate(PL_POS) if pl_label == 'TR']
    C = [pl_ind for pl_ind, pl_label in enumerate(PL_POS) if pl_label == 'BL']
    D = [pl_ind for pl_ind, pl_label in enumerate(PL_POS) if pl_label == 'BR']
    #print(f'A, B, C, D: {A}, {B}, {C}, {D}')
    if len(A) == 2:
        aus_serve = True
        occ_player = 'TR'
        occ_partner = 'TL'
        min_X_ind = G_x.index(min(G_x[A[0]], G_x[A[1]]))
        max_X_ind = G_x.index(max(G_x[A[0]], G_x[A[1]]))
        r[max_X_ind]['player'] = 'TR'
        occ_player_id = max_X_ind
        occ_partner_id = min_X_ind
        r.pop(occ_player_id)   # remove the occluded player
        occ_player_init_pos = (G_x[occ_partner_id], Upper_Line)
        
    if len(B) == 2:
        aus_serve = True
        occ_player = 'TL'
        occ_partner = 'TR'
        min_X_ind = G_x.index(min(G_x[B[0]], G_x[B[1]]))
        max_X_ind = G_x.index(max(G_x[B[0]], G_x[B[1]]))
        r[min_X_ind]['player'] = 'TL'
        occ_player_id = min_X_ind
        occ_partner_id = max_X_ind
        r.pop(occ_player_id)   # remove the occluded player 
        occ_player_init_pos = (G_x[occ_partner_id], Upper_Line)
        
    if len(C) == 2:
        #aus_serve = True
        #occ_player = 'BL'
        min_X_ind = G_x.index(min(G_x[C[0]], G_x[C[1]]))
        max_X_ind = G_x.index(max(G_x[C[0]], G_x[C[1]]))
        r[max_X_ind]['player'] = 'BR'
        #occ_player_id = min_X_ind
        #occ_partner_id = max_X_ind
        #r.remove(r[occ_player_id])   # remove the occluded player
        #occ_player_init_pos = ??
            
    if len(D) == 2:
    	#aus_serve = True
    	#occ_player = 'BR'
        min_X_ind = G_x.index(min(G_x[D[0]], G_x[D[1]]))
        max_X_ind = G_x.index(max(G_x[D[0]], G_x[D[1]]))
        r[min_X_ind]['player'] = 'BL'
        #occ_player_id = max_X_ind
        #occ_partner_id = min_X_ind
        #r.remove(r[occ_player_id])   # remove the occluded player
        #occ_player_init_pos = ??
        
    ###### in case during the occlusion, only 1 box is detected ######
    if (len(A)==1) and (len(B)==0):
        aus_serve = True
        occ_player = 'TR'
        occ_partner = 'TL'
        occ_player_init_pos = (G_x[A[0]], Upper_Line)
        
    if (len(A)==0) and (len(B)==1):
        aus_serve = True
        occ_player = 'TL'
        occ_partner = 'TR'
        occ_player_init_pos = (G_x[B[0]], Upper_Line)
        
    Occ = dict()
    Occ["is_locked"] = aus_serve  # False or True
    Occ["init_occ_frame"] = frame if aus_serve else None
    Occ["who_is_occ"] = occ_player  # None or 'TL' or 'TR'
    Occ["who_is_occ_partner"] = occ_partner # None or 'TL' or 'TR'
    Occ["init_est"] = occ_player_init_pos # None or (X,Y)
    return r, Occ


def DrawLabel(r, img):
    color = [(255, 0, 0), (0, 255, 0), (0, 0, 255),(255, 255, 0)]  # , (255,0,255)]
    for i in range(0, len(r)):
        bbox = r[i]['bbox']
        cv2.rectangle(img, (int(bbox[0]), int(bbox[1])),(int(bbox[2]), int(bbox[3])), color[i])
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, r[i]['player'], (int((bbox[0]+bbox[2])/2-10),int(bbox[1])-5), font, 0.8, color[i], 2, cv2.LINE_AA)
    return img

def DrawPosition(r, img):
    color = (0, 0, 0)
    for i in range(0, len(r)):
        G_x = 1/2 * (r[i]['keypoints'][11][0] + r[i]['keypoints'][12][0])  # mean(L_hip.x, R_hip.x)
        G_y = 1/2 * (r[i]['keypoints'][15][1] + r[i]['keypoints'][16][1])  # mean(L_ankle.y, R_ankle.y)
        cv2.circle(img, (int(G_x), int(G_y)), 4, color, -1)

def DrawPositionNN(pose_results_labeled, model, img):
    color = (0, 255, 255)
    X = []
    hip = []
    r = deepcopy(pose_results_labeled)
    for i in range(0, len(r)):
        kpts = r[i]['keypoints'][:,:2]
        X.append(kpts)
        hip_x = 1/2 * (kpts[11][0] + kpts[12][0])
        hip_y = 1/2 * (kpts[11][1] + kpts[12][1])
        hip.append([hip_x, hip_y])
    for i in range(0, len(r)):
        for j in range(0, len(X[i])):
            X[i][j] = transformar_a_coordenadas_hip(hip[i], X[i][j])
    X = torch.tensor(X, dtype=torch.float32)
    X = torch.reshape(X, (len(X), X.shape[1]*X.shape[2]))
    predicciones = model(X)
    predicciones = predicciones.tolist()
    for i in range(0, len(predicciones)):
        predicciones[i] = deshacer_normalizacion(predicciones[i], hip[i])

    for i in range(0, len(predicciones)):
        cv2.circle(img, (int(predicciones[i][0]), int(predicciones[i][1])), 4, color, -1)