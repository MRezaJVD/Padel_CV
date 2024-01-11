from sheet import Sheet, PointsID
from perspose import PersPos
from courtplayers import DiscardOutPersons, PlayerPosition, DrawLabel, DrawPosition, DrawPositionNN
from tracking import TrackPlayers, WritePrevBox
from results import print_and_save_results

import os
import cv2
import torch
from argparse import ArgumentParser

from mmdet.apis import inference_detector, init_detector
from mmpose.datasets import DatasetInfo
from mmpose.apis import (inference_top_down_pose_model, init_pose_model, process_mmdet_results, vis_pose_result)
from mmtrack.apis import init_model, inference_mot
############################## parsing args ##############################
parser = ArgumentParser()
parser.add_argument('det_config', help='Config file for detection')
parser.add_argument('det_checkpoint', help='Checkpoint file for detection')
parser.add_argument('pose_config', help='Config file for pose estimation')
parser.add_argument('pose_checkpoint', help='Checkpoint file for pose estimation')
parser.add_argument('trck_config', help='Config file for tracking')
parser.add_argument('trck_checkpoint', help='Checkpoint file for tracking')
parser.add_argument('spread_sheet', help='Informative excel file of the match')
parser.add_argument('video_path', type=str, help='Video path')
parser.add_argument('court_path', type=str,help='Court corners coordinates')
parser.add_argument('nnmodel', help='Neural Network model for position estimation')
parser.add_argument('--show', action='store_true',default=False, help='whether to show img')
parser.add_argument('--device', default='cuda:0',help='Device used for inference')
parser.add_argument('--det-cat-id', type=int, default=1,help='Category id for bounding box detection model')
parser.add_argument('--bbox-thr', type=float, default=0.1,help='Bounding bbox score threshold')  # antes: 0.3
parser.add_argument('--kpt-thr', type=float, default=0.1,help='Keypoint score threshold')  # antes: 0.3
args = parser.parse_args()

############################## Read sheet ######################################
Padel_SpSheet = Sheet(args.spread_sheet)
df_plays, N_games = Padel_SpSheet.read_playsheet()
#df_shots, N_shots = Padel_SpSheet.read_shotsheet()
persp = PersPos(args.court_path, args.spread_sheet)
m = persp.getPerspective()
play = PointsID(args.spread_sheet)
############################ Preparing the video ################################
cap = cv2.VideoCapture(args.video_path)  # Read the video
cv2.namedWindow('frame', cv2.WINDOW_NORMAL)  # Resizable window
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # Total frames
all_frames = int(df_plays['End frame'].iloc[-1])
in_game_times = int(df_plays['Duration'].sum(axis=0))
print(f"Percentage of in-games frames over all the match is: {in_game_times/all_frames*100}")
fps = cap.get(cv2.CAP_PROP_FPS)
width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
det_model = init_detector(args.det_config, args.det_checkpoint, device=args.device)
pose_model = init_pose_model(args.pose_config, args.pose_checkpoint, device=args.device)
#track_model = init_model(args.trck_config, args.trck_checkpoint, device=args.device)
nnmodel = torch.load(args.nnmodel)
dataset = pose_model.cfg.data['test']['type']
assert (dataset == 'TopDownCocoDataset')

############################# Creating folder for saving #######################
out_basename = os.path.join(os.path.basename(args.pose_checkpoint[0:-4]).split(
    "-")[0], os.path.basename(args.det_checkpoint[0:-4]).split("-")[0])
os.makedirs(os.path.join("../Padel_Vis_full_match/", out_basename), exist_ok=True)

############################# Run the analysis #######################
frame_no = 0
game_no = 0
shot_no = 0
while(cap.isOpened()):
    while (df_plays["Alternative_Cam"][game_no] == '*'): # or (df_plays["One_Side_Serve"][game_no] == '*'):
        play.PointsID2PlaySheet(game_no)
        frame_no = int(df_plays["End frame"][game_no])
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
        game_no += 1
    flag, img = cap.read()
    if not flag:
        break
    serve_frame = int(df_plays["Start frame"][game_no])
    end_frame = int(df_plays["End frame"][game_no])
    ##################### Before serve ###################
    if (frame_no < serve_frame):
        print(f" Out-ranged frame: {frame_no}")
        cv2.imshow('frame', img)
        if cv2.waitKey(25) == ord('q'):
            break
    ##################### Serve and game ##################
    if ((frame_no >= serve_frame) and (frame_no <= end_frame)):
        print(f" Processing frame {frame_no}")
        os.makedirs(os.path.join(os.path.join("../Padel_Vis_full_match/", out_basename), f'game_{game_no+1}'), exist_ok=True)
        mmdet_results = inference_detector(det_model, img)
        person_results = process_mmdet_results(mmdet_results, args.det_cat_id) # 1 is the person category

        ########################################use of mmtrack##############################################
        #track_results = inference_mot(track_model, img, frame_no)
        #person_track_results = process_mmdet_results(track_results, args.det_cat_id) # 1 is the person category
        #person_track_results = [{'bbox': item['bbox'][1:]} for item in person_track_results] # remove the object number
        ####################################################################################################
        pose_results, returned_outputs = inference_top_down_pose_model(
            pose_model,
            img,
            person_results,  # before: person_results
            bbox_thr=args.bbox_thr,
            format='xyxy',
            dataset=dataset)
        #who_is_hitter, shot_no = Padel_SpSheet.ShotChecker(shot_no, frame_no) # Shot & hitter identifier 
        if (frame_no == serve_frame):
            print(f'************** Serve frame is: ***********: {frame_no}')
            img, pose_results, court_all_points = DiscardOutPersons(img, pose_results, args.court_path)
            pose_results_labeled, Occ = PlayerPosition(pose_results, court_all_points, frame_no)
            img = DrawLabel(pose_results_labeled, img)
            pose_results_labeled = persp.FramePerspCalc(pose_results_labeled, m, frame_no)
        else:
            img = WritePrevBox(pose_results_labeled, img, (0, 0, 255))
            try:
                pose_results_labeled, D_composed = TrackPlayers(img, pose_results, pose_results_labeled, args.court_path, frame_no, Occ)  # inherit of the label
                print(f'D_composed is: \n{D_composed}')
                img = DrawLabel(pose_results_labeled, img)
                pose_results_labeled = persp.FramePerspCalc(pose_results_labeled, m, frame_no)
                if D_composed is not None: persp.FillOccPos(D_composed) 
            except Exception as e:
                print('an exception hass occured')
                print(e)
        #if who_is_hitter is not None:
        #    print('a shot is detected')
        #    pass 
        print_and_save_results(pose_results_labeled, os.path.join(os.path.join(os.path.join("../Padel_Vis_full_match/", out_basename), f'game_{game_no+1}'), f'{frame_no}.json'))
        out_frame = os.path.join(os.path.join(os.path.join("../Padel_Vis_full_match/", out_basename), f'game_{game_no+1}'), f'{frame_no}.jpg')
        DrawPosition(pose_results_labeled, img)  # draw the position of the players
        DrawPositionNN(pose_results_labeled, nnmodel, img)  # draw the position of the players
        vis_pose_result(
            pose_model,
            img,
            pose_results_labeled, # before: pose_results
            dataset=dataset,
            kpt_score_thr=args.kpt_thr,
            show=args.show,
            out_file=out_frame)
    ################### End of game ###################
    if (frame_no == end_frame):
        print(f" end of the game is frame: {frame_no}")
        gamePoses = persp.GetGamePositions()  # get position dict
        Padel_SpSheet.WritePos2Sheet(gamePoses)
        persp.ResetDict()
        play.PointsID2PlaySheet(game_no)
        game_no += 1

    if (game_no == N_games):  # check if all the game is read
        print(f" ############### End of the Padel analysis task ############### ")
        break
    frame_no += 1
cap.release()
cv2.destroyAllWindows()
