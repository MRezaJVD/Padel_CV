import cv2
import pandas as pd
import numpy as np

class PersPos():
    def __init__(self, coord_file, sheets):
        self.Corners = coord_file
        # create a dict,keys() are df columns
        pos_sheet = pd.read_excel(sheets, sheet_name="Positions")
        self.D = dict([(key, []) for key in pos_sheet.columns])

    def ResetDict(self):
        DictKeys = self.D.keys()
        self.D = dict([(key, []) for key in DictKeys])

    def getPerspective(self):
        f = open(self.Corners, 'r')
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

    def FramePerspCalc(self, r, m, frame_no):
        self.D["Frame"].append(str(frame_no))
        max_len = 0
        for i in range(0, len(r)):
            G_x = 1/2 * (r[i]['keypoints'][11][0] + r[i]['keypoints'][12][0])  # mean(L_hip.x, R_hip.x)
            G_y = 1/2 * (r[i]['keypoints'][15][1] + r[i]['keypoints'][16][1])  # mean(L_ankle.y, R_ankle.y)
            self.D[r[i]['player']+"_x"].append(str(G_x))
            self.D[r[i]['player']+"_y"].append(str(G_y))
            points = cv2.perspectiveTransform(np.float32([[[G_x, G_y]]]), m)[0][0]
            r[i]['real_2D'] = points
            self.D[r[i]['player']+"_xm"].append(str(points[0]))
            self.D[r[i]['player']+"_ym"].append(str(points[1]))
        ############ To check if we have miss detection #############
        for key in self.D.keys():  
            max_len = max(max_len, len(self.D[key])) # maximum length among all of lists
        for key in self.D.keys():
            if len(self.D[key]) != max_len: self.D[key].append("-")  # filling of missing values
        return r

    def FillOccPos(self, D_composed):
        for key in D_composed.keys():
            for i in range(len(D_composed[key])):
                self.D[key][i] = D_composed[key][i]

    def GetGamePositions(self):
        return self.D
    