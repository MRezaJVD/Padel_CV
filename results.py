import json
import numpy as np

######################### Global Encoder #########################
class GlobalEncoder(json.JSONEncoder):
    """
    This class is useful to make all dataframe types
    serializable to write as json style
    """
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, torch.Tensor):
            return obj.tolist()
        elif isinstance(obj, detectron2.structures.boxes.Boxes):
            return obj.tensor.cpu().tolist()
        # else:
            # return super().default(obj)
        return json.JSONEncoder.default(self, obj)


############################# Write results as json ###################################
def print_and_save_results(r, json_out_file):
    data = []  # data for JSON output (list of dictionaries)
    L = len(r)
    print(" Detected persons: ", L)
    for i in range(L):
        #print("  Person ", i)
        bbox = [int(r[i]["bbox"][k]) for k in range(0, 4)]
        keypoints = r[i]["keypoints"]
        pl_label = r[i]["player"]
        Perspective2D = r[i]["real_2D"]
        rows = keypoints.shape[0]
        joints = []
        for row in range(rows):
            x = int(keypoints[row, 0])
            y = int(keypoints[row, 1])
            p = int(keypoints[row, 2]*100)/100
            #print("   ",x,y,p)
            joints.append([x, y, p])
        d = dict()
        d["bbox"] = bbox
        d["keypoints"] = joints
        d["player"] = pl_label
        d["TopView2D"] = Perspective2D
        data.append(d)

    # dump to JSON file
    with open(json_out_file, 'w') as outfile:
        json.dump(data, outfile, indent=4, cls=GlobalEncoder)
