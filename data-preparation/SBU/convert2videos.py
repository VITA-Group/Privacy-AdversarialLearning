import numpy as np
import cv2
import os
from skimage import io

home_dir = "/home/wuzhenyu_sjtu/Privacy-AdversarialLearning/data-collection/SBU"


def get_SBU_actor_action_setting(path):
    return path.split('/')[-3], path.split('/')[-2], path.split('/')[-1],


def write_video(X, Y_action, Y_actor, Y_setting, is_depthmap=False):
    WIDTH, HEIGHT = 640, 480
    for i in range(len(X)):
        print(os.getcwd())
        fourcc = cv2.VideoWriter_fourcc(*'XVID') # Be sure to use lower case
        output = os.path.join(home_dir, "videos/{}_{}_{}.avi".format(Y_actor[i], Y_action[i], Y_setting[i]))
        if is_depthmap:
            out = cv2.VideoWriter(output, fourcc, 10.0, (WIDTH, HEIGHT), False)
        else:
            out = cv2.VideoWriter(output, fourcc, 10.0, (WIDTH, HEIGHT), True)
        vid = X[i]
        vid = vid.astype('uint8')
        print(vid.shape)
        print(output)
        for i in range(vid.shape[0]):
            frame = vid[i]
            if is_depthmap:
                frame = frame.reshape(HEIGHT, WIDTH, 1)
            else:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                frame = frame.reshape(HEIGHT, WIDTH, 3)
            #print(frame)
            out.write(frame)
        out.release()
        cv2.destroyAllWindows()


path = os.path.join(home_dir, "frames")
path = os.path.normpath(path)
X = []
Y_action = []
Y_actor = []
Y_setting = []
print(path)
for root, dirs, files in os.walk(path):
    depth = root[len(path):].count(os.path.sep)
    if depth == 3:
        print(root)
        files = [file for file in files if file.startswith("depth")]
        files.sort()
        actor, action, setting = get_SBU_actor_action_setting(root)
        framelst = []
        for file in files:
            filename = os.path.join(root, file)
            frame = io.imread(filename)
            framelst.append(frame)
        X.append(np.asarray(framelst))
        Y_action.append(action)
        Y_actor.append(actor)
        Y_setting.append(setting)

write_video(X, Y_action, Y_actor, Y_setting, is_depthmap=True)