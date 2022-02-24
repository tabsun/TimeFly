import cv2
import numpy as np
import os
import insightface
from tqdm import tqdm
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image
import thinplate as tps

def limit_image(image, size=(720,1280)):
    h, w = image.shape[:2]
    std_w, std_h = size
    full = np.zeros((std_h, std_w, 3), dtype=np.uint8)
    scale = min(std_h/h, std_w/w)
    inner_h, inner_w = int(h*scale), int(w*scale)
    off_x, off_y = (std_w-inner_w)//2, (std_h-inner_h)//2

    full[off_y:off_y+inner_h, off_x:off_x+inner_w, :] = cv2.resize(image, (inner_w, inner_h))
    return full

class Plane:
    def __init__(self):
        self.app = FaceAnalysis(allowed_modules=['detection', 'landmark_2d_106'])
        self.app.prepare(ctx_id=0, det_size=(640, 640))

    def detect_largest_face(self, image):
        faces = self.app.get(image)
        if(not faces): return None

        max_face_area = -float('inf')
        largest_face = None
        for face in faces:
            x1, y1, x2, y2 = face.bbox
            if((x2-x1)*(y2-y1) > max_face_area):
                max_face_area = (x2-x1)*(y2-y1)
                largest_face = face
        return largest_face

    def warp_image(self, image, src, dst):
        dshape = tuple(image.shape[:2])
        #t = cv2.getTickCount()
        param = tps.tps_theta_from_points(src, dst, reduced=True)
        #t = cv2.getTickCount() - t
        #print("Estimate : %gms" % (t*1000/cv2.getTickFrequency()))

        #t = cv2.getTickCount()
        grid = tps.tps_grid(param, dst, dshape)
        #t = cv2.getTickCount() - t
        #print("Grid : %gms" % (t*1000/cv2.getTickFrequency()))

        #t = cv2.getTickCount()
        mapx, mapy = tps.tps_grid_to_remap(grid, image.shape)
        #t = cv2.getTickCount() - t
        #print("Mapx Mapy : %gms" % (t*1000/cv2.getTickFrequency()))

        #t = cv2.getTickCount()
        temp = cv2.remap(image, mapx, mapy, cv2.INTER_CUBIC)
        #t = cv2.getTickCount() - t
        #print("Remap : %gms" % (t*1000/cv2.getTickFrequency()))
        return param, temp

    def merge_image(self, A, B, theta=0.5):
        image = A * (1. - theta) + B * theta
        image = np.round(image).astype(np.uint8)
        return image

    def fly(self, src_image, src, dst_image, dst, theta=0.5):
        #t = cv2.getTickCount()
        assert(src_image.shape == dst_image.shape)
        mid = (dst - src)*theta + src
        _, src_image = self.warp_image(src_image, src, mid)
        _, dst_image = self.warp_image(dst_image, dst, mid)
        image = self.merge_image(src_image, dst_image, theta=theta)
        #t = cv2.getTickCount() - t
        #print("Merge one image: %gms" % (t*1000/cv2.getTickFrequency()))
        return image

    def gen_sequence(self, A, B, std_size, t=3.):
        image_A = limit_image(A, size=std_size)
        image_B = limit_image(B, size=std_size)
        A_landmarks = self.detect_largest_face(image_A).landmark_2d_106
        B_landmarks = self.detect_largest_face(image_B).landmark_2d_106
        h, w, _ = image_A.shape
        A_landmarks[:, 0] /= w
        A_landmarks[:, 1] /= h
        B_landmarks[:, 0] /= w
        B_landmarks[:, 1] /= h

        N = int(t*25)
        seq = []
        for i in tqdm(range(N)):
            theta = i / N
            image = plane.fly(image_A, A_landmarks, image_B, B_landmarks, theta=theta)
            seq.append(image)
        return seq

if __name__ == '__main__':
    import shutil
    import subprocess

    plane = Plane()

    fnames = ['01.png', '02.jpg', '031.jpg','032.jpg','033.jpg','04.jpg','05.png','06.jpg','061.jpg', '07.jpg','08.jpg', '666.jpg','09.jpg',  '11.jpg']
    output_path = 'out.mp4'

    images = []
    for i in range(len(fnames)-1): 
        A = cv2.imread(os.path.join('images', fnames[i]))
        B = cv2.imread(os.path.join('images', fnames[i+1]))
        seq = plane.gen_sequence(A, B, std_size=(720, 1280), t=3.)
        if(i == 0):
            A = limit_image(A, size=(720, 1280))
            for _ in range(6):
                images.append(A)
        images = images + seq
        B = limit_image(B, size=(720, 1280))
        if(i == len(fnames)-2):
            for _ in range(20):
                images.append(B)
        else:
            for _ in range(3):
                images.append(B)
    
    if(os.path.exists('tmp')):
        shutil.rmtree('tmp')
    os.makedirs('tmp')
    for i, image in enumerate(images):
        cv2.imwrite(os.path.join('tmp', '%03d.png'%i), image)
    print("Get %d images" % len(images))

    cmd = 'ffmpeg -y -i tmp/%03d.png -vcodec libx264 -vf fps=25 -pix_fmt yuv420p ' + output_path
    subprocess.call(cmd, shell=True)

    shutil.rmtree('tmp')
