import os
import torch
import logging
import cv2
import numpy as np
import math
import PIL

from insightface.app import FaceAnalysis

from .flow import Flow

def draw_kps(image_pil, kps, color_list=[(255,0,0), (0,255,0), (0,0,255), (255,255,0), (255,0,255)]):

    stickwidth = 4
    limbSeq = np.array([[0, 2], [1, 2], [3, 2], [4, 2]])
    kps = np.array(kps)

    w, h = image_pil.size
    out_img = np.zeros([h, w, 3])

    for i in range(len(limbSeq)):
        index = limbSeq[i]
        color = color_list[index[0]]

        x = kps[index][:, 0]
        y = kps[index][:, 1]
        length = ((x[0] - x[1]) ** 2 + (y[0] - y[1]) ** 2) ** 0.5
        angle = math.degrees(math.atan2(y[0] - y[1], x[0] - x[1]))
        polygon = cv2.ellipse2Poly((int(np.mean(x)), int(np.mean(y))), (int(length / 2), stickwidth), int(angle), 0, 360, 1)
        out_img = cv2.fillConvexPoly(out_img.copy(), polygon, color)
    out_img = (out_img * 0.6).astype(np.uint8)

    for idx_kp, kp in enumerate(kps):
        color = color_list[idx_kp]
        x, y = kp
        out_img = cv2.circle(out_img.copy(), (int(x), int(y)), 10, color, -1)

    out_img_pil = PIL.Image.fromarray(out_img.astype(np.uint8))
    return out_img_pil

class InsightFaceAnalysis(Flow):
    def __init__(self, face_embeddings, face_controlnet_image):
        self.face_embeddings = face_embeddings
        self.face_controlnet_image = face_controlnet_image

class DoInsightFaceAnalysis(Flow):
    """
    Video -> DoInsightFaceAnalisys -> InsightFaceAnalysis
    """

    def __init__(self, model_path: str = "./", cache=None):

        self.app = FaceAnalysis(
                name='antelopev2',
                root=model_path,
                providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

        self.app.prepare(ctx_id=0, det_size=(640, 640))

        self.cache = cache

    def apply(self, video):

        logging.debug("DoInsightFaceAnalisys apply %s", video)

        if self.cache is not None:
            if os.path.isfile(self.cache):
                self.face_embeddings, self.face_controlnet_image = torch.load(self.cache)
            else:
                self.face_embeddings, self.face_controlnet_image = self.analyze(video)
                torch.save([self.face_embeddings, self.face_controlnet_image], self.cache)
        else:
            self.face_embeddings, self.face_controlnet_image = self.analyze(video)

        return InsightFaceAnalysis(
                face_embeddings=self.face_embeddings,
                face_controlnet_image=self.face_controlnet_image,
                )

    def analyze(self, video):
        face_embeddings = []
        face_controlnet_image = []

        for frame in video.hwc()[0]:
            face_image = PIL.Image.fromarray(np.array(frame.detach().cpu()))
            face_info = self.app.get(cv2.cvtColor(np.array(face_image), cv2.COLOR_RGB2BGR))
            face_info = sorted(face_info, key=lambda x:(x['bbox'][2]-x['bbox'][0])*x['bbox'][3]-x['bbox'][1])[-1]  # only use the maximum face
            face_emb = face_info['embedding']
            face_kps = draw_kps(face_image, face_info['kps'])

            face_embeddings.append(face_emb)
            face_controlnet_image.append(torch.tensor(np.array(face_kps)))

        face_controlnet_image = torch.stack(face_controlnet_image)
        face_embeddings = torch.tensor(face_embeddings)

        return face_embeddings, face_controlnet_image


