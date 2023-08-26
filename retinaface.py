'''
This py file is modified from test_widerface to serve the purpose of the generation of Phantom Sponges.
'''
from __future__ import print_function
import os
import torch
import torch.backends.cudnn as cudnn
import numpy as np

from attack.Retinaface.data import cfg_re50
from attack.Retinaface.layers.functions.prior_box import PriorBox
from attack.Retinaface.utils.nms.py_cpu_nms import py_cpu_nms
from attack.Retinaface.models.retinaface import RetinaFace
from attack.Retinaface.utils.box_utils import decode, decode_landm
from attack.Retinaface.utils.timer import Timer


def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    print('Missing keys:{}'.format(len(missing_keys)))
    print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True


def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def load_model(model, pretrained_path, load_to_cpu):
    print('Loading pretrained model from {}'.format(pretrained_path))
    if load_to_cpu:
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
    else:
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model


class RetinaFaceRes50:
    def __init__(self):
        self.anchor_num = 896 # copy from blazeface, it can be any number to fix the detections output size
        self.confidence_threshold = 0.05 # change it to maximize the performance of generation
        self.nms_threshold = 0.01 # change it to maximize the performance of generation
        self.cfg = cfg_re50
        # net and model
        net = RetinaFace(cfg=self.cfg, phase = 'test')
        self.model = load_model(net, 'attack/Retinaface/weights/Resnet50_Final.pth', not torch.cuda.is_available())
        self.model.eval()
        print('Finished loading model!')
        print(self.model)
        cudnn.benchmark = True

    def to_device(self, device):
        self.device = device
        self.model.to(self.device)

    def predict_on_batch(self, img_batch):
        '''
        This follows the output format from BlazeFace to adapt to UAP generation.
        Returns:
            A list containing a tensor of face detections for each image in 
            the batch. If no faces are found for an image, returns a tensor
            of shape (anchor_num, 17).
        Each face detection is a PyTorch tensor consisting of 17 numbers:
            - x, y, w, h
            - x,y-coordinates for the 6 keypoints, all 0 here
            - confidence score
        '''
        batch_detections = []
        # testing dataset
        _t = {'forward_pass': Timer(), 'misc': Timer()}
    
        # testing begin
        for i, img in enumerate(img_batch):
            img *= 255 # input image is 0-1 but model input must be 0-255
            face_tensors = torch.ones([self.anchor_num, 17], dtype=torch.float) * -1
    
            # testing scale
            _, im_height, im_width = img.shape
            scale = torch.Tensor([img.shape[2], img.shape[1], img.shape[2], img.shape[1]])
            rgb_means = torch.tensor([104, 117, 123], dtype=img.dtype)
            img -= rgb_means.view(3, 1, 1)
            img = img.unsqueeze(0)
            img = img.to(self.device)
            scale = scale.to(self.device)
    
            _t['forward_pass'].tic()
            loc, conf, landms = self.model(img)  # forward pass
            _t['forward_pass'].toc()
            _t['misc'].tic()
            priorbox = PriorBox(self.cfg, image_size=(im_height, im_width))
            priors = priorbox.forward()
            priors = priors.to(self.device)
            prior_data = priors.data
            boxes = decode(loc.data.squeeze(0), prior_data, self.cfg['variance'])
            boxes = boxes * scale
            boxes = boxes.cpu().numpy()
            scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
            landms = decode_landm(landms.data.squeeze(0), prior_data, self.cfg['variance'])
            scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                                   img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                                   img.shape[3], img.shape[2]])
            scale1 = scale1.to(self.device)
            landms = landms * scale1
            landms = landms.cpu().numpy()
    
            # ignore low scores
            inds = np.where(scores > self.confidence_threshold)[0]
            boxes = boxes[inds]
            landms = landms[inds]
            scores = scores[inds]
    
            # keep top-K before NMS
            order = scores.argsort()[::-1]
            boxes = boxes[order]
            landms = landms[order]
            scores = scores[order]
    
            # do NMS
            dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
            keep = py_cpu_nms(dets, self.nms_threshold)
            dets = dets[keep, :]
            landms = landms[keep]
    
            dets = np.concatenate((dets, landms), axis=1)
            _t['misc'].toc()
    
            # based on uap_phantom_sponge.py, bboxes_area function will take in [x, y, w, h] 
            bboxs = dets
            for idx, box in enumerate(bboxs):
                x = int(box[0])
                y = int(box[1])
                w = int(box[2]) - int(box[0])
                h = int(box[3]) - int(box[1])
                face_tensor = [x, y, w, h] + ([0 for i in range(12)]) + [box[4]] # 17 values
                face_tensors[idx] = torch.tensor(face_tensor)
            batch_detections.append(face_tensors)
            print('im_detect: {:d}/{:d} forward_pass_time: {:.4f}s misc: {:.4f}s'.format(i + 1, len(img_batch), _t['forward_pass'].average_time, _t['misc'].average_time))
        
        return batch_detections
    
