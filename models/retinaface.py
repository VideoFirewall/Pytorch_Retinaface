import numpy as np
import torch
import torch.nn as nn
import torchvision.models.detection.backbone_utils as backbone_utils
import torchvision.models._utils as _utils
import torch.nn.functional as F
from collections import OrderedDict

from .net import MobileNetV1 as MobileNetV1
from .net import FPN as FPN
from .net import SSH as SSH

# for PhantomSponges use
from attack.Retinaface.data import cfg_re50
from attack.Retinaface.layers.functions.prior_box import PriorBox
from attack.Retinaface.utils.nms.py_cpu_nms import py_cpu_nms
from attack.Retinaface.utils.box_utils import decode, decode_landm
from attack.Retinaface.utils.timer import Timer


class ClassHead(nn.Module):
    def __init__(self,inchannels=512,num_anchors=3):
        super(ClassHead,self).__init__()
        self.num_anchors = num_anchors
        self.conv1x1 = nn.Conv2d(inchannels,self.num_anchors*2,kernel_size=(1,1),stride=1,padding=0)

    def forward(self,x):
        out = self.conv1x1(x)
        out = out.permute(0,2,3,1).contiguous()
        
        return out.view(out.shape[0], -1, 2)

class BboxHead(nn.Module):
    def __init__(self,inchannels=512,num_anchors=3):
        super(BboxHead,self).__init__()
        self.conv1x1 = nn.Conv2d(inchannels,num_anchors*4,kernel_size=(1,1),stride=1,padding=0)

    def forward(self,x):
        out = self.conv1x1(x)
        out = out.permute(0,2,3,1).contiguous()

        return out.view(out.shape[0], -1, 4)

class LandmarkHead(nn.Module):
    def __init__(self,inchannels=512,num_anchors=3):
        super(LandmarkHead,self).__init__()
        self.conv1x1 = nn.Conv2d(inchannels,num_anchors*10,kernel_size=(1,1),stride=1,padding=0)

    def forward(self,x):
        out = self.conv1x1(x)
        out = out.permute(0,2,3,1).contiguous()

        return out.view(out.shape[0], -1, 10)

class RetinaFace(nn.Module):
    def __init__(self, cfg = None, phase = 'train'):
        """
        :param cfg:  Network related settings.
        :param phase: train or test.
        """
        super(RetinaFace,self).__init__()

        # for predict_on_batch use
        self.cfg = cfg
        self.anchor_num = 896 # copy from blazeface, it can be any number to fix the detections output size
        self.confidence_threshold = 0.05 # change it to maximize the performance of generation
        self.nms_threshold = 0.01 # change it to maximize the performance of generation

        self.phase = phase
        backbone = None
        if cfg['name'] == 'mobilenet0.25':
            backbone = MobileNetV1()
            if cfg['pretrain']:
                checkpoint = torch.load("./weights/mobilenetV1X0.25_pretrain.tar", map_location=torch.device('cpu'))
                from collections import OrderedDict
                new_state_dict = OrderedDict()
                for k, v in checkpoint['state_dict'].items():
                    name = k[7:]  # remove module.
                    new_state_dict[name] = v
                # load params
                backbone.load_state_dict(new_state_dict)
        elif cfg['name'] == 'Resnet50':
            import torchvision.models as models
            backbone = models.resnet50(pretrained=cfg['pretrain'])

        self.body = _utils.IntermediateLayerGetter(backbone, cfg['return_layers'])
        in_channels_stage2 = cfg['in_channel']
        in_channels_list = [
            in_channels_stage2 * 2,
            in_channels_stage2 * 4,
            in_channels_stage2 * 8,
        ]
        out_channels = cfg['out_channel']
        self.fpn = FPN(in_channels_list,out_channels)
        self.ssh1 = SSH(out_channels, out_channels)
        self.ssh2 = SSH(out_channels, out_channels)
        self.ssh3 = SSH(out_channels, out_channels)

        self.ClassHead = self._make_class_head(fpn_num=3, inchannels=cfg['out_channel'])
        self.BboxHead = self._make_bbox_head(fpn_num=3, inchannels=cfg['out_channel'])
        self.LandmarkHead = self._make_landmark_head(fpn_num=3, inchannels=cfg['out_channel'])

    def _make_class_head(self,fpn_num=3,inchannels=64,anchor_num=2):
        classhead = nn.ModuleList()
        for i in range(fpn_num):
            classhead.append(ClassHead(inchannels,anchor_num))
        return classhead
    
    def _make_bbox_head(self,fpn_num=3,inchannels=64,anchor_num=2):
        bboxhead = nn.ModuleList()
        for i in range(fpn_num):
            bboxhead.append(BboxHead(inchannels,anchor_num))
        return bboxhead

    def _make_landmark_head(self,fpn_num=3,inchannels=64,anchor_num=2):
        landmarkhead = nn.ModuleList()
        for i in range(fpn_num):
            landmarkhead.append(LandmarkHead(inchannels,anchor_num))
        return landmarkhead

    def forward(self,inputs):
        out = self.body(inputs)

        # FPN
        fpn = self.fpn(out)

        # SSH
        feature1 = self.ssh1(fpn[0])
        feature2 = self.ssh2(fpn[1])
        feature3 = self.ssh3(fpn[2])
        features = [feature1, feature2, feature3]

        bbox_regressions = torch.cat([self.BboxHead[i](feature) for i, feature in enumerate(features)], dim=1)
        classifications = torch.cat([self.ClassHead[i](feature) for i, feature in enumerate(features)],dim=1)
        ldm_regressions = torch.cat([self.LandmarkHead[i](feature) for i, feature in enumerate(features)], dim=1)

        if self.phase == 'train':
            output = (bbox_regressions, classifications, ldm_regressions)
        else:
            output = (bbox_regressions, F.softmax(classifications, dim=-1), ldm_regressions)
        return output
    
    # for PhantomSponges use

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
            modified_img = img * 255 # input image is 0-1 but model input must be 0-255
            face_tensors = torch.zeros([self.anchor_num, 17], dtype=torch.float)
    
            # testing scale
            _, im_height, im_width = modified_img.shape
            scale = torch.Tensor([modified_img.shape[2], modified_img.shape[1], modified_img.shape[2], modified_img.shape[1]])
            rgb_means = torch.tensor([104, 117, 123], dtype=modified_img.dtype)
            modified_img -= rgb_means.view(3, 1, 1)
            modified_img = modified_img.unsqueeze(0)
    
            _t['forward_pass'].tic()
            loc, conf, landms = self.forward(modified_img)  # forward pass
            _t['forward_pass'].toc()
            _t['misc'].tic()
            priorbox = PriorBox(self.cfg, image_size=(im_height, im_width))
            priors = priorbox.forward()
            prior_data = priors.data
            boxes = decode(loc.data.squeeze(0), prior_data, self.cfg['variance'])
            boxes = boxes * scale
            boxes = boxes.cpu().numpy()
            scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
            landms = decode_landm(landms.data.squeeze(0), prior_data, self.cfg['variance'])
            scale1 = torch.Tensor([modified_img.shape[3], modified_img.shape[2], modified_img.shape[3], modified_img.shape[2],
                                   modified_img.shape[3], modified_img.shape[2], modified_img.shape[3], modified_img.shape[2],
                                   modified_img.shape[3], modified_img.shape[2]])
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
