# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import

from libs.configs import cfgs


class_names = [
        'unworking_chimney', 'working_chimney', 'unworking_condensing_tower', 'working_condensing_tower']

classes_originID = {
    'unworking_chimney': 1,
    'working_chimney': 2,
    'unworking_condensing_tower': 3,
    'working_condensing_tower': 4
    }


def get_coco_label_dict():
    originID_classes = {item: key for key, item in classes_originID.items()}
    NAME_LABEL_MAP = dict(zip(class_names, range(len(class_names))))
    return NAME_LABEL_MAP

if cfgs.DATASET_NAME == 'ship':
    NAME_LABEL_MAP = {
        'back_ground': 0,
        'ship': 1
    }
elif cfgs.DATASET_NAME == 'aeroplane':
    NAME_LABEL_MAP = {
        'back_ground': 0,
        'aeroplane': 1
    }
elif cfgs.DATASET_NAME == 'WIDER':
    NAME_LABEL_MAP = {
        'back_ground': 0,
        'face': 1
    }
elif cfgs.DATASET_NAME == 'icdar':
    NAME_LABEL_MAP = {
        'back_ground': 0,
        'text': 1
    }
elif cfgs.DATASET_NAME.startswith('DOTA'):
    NAME_LABEL_MAP = {
        'back_ground': 0,
        'roundabout': 1,
        'tennis-court': 2,
        'swimming-pool': 3,
        'storage-tank': 4,
        'soccer-ball-field': 5,
        'small-vehicle': 6,
        'ship': 7,
        'plane': 8,
        'large-vehicle': 9,
        'helicopter': 10,
        'harbor': 11,
        'ground-track-field': 12,
        'bridge': 13,
        'basketball-court': 14,
        'baseball-diamond': 15
    }
elif cfgs.DATASET_NAME == 'coco':
    NAME_LABEL_MAP = get_coco_label_dict()
elif cfgs.DATASET_NAME == 'pascal':
    NAME_LABEL_MAP = {
        'back_ground': 0,
        'unworking_chimney': 1,
        'working_chimney': 2,
        'unworking_condensing_tower': 3,
        'working_condensing_tower': 4,
    }
else:
    assert 'please set label dict!'


def get_label_name_map():
    reverse_dict = {}
    for name, label in NAME_LABEL_MAP.items():
        reverse_dict[label] = name
    return reverse_dict


LABEl_NAME_MAP = get_label_name_map()
