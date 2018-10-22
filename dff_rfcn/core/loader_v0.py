# --------------------------------------------------------
# Deep Feature Flow
# Copyright (c) 2017 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Modified by Xizhou Zhu, Yuwen Xiong
# --------------------------------------------------------
# Based on:
# MX-RCNN
# Copyright (c) 2016 by Contributors
# Licence under The Apache 2.0 License
# https://github.com/ijkguo/mx-rcnn/
# --------------------------------------------------------

import numpy as np
import mxnet as mx
from mxnet.executor_manager import _split_input_slice

from config.config import config
from utils.image import tensor_vstack
from rpn.rpn import get_rpn_testbatch, get_rpn_pair_batch, assign_pyramid_anchor
from rcnn import get_rcnn_testbatch


class TestLoader(mx.io.DataIter):
    def __init__(self, roidb, config, batch_size=1, shuffle=False,
                 has_rpn=False):
        super(TestLoader, self).__init__()

        # save parameters as properties
        self.cfg = config
        self.roidb = roidb
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.has_rpn = has_rpn

        # infer properties from roidb
        self.size = np.sum([x['frame_seg_len'] for x in self.roidb])
        self.index = np.arange(self.size)

        # decide data and label names (only for training)
        self.data_name = ['data', 'im_info', 'data_key', 'feat_key']
        self.label_name = None

        #
        self.cur_roidb_index = 0
        self.cur_frameid = 0
        self.data_key = None
        self.key_frameid = 0
        self.cur_seg_len = 0
        self.key_frame_flag = -1

        # status variable for synchronization between get_data and get_label
        self.cur = 0
        self.data = None
        self.label = []
        self.im_info = None

        # get first batch to fill in provide_data and provide_label
        self.reset()
        self.get_batch()

    @property
    def provide_data(self):
        return [[(k, v.shape) for k, v in zip(self.data_name, idata)] for idata in self.data]

    @property
    def provide_label(self):
        return [None for _ in range(len(self.data))]

    @property
    def provide_data_single(self):
        return [(k, v.shape) for k, v in zip(self.data_name, self.data[0])]

    @property
    def provide_label_single(self):
        return None

    def reset(self):
        self.cur = 0
        if self.shuffle:
            np.random.shuffle(self.index)

    def iter_next(self):
        return self.cur < self.size

    def next(self):
        if self.iter_next():
            self.get_batch()
            self.cur += self.batch_size
            self.cur_frameid += 1
            if self.cur_frameid == self.cur_seg_len:
                self.cur_roidb_index += 1
                self.cur_frameid = 0
                self.key_frameid = 0
            elif self.cur_frameid - self.key_frameid == self.cfg.TEST.KEY_FRAME_INTERVAL:
                self.key_frameid = self.cur_frameid
            return self.im_info, self.key_frame_flag, mx.io.DataBatch(data=self.data, label=self.label,
                                   pad=self.getpad(), index=self.getindex(),
                                   provide_data=self.provide_data, provide_label=self.provide_label)
        else:
            raise StopIteration

    def getindex(self):
        return self.cur / self.batch_size

    def getpad(self):
        if self.cur + self.batch_size > self.size:
            return self.cur + self.batch_size - self.size
        else:
            return 0

    def get_batch(self):
        cur_roidb = self.roidb[self.cur_roidb_index].copy()
        cur_roidb['image'] = cur_roidb['pattern'] % self.cur_frameid
        self.cur_seg_len = cur_roidb['frame_seg_len']
        data, label, im_info = get_rpn_testbatch([cur_roidb], self.cfg)
        if self.key_frameid == self.cur_frameid: # key frame
            self.data_key = data[0]['data'].copy()
            if self.key_frameid == 0:
                self.key_frame_flag = 0
            else:
                self.key_frame_flag = 1
        else:
            self.key_frame_flag = 2
        extend_data = [{'data': data[0]['data'],
                        'im_info': data[0]['im_info'],
                        'data_key': self.data_key,
                        'feat_key': np.zeros((1,self.cfg.network.DFF_FEAT_DIM,1,1))}]
        self.data = [[mx.nd.array(extend_data[i][name]) for name in self.data_name] for i in xrange(len(data))]
        self.im_info = im_info


class PyramidAnchorIterator(mx.io.DataIter):

    def __init__(self, feat_sym, roidb, cfg, batch_size=1, shuffle=False, ctx=None, work_load_list=None,
                 feat_strides=(4, 8, 16, 16, 32), anchor_scales=(8, ), anchor_ratios=(0.5, 1, 2), allowed_border=0,
                 aspect_grouping=False):
        """
        This Iter will provide roi data to Fast R-CNN network
        :param feat_sym: to infer shape of assign_output
        :param roidb: must be preprocessed
        :param batch_size: must divide BATCH_SIZE(128)
        :param shuffle: bool
        :param ctx: list of contexts
        :param work_load_list: list of work load
        :param aspect_grouping: group images with similar aspects
        :param normalize_target: normalize rpn target
        :param bbox_mean: anchor target mean
        :param bbox_std: anchor target std
        :return: AnchorLoader
        """
        super(PyramidAnchorIterator, self).__init__()

        # save parameters as properties
        self.feat_sym = feat_sym
        self.roidb = roidb
        self.cfg = cfg
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.ctx = ctx
        if self.ctx is None:
            self.ctx = [mx.cpu()]
        self.work_load_list = work_load_list
        self.feat_strides = feat_strides
        self.anchor_scales = anchor_scales
        self.anchor_ratios = anchor_ratios
        self.allowed_border = allowed_border
        self.aspect_grouping = aspect_grouping

        # infer properties from roidb
        self.size = len(roidb)
        self.index = np.arange(self.size)

        # decide data and label names
        if config.TRAIN.END2END:
            self.data_name = ['data', 'data_ref', 'eq_flag', 'im_info', 'gt_boxes']
        else:
            self.data_name = ['data']
        # fpn
        self.feat_pyramid_level = np.log2(self.cfg.network.RPN_FEAT_STRIDE).astype(int)
        self.feat_pyramid_level[1] = 5
        # self.feat_pyramid_level[3] = 5
        # self.feat_pyramid_level[4] = 6
        # end

        self.label_name = ['label', 'bbox_target', 'bbox_weight']

        # status variable for synchronization between get_data and get_label
        self.cur = 0
        self.batch = None
        self.data = None
        self.label = None

        # get first batch to fill in provide_data and provide_label
        self.reset()
        self.get_batch_parallel()

    @property
    def provide_data(self):
        return [[(k, v.shape) for k, v in zip(self.data_name, self.data[i])] for i in xrange(len(self.data))]

    @property
    def provide_label(self):
        return [[(k, v.shape) for k, v in zip(self.label_name, self.label[i])] for i in xrange(len(self.data))]

    @property
    def provide_data_single(self):
        return [(k, v.shape) for k, v in zip(self.data_name, self.data[0])]

    @property
    def provide_label_single(self):
        return [(k, v.shape) for k, v in zip(self.label_name, self.label[0])]

    def reset(self):
        self.cur = 0
        if self.shuffle:
            if self.aspect_grouping:
                widths = np.array([r['width'] for r in self.roidb])
                heights = np.array([r['height'] for r in self.roidb])
                horz = (widths >= heights)
                vert = np.logical_not(horz)
                horz_inds = np.where(horz)[0]
                vert_inds = np.where(vert)[0]
                inds = np.hstack((np.random.permutation(horz_inds), np.random.permutation(vert_inds)))
                extra = inds.shape[0] % self.batch_size
                inds_ = np.reshape(inds[:-extra], (-1, self.batch_size))
                row_perm = np.random.permutation(np.arange(inds_.shape[0]))
                inds[:-extra] = np.reshape(inds_[row_perm, :], (-1,))
                self.index = inds
            else:
                np.random.shuffle(self.index)

    def iter_next(self):
        return self.cur + self.batch_size <= self.size

    def next(self):
        if self.iter_next():
            self.get_batch_parallel()
            self.cur += self.batch_size
            return mx.io.DataBatch(data=self.data, label=self.label,
                                   pad=self.getpad(), index=self.getindex(),
                                   provide_data=self.provide_data, provide_label=self.provide_label)
        else:
            raise StopIteration

    def getindex(self):
        return self.cur / self.batch_size

    def getpad(self):
        if self.cur + self.batch_size > self.size:
            return self.cur + self.batch_size - self.size
        else:
            return 0

    def infer_shape(self, max_data_shape=None, max_label_shape=None):
        """ Return maximum data and label shape for single gpu """
        if max_data_shape is None:
            max_data_shape = []
        if max_label_shape is None:
            max_label_shape = []
        max_shapes = dict(max_data_shape + max_label_shape)
        input_batch_size = max_shapes['data'][0]
        im_info = [[max_shapes['data'][2], max_shapes['data'][3], 1.0]]

        # fpn
        # y[1][0]?
        feat_shape = [y[1] for y in [x.infer_shape(**max_shapes) for x in self.feat_sym]]
        label = assign_pyramid_anchor(feat_shape, np.zeros((0, 5)), im_info,
                                      self.cfg, self.feat_strides, self.anchor_scales,self.anchor_ratios, self.allowed_border)
        # end

        label = [label[k] for k in self.label_name]
        label_shape = [(k, tuple([input_batch_size] + list(v.shape[1:]))) for k, v in zip(self.label_name, label)]
        return max_data_shape, label_shape

    def get_batch_parallel(self):
        cur_from = self.cur
        cur_to = min(cur_from + self.batch_size, self.size)
        roidb = [self.roidb[self.index[i]] for i in range(cur_from, cur_to)]
        # decide multi device slice
        work_load_list = self.work_load_list
        ctx = self.ctx
        if work_load_list is None:
            work_load_list = [1] * len(ctx)
        assert isinstance(work_load_list, list) and len(work_load_list) == len(ctx), \
            "Invalid settings for work load. "
        slices = _split_input_slice(self.batch_size, work_load_list)

        # fpn
        rst = []
        for idx, islice in enumerate(slices):
            iroidb = [roidb[i] for i in range(islice.start, islice.stop)]
            rst.append(par_assign_anchor_wrapper(self.cfg, iroidb, self.feat_sym, self.feat_strides, self.anchor_scales,
                                                 self.anchor_ratios, self.allowed_border))
        # end

        all_data = [_['data'] for _ in rst]
        all_label = [_['label'] for _ in rst]
        self.data = [[mx.nd.array(data[key]) for key in self.data_name] for data in all_data]
        self.label = [[mx.nd.array(label[key]) for key in self.label_name] for label in all_label]

# fpn
def par_assign_anchor_wrapper(cfg, iroidb, feat_sym, feat_strides,
                              anchor_scales, anchor_ratios, allowed_border):
    # get testing data for multigpu
    data, rpn_label = get_rpn_pair_batch(iroidb, cfg)
    data_shape = {k: v.shape for k, v in data.items()}
    del data_shape['im_info']

    # add gt_boxes to data for e2e
    data['gt_boxes'] = rpn_label['gt_boxes'][np.newaxis, :, :]

    # fpn
    feat_shape = []
    for x in feat_sym:
        _, out_shape, _ = x.infer_shape(**data_shape)
        feat_shape.append(out_shape)
    # feat_shape = [y[1] for y in [x.infer_shape(**data_shape) for x in feat_sym]]
    # end

    label = assign_pyramid_anchor(feat_shape, rpn_label['gt_boxes'],
                                  data['im_info'], cfg, feat_strides,
                                  anchor_scales, anchor_ratios, allowed_border)
    return {'data': data, 'label': label}