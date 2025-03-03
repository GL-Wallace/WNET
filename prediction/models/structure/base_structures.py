# Copyright (c) Carizon. All rights reserved.

from collections import OrderedDict, namedtuple
from typing import Any, Callable, Dict, Optional, Sequence

from torch import nn


def _as_list(obj: Any) -> Sequence:
    """Convert the argument to a list if it is not already."""

    if isinstance(obj, (list, tuple)):
        return obj
    elif isinstance(obj, set):
        return list(obj)
    else:
        return [obj]


class BasicTrajPredStructure(nn.Module):
    """The base class of the anchor-based model structure."""

    def __init__(
        self,
        backbone: Callable,
        necks: Optional[Dict] = None,
        heads: Optional[Dict] = None,
        post_process: Optional[Callable] = None,
        losses: Optional[Dict] = None,
        is_int_infer_model: bool = False,
    ):
        """Initialize method.

        Args:
            backbone: a dict for building backbone or callable instance.
            necks: the OrderedDict to build necks. Each element is a dict
                for building neck.
            heads: the OrderedDict to build heads. Each element is a dict
                for building head.
            post_process: the post-processing model.
            losses: the losses.
            is_int_infer_model: whether the model is for int inference.
        """
        super(BasicTrajPredStructure, self).__init__()
        self.is_int_infer_model = is_int_infer_model

        # Model structure.
        self.backbone = backbone
        self.neck_name_list = []
        if necks is not None:
            for neck_name, neck in necks.items():
                if neck is None:
                    continue
                setattr(self, neck_name, neck)
                self.neck_name_list.append(neck_name)
        # self.head_name_list = []
        # if heads is not None:
        #     for head_name, head in heads.items():
        #         if head is None:
        #             continue
        #         setattr(self, head_name, head)
        #         self.head_name_list.append(head_name)
        # self.post_process = post_process

        # Loss.
        self.losses = None
        if losses is not None:
            self.losses = nn.ModuleList(_as_list(losses))

    def build_custom_structure(self):
        """Customize structure function, should be defined in sub-classes."""
        pass

    def custom_data_preprocess(self, data: Dict):  # noqa: D401
        """Data preprocess function, should be defined in sub-classes."""

        raise NotImplementedError

    def custom_head_out_postprocess(self, head_outs: Dict, head_cls_name: str):
        """Post process after head out, should be defined in sub-classes."""
        return head_outs

    @staticmethod
    def _build(obj, builder):
        if obj is None:
            return None
        return builder(obj) if isinstance(obj, dict) else obj

    def forward(self, data):
        """Forward.

        Args:
            data: the original data. We do not care its format. The user
                should perform all the data pre-processing operation
                in `custom_data_preprocess` by himself.
        """
        model_result = OrderedDict()
        data, backbone_data, gt_data = self.custom_data_preprocess(data)
        feats = self.backbone(backbone_data)
        data["feats"] = feats
        neck_feats = {}
        for neck_name in self.neck_name_list:
            cur_neck = getattr(self, neck_name)
            neck_feats.update(cur_neck(data))
        if not len(neck_feats):
            neck_feats = feats

        all_head_finish_pp = True
        all_head_finish_loss = True
        if self.is_int_infer_model:
            for head_name in self.head_name_list:
                cur_head = getattr(self, head_name)
                cur_result = cur_head(neck_feats)

                if cur_head.post_process is not None:
                    cur_result.update(cur_head.post_process(cur_result, data))
                else:
                    all_head_finish_pp = False

                cur_result = {
                    head_name + "_" + key: value
                    for key, value in cur_result.items()
                }
                model_result.update(cur_result)

            # If all heads have their own post processing methods, the common
            # post processing will be skipped.
            if not all_head_finish_pp and self.post_process is not None:
                model_result.update(self.post_process(model_result, data))

            # For int infer, output named tuple.
            custom_name = self.__class__.__name__ + "Output"
            CustomOutput = namedtuple(custom_name, model_result.keys())
            model_result = CustomOutput(**model_result)
        else:
            model_result.update(data)
            for head_name in self.head_name_list:
                head_outs = OrderedDict()
                cur_head = getattr(self, head_name)
                out = cur_head(neck_feats)
                head_outs.update(out)
                head_outs.update(gt_data)
                head_cls_name = cur_head.__class__.__name__
                head_outs = self.custom_head_out_postprocess(
                    head_outs, head_cls_name
                )
                if self.training:
                    if cur_head.loss is not None:
                        head_outs.update(cur_head.loss(head_outs))
                    else:
                        all_head_finish_loss = False

                if cur_head.post_process is not None:
                    head_outs.update(cur_head.post_process(head_outs, data))
                else:
                    all_head_finish_pp = False

                head_results = {
                    head_name + "_" + key: value
                    for key, value in head_outs.items()
                }
                model_result.update(head_results)

            # If all heads have their own loss functions, the common loss will
            # be skipped. The user must ensure all heads can generate loss.
            if not all_head_finish_loss:
                if self.losses is not None:
                    for loss in self.losses:
                        model_result.update(loss(model_result))
                else:
                    raise NotImplementedError(
                        "Loss is not defined for all heads, please check."
                    )

            # If all heads have their own post processing methods, the common
            # post processing will be skipped.
            if not all_head_finish_pp and self.post_process is not None:
                model_result.update(self.post_process(model_result, data))

        return model_result
