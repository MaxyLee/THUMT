# coding=utf-8
# Copyright 2017-2020 The THUMT Authors

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import thumt.models.transformer
import thumt.models.prefix
import thumt.models.visual_prefix
import thumt.models.visual_prefix_v2
import thumt.models.visual_prefix_v3
import thumt.models.visual_prefix_v4


def get_model(name):
    name = name.lower()

    if name == "transformer":
        return thumt.models.transformer.Transformer
    elif name == "prefix_transformer":
        return thumt.models.prefix.PrefixTransformer
    elif name == "visual_prefix_transformer":
        return thumt.models.visual_prefix.VisualPrefixTransformer
    elif name == "visual_prefix_transformer_v2":
        return thumt.models.visual_prefix_v2.VisualPrefixTransformer
    elif name == "visual_prefix_transformer_v3":
        return thumt.models.visual_prefix_v3.VisualPrefixTransformer
    elif name == "visual_prefix_transformer_v4":
        return thumt.models.visual_prefix_v4.VisualPrefixTransformer
    else:
        raise LookupError("Unknown model %s" % name)
