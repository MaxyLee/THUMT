# coding=utf-8
# Copyright 2017-2020 The THUMT Authors

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import thumt.models.transformer
import thumt.models.prefix
import thumt.models.visual_prefix


def get_model(name):
    name = name.lower()

    if name == "transformer":
        return thumt.models.transformer.Transformer
    elif name == "prefix_transformer":
        return thumt.models.prefix.PrefixTransformer
    elif name == "visual_prefix_transformer":
        return thumt.models.visual_prefix.VisualPrefixTransformer
    else:
        raise LookupError("Unknown model %s" % name)
