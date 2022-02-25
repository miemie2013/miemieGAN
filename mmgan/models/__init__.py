#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# @miemie2013

from .architectures.styleganv2ada_model import StyleGANv2ADAModel

from .generators.generator_styleganv2ada import StyleGANv2ADA_SynthesisNetwork
from .generators.generator_styleganv2ada import StyleGANv2ADA_MappingNetwork
from .generators.generator_styleganv2ada import StyleGANv2ADA_AugmentPipe

from .discriminators.discriminator_styleganv2ada import StyleGANv2ADA_Discriminator

