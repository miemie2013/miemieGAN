#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# @miemie2013

from .architectures.styleganv2ada_model import StyleGANv2ADAModel
from .architectures.styleganv3_model import StyleGANv3Model
from .architectures.starganv2_model import StarGANv2Model

from .generators.generator_styleganv2ada import StyleGANv2ADA_SynthesisNetwork
from .generators.generator_styleganv2ada import StyleGANv2ADA_MappingNetwork
from .generators.generator_styleganv2ada import StyleGANv2ADA_AugmentPipe
from .generators.generator_starganv2 import StarGANv2Mapping, StarGANv2Generator, StarGANv2Style

from .generators.generator_styleganv3 import StyleGANv3_SynthesisNetwork
from .generators.generator_styleganv3 import StyleGANv3_MappingNetwork

from .discriminators.discriminator_styleganv2ada import StyleGANv2ADA_Discriminator
from .discriminators.discriminator_styleganv3 import StyleGANv3_Discriminator
from .discriminators.discriminator_starganv2 import StarGANv2Discriminator

from .networks.inception_pytorch import Inception_v3

