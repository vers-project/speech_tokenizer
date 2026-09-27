# Modified from SpeechTokenizer (https://github.com/ZhangXInFD/SpeechTokenizer), Apache-2.0. See README.md.
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 30 15:47:55 2023
@author: zhangxin
"""

from .modules.seanet import SEANetEncoder, SEANetDecoder
from .quantization  import ResidualVectorQuantizer
import torch.nn as nn
from einops import rearrange
import torch
import numpy as np

class SpeechTokenizer(nn.Module):
    def __init__(self, config):
        '''
        
        Parameters
        ----------
        config : json
            Model Config.

        '''
        super().__init__()
        self.encoder = SEANetEncoder(n_filters=config.get('n_filters'), 
                                     dimension=config.get('dimension'), 
                                     ratios=config.get('strides'),
                                     lstm=config.get('lstm_layers'),
                                     bidirectional=config.get('bidirectional'),
                                     dilation_base=config.get('dilation_base'),
                                     residual_kernel_size=config.get('residual_kernel_size'),
                                     n_residual_layers=config.get('n_residual_layers'),
                                     activation=config.get('activation'))
        self.sample_rate = config.get('sample_rate')
        self.n_q = config.get('n_q')
        self.downsample_rate = np.prod(config.get('strides'))
        self.quantizer = ResidualVectorQuantizer(dimension=config.get('dimension'), n_q=config.get('n_q'), bins=config.get('codebook_size'))
        self.decoder = SEANetDecoder(n_filters=config.get('n_filters'), 
                                     dimension=config.get('dimension'), 
                                     ratios=config.get('strides'),
                                     lstm=config.get('lstm_layers'),
                                     bidirectional=False,
                                     dilation_base=config.get('dilation_base'),
                                     residual_kernel_size=config.get('residual_kernel_size'),
                                     n_residual_layers=config.get('n_residual_layers'),
                                     activation=config.get('activation'))

        transform_dimensions = config.get("codes_transform_dimensions")
        transforms = []
        for dim in [*transform_dimensions, *([None]*(self.n_q-len(transform_dimensions)))]:
            if dim is None:
                transforms.append(nn.Identity())
            else:
                transforms.append(nn.Linear(config.get('dimension'), dim))
        self.transforms = nn.ModuleList(transforms)
    
    @classmethod
    def load_from_checkpoint(cls, 
                             config: dict, 
                             ckpt_path: str):
        '''

        Parameters
        ----------
        config_path : str
            Path of model configuration file.
        ckpt_path : str
            Path of model  checkpoint.

        Returns
        -------
        model : SpeechTokenizer
            SpeechTokenizer model.

        '''
        model = cls(config)
        params = torch.load(ckpt_path, map_location='cpu')
        model.load_state_dict(params, strict=False)
        return model
    
    
    def forward(self, 
                x: torch.Tensor, 
                n_q: int=None, 
                layers: list=[0]):
        '''
        
        Parameters
        ----------
        x : torch.tensor
            Input wavs. Shape: (batch, channels, timesteps).
        n_q : int, optional
            Number of quantizers in RVQ used to encode. The default is all layers.
        layers : list[int], optional
            Layers of RVQ should return quantized result. The default is the first layer.

        Returns
        -------
        o : torch.tensor
            Output wavs. Shape: (batch, channels, timesteps).
        commit_loss : torch.tensor
            Commitment loss from residual vector quantizers.
        feature : torch.tensor
            Output of RVQ's first layer. Shape: (batch, timesteps, dimension)

        '''
        n_q = n_q if n_q else self.n_q
        e = self.encoder(x)
        quantized, code_indices, commitment_loss, codes = self.quantizer(e, n_q=n_q, layers=layers)
        
        projected_codes = []
        for code, transform in zip(codes, self.transforms):
            code = rearrange(code, 'b d t -> b t d')
            projected_code = transform(code)
            projected_codes.append(projected_code)

        reconstructed = self.decoder(quantized)
        return reconstructed, commitment_loss, projected_codes
    
    def forward_feature(self, 
                        x: torch.Tensor, 
                        layers: list=None):
        '''

        Parameters
        ----------
        x : torch.tensor
            Input wavs. Shape should be (batch, channels, timesteps).
        layers : list[int], optional
            Layers of RVQ should return quantized result. The default is all layers.

        Returns
        -------
        quantized_list : list[torch.tensor]
            Quantized of required layers.

        '''
        e = self.encoder(x)
        layers = layers if layers else list(range(self.n_q))
        quantized, codes, commit_loss, quantized_list = self.quantizer(e, layers=layers)
        return quantized_list
    
    def encode(self, 
               x: torch.Tensor, 
               n_q: int=None, 
               st: int=None):
        '''

        Parameters
        ----------
        x : torch.tensor
            Input wavs. Shape: (batch, channels, timesteps).
        n_q : int, optional
            Number of quantizers in RVQ used to encode. The default is all layers.
        st : int, optional
            Start quantizer index in RVQ. The default is 0.

        Returns
        -------
        codes : torch.tensor
            Output indices for each quantizer. Shape: (n_q, batch, timesteps)

        '''
        e = self.encoder(x)
        if st is None:
            st = 0
        n_q = n_q if n_q else self.n_q
        codes = self.quantizer.encode(e, n_q=n_q, st=st)
        return codes
    
    def decode(self, 
               codes: torch.Tensor, 
               st: int=0):
        '''

        Parameters
        ----------
        codes : torch.tensor
            Indices for each quantizer. Shape: (n_q, batch, timesteps).
        st : int, optional
            Start quantizer index in RVQ. The default is 0.

        Returns
        -------
        o : torch.tensor
            Reconstruct wavs from codes. Shape: (batch, channels, timesteps)

        '''
        quantized = self.quantizer.decode(codes, st=st)
        o = self.decoder(quantized)
        return o