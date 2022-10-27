# MIT License
#
# Copyright (c) 2020 CNRS
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from pyannote.audio.core.model import Model
from pyannote.audio.core.task import Task
from pyannote.audio.models.blocks.sincnet import SincNet
from pyannote.audio.utils.params import merge_dict
from pyannote.core.utils.generators import pairwise
from torchaudio.transforms import Spectrogram, InverseSpectrogram


class StereoSyncNet_IPD(Model):
    """PyanNet segmentation model

    SincNet > LSTM > Feed forward > Classifier

    Parameters
    ----------
    sample_rate : int, optional
        Audio sample rate. Defaults to 16kHz (16000).
    num_channels : int, optional
        Number of channels. Defaults to mono (1).
    sincnet : dict, optional
        Keyword arugments passed to the SincNet block.
        Defaults to {"stride": 1}.
    lstm : dict, optional
        Keyword arguments passed to the LSTM layer.
        Defaults to {"hidden_size": 128, "num_layers": 2, "bidirectional": True},
        i.e. two bidirectional layers with 128 units each.
        Set "monolithic" to False to split monolithic multi-layer LSTM into multiple mono-layer LSTMs.
        This may proove useful for probing LSTM internals.
    linear : dict, optional
        Keyword arugments used to initialize linear layers
        Defaults to {"hidden_size": 128, "num_layers": 2},
        i.e. two linear layers with 128 units each.
    """

    SINCNET_DEFAULTS = {"stride": 10}
    LSTM_DEFAULTS = {
        "hidden_size": 128,
        "num_layers": 2,
        "bidirectional": True,
        "monolithic": True,
        "dropout": 0.0,
    }
    LINEAR_DEFAULTS = {"hidden_size": 128, "num_layers": 2}
    NUM_CHANNELS_DEFAULTS = 2

    def __init__(
        self,
        sincnet: dict = None,
        lstm: dict = None,
        linear: dict = None,
        sample_rate: int = 16000,
        num_channels: int = 2,
        task: Optional[Task] = None,
    ):

        super().__init__(sample_rate=sample_rate, num_channels=1, task=task) ##infact Syncnet has just one channel


        self.fft_size = 128
        #self.phase_diff_mode = phase_diff_mode
        self.spectrogram = Spectrogram(n_fft=self.fft_size, power=None , hop_length= 10, center=False)  # Complex STFT
        self.num_freq_bins = self.fft_size // 2 + 1

        sincnet = merge_dict(self.SINCNET_DEFAULTS, sincnet)
        sincnet["sample_rate"] = sample_rate
        lstm = merge_dict(self.LSTM_DEFAULTS, lstm)
        lstm["batch_first"] = True
        linear = merge_dict(self.LINEAR_DEFAULTS, linear)
        num_channels = self.NUM_CHANNELS_DEFAULTS
        self.save_hyperparameters("sincnet", "lstm", "linear", "num_channels")

        self.wav_norm1d = nn.InstanceNorm1d(1, affine=True)

        self.conv1d_spatial = nn.ModuleList()
        self.pool1d_spatial = nn.ModuleList()
        self.norm1d_spatial = nn.ModuleList()

        self.conv1d_spatial.append(nn.Conv1d(65, 60, 5, stride=1))
        self.pool1d_spatial.append(nn.MaxPool1d(3, stride=3, padding=0, dilation=1))
        self.norm1d_spatial.append(nn.InstanceNorm1d(60, affine=True))

        self.conv1d_spatial.append(nn.Conv1d(60, 60, 5, stride=1))
        self.pool1d_spatial.append(nn.MaxPool1d(3, stride=3, padding=0, dilation=1))
        self.norm1d_spatial.append(nn.InstanceNorm1d(60, affine=True))

        self.conv1d_spatial.append(nn.Conv1d(60, 60, 5, stride=1))
        self.pool1d_spatial.append(nn.MaxPool1d(3, stride=3, padding=0, dilation=1))
        self.norm1d_spatial.append(nn.InstanceNorm1d(60, affine=True))        


        self.sincnet = SincNet(**self.hparams.sincnet)

        monolithic = lstm["monolithic"]
        if monolithic:
            multi_layer_lstm = dict(lstm)
            del multi_layer_lstm["monolithic"]
            self.lstm = nn.LSTM(120, **multi_layer_lstm)

        else:
            num_layers = lstm["num_layers"]
            if num_layers > 1:
                self.dropout = nn.Dropout(p=lstm["dropout"])

            one_layer_lstm = dict(lstm)
            one_layer_lstm["num_layers"] = 1
            one_layer_lstm["dropout"] = 0.0
            del one_layer_lstm["monolithic"]

            self.lstm = nn.ModuleList(
                [
                    nn.LSTM(
                        120
                        if i == 0
                        else lstm["hidden_size"] * (2 if lstm["bidirectional"] else 1),
                        **one_layer_lstm
                    )
                    for i in range(num_layers)
                ]
            )

        if linear["num_layers"] < 1:
            return

        lstm_out_features: int = self.hparams.lstm["hidden_size"] * (
            2 if self.hparams.lstm["bidirectional"] else 1
        )
        self.linear = nn.ModuleList(
            [
                nn.Linear(in_features, out_features)
                for in_features, out_features in pairwise(
                    [lstm_out_features,]
                    + [self.hparams.linear["hidden_size"]]
                    * self.hparams.linear["num_layers"]
                )
            ]
        )

    def build(self):

        if self.hparams.linear["num_layers"] > 0:
            in_features = self.hparams.linear["hidden_size"]
        else:
            in_features = self.hparams.lstm["hidden_size"] * (
                2 if self.hparams.lstm["bidirectional"] else 1
            )

        self.classifier = nn.Linear(in_features, len(self.specifications.classes))
        self.activation = self.default_activation()

    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        """Pass forward

        Parameters
        ----------
        waveforms : (batch, channel, sample)

        Returns
        -------
        scores : (batch, frame, classes)
        """

        stereo_channel1 = waveforms[:, 0, :].unsqueeze(1)
       
        stereo_channel2 = waveforms[:, 1, :].unsqueeze(1)
   
        '''
        if waveforms.shape[1] == 2:
            stereo_channel2 = waveforms[:, 1, :].unsqueeze(1)
        else :
            stereo_channel1 = waveforms[:, 7, :].unsqueeze(1)
            stereo_channel2 = waveforms[:, 7, :].unsqueeze(1)
        '''
        ##IPD
        
        normalized_input1 = self.wav_norm1d(stereo_channel1)

        normalized_input2 = self.wav_norm1d(stereo_channel2)

        normalized_stereo  = torch.cat([normalized_input1, normalized_input2], dim=1)
        

        stft = self.spectrogram(normalized_stereo)

        ipd = torch.angle(stft[:, 1:]) - torch.angle(stft[:, 0:1])
        ipd = ipd % (2 * torch.pi)
        ild =  torch.log(stft[:, 0:1].abs() + 1e-8) - torch.log(stft[:, 1:].abs()+ 1e-8)

        pattern = "batch channel freq frame -> batch (channel freq) frame "
        
        ipd = rearrange(ipd, pattern)
        ild = rearrange(ild, pattern)
        
        for (conv1d, pool1d, norm1d) in zip(self.conv1d_spatial, self.pool1d_spatial, self.norm1d_spatial):
            ipd = F.leaky_relu(norm1d(pool1d(conv1d(ipd))))#[1, 60, 276]
            ild = F.leaky_relu(norm1d(pool1d(conv1d(ild))))

        ### something bad is appening here
        ipd = torch.nan_to_num(ipd)
        assert not (torch.isnan(ipd).any())
        ####

        output1_sincnet = self.sincnet(stereo_channel1)
     
        #output2_sincnet = self.sincnet(stereo_channel2)
    
        if output1_sincnet.shape[2] != ipd.shape[2]:
            ipd = F.pad(ipd, (0, output1_sincnet.shape[2]- ipd.shape[2]), 'replicate')
            ild = F.pad(ild, (0, output1_sincnet.shape[2]- ipd.shape[2]), 'replicate')

        outputs  = torch.cat([output1_sincnet, ipd], dim=1)
        
        if self.hparams.lstm["monolithic"]:
            outputs, _ = self.lstm(
                rearrange(outputs, "batch feature frame -> batch frame feature")
            )
        else:
            outputs = rearrange(outputs, "batch feature frame -> batch frame feature")
            for i, lstm in enumerate(self.lstm):
                outputs, _ = lstm(outputs)
                if i + 1 < self.hparams.lstm["num_layers"]:
                    outputs = self.dropout(outputs)

        if self.hparams.linear["num_layers"] > 0:
            for linear in self.linear:
                outputs = F.leaky_relu(linear(outputs))

        return self.activation(self.classifier(outputs))
