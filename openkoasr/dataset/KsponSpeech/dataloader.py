#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright (c) 2025 Gwantae Kim. All Rights Reserved.
# Licensed under the MIT License.

import os
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np
#import soundfile as sf
import librosa

from openkoasr.dataset.KsponSpeech import utils

class KsponSpeechDataset(Dataset):
    def __init__(self, config):
        self.rootpath = config.rootpath
        self.subset = config.subset
        self.label_file = os.path.join(self.rootpath, "KsponSpeech_scripts", f"eval_{self.subset}.trn")
        self.data = self.parse_data()

    def generate_dataloader(self, batch_size, shuffle, num_workers):
        self.data = self.parse_data()
        
        dataloader = DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers
        )
        return dataloader

    def parse_data(self):
        # Load label file
        with open(self.label_file, 'r', encoding='utf-8') as f:
            labels = f.readlines()

        # parse data
        data = []
        for sample in tqdm(labels):
            audiofile, text = sample.split("::")[0][:-1], sample.split("::")[1][1:]
            text = utils.normalize_text(text)
            data.append((audiofile, text))
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        audio_file = os.path.join(self.rootpath, self.data[idx][0])

        # Load audio file
        #audio = np.memmap(audio_file, dtype='h', mode='r').astype(np.float32) / 32767.0 
        #audio = torch.from_numpy(audio)
        # audio = torchaudio.load(audio_file)
        #audio, sample_rate = torchaudio.load_with_torchcodec(
        #    audio_file,
        #    format="s16le",  # PCM 파일의 인코딩 포맷을 지정 (예: 16비트 리틀엔디언)
        #    channels_first=True  # True면 [채널, 시간], False면 [시간, 채널]
        #)

        with open(audio_file, 'rb') as pcm_file:
            buf = pcm_file.read()
            if len(buf) % 2 != 0:
                buf = buf[:-1]  # 짝수 바이트로 맞추기
            pcm_data = np.frombuffer(buf, dtype=np.int16)
            audio = librosa.util.buf_to_float(pcm_data, n_bytes=2)
        #audio, sample_rate = sf.read(audio_file, dtype='int16')
        #audio = audio.astype(np.float32) / 32768.0  # Normalize
        audio = torch.from_numpy(audio)

        # Load label file
        label = self.data[idx][1]

        return audio, label

# Example usage
if __name__ == "__main__":
    from configs import dataset_config_dict
    rootpath = dataset_config_dict['KsponSpeech']
    dataset = KsponSpeechDataset(rootpath)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)

    for batch_idx, (audio_data, labels) in enumerate(dataloader):
        print(f"Batch {batch_idx}:")
        print(f"Audio Data Shape: {audio_data.shape}")
        print(f"Labels: {labels}")
