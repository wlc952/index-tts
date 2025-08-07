import os
import sys
import time
from typing import Dict, List, Tuple
import ctypes

import torch
import torchaudio
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from omegaconf import OmegaConf
import numpy as np
from untool import (
    get_model_info_p,
    convert_model_info,
    move_to_device,
    compile_io_addr,
    fill_api_info,
    run_model,
    free_model,
    untensor_create,
    untensor_destroy,
    untensor_sync,
    untensor_s2d_bytes,
    untensor_d2d_bytes_offset,
    untensor_show,
)

# 添加当前目录到Python路径，以支持绝对导入
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.feature_extractors import MelSpectrogramFeatures
from utils.front import TextNormalizer, TextTokenizer
from utils.common import cal_topk

class IndexTTS:
    def __init__(
        self,
        cfg_path="checkpoints/config.yaml",
        model_dir="checkpoints",
        device_id: int = 0,
    ):
        self.device = "cpu"
        self.device_id = device_id

        self.cfg = OmegaConf.load(cfg_path)
        self.model_dir = model_dir
        self.bpe_path = os.path.join(self.model_dir, "bpe.model")
        self.bmodel_path = os.path.join(
            self.model_dir, "indextts_bm1684x_f16_seq256.bmodel"
        )
        self.normalizer = TextNormalizer()
        self.normalizer.load()
        print(">> TextNormalizer loaded")
        self.tokenizer = TextTokenizer(self.bpe_path, self.normalizer)
        print(">> bpe model loaded from:", self.bpe_path)

        self.cache_audio_prompt = None
        self.cache_cond_mel = None
        self.model_version = 1.5
        self.stop_text_token = 1
        self.start_text_token = 0
        self.start_mel_token = 8192
        self.stop_mel_token = 8193
        self.mel_length_compression = 1024
        self.SEQLEN = 256

        self._init_bmodel(self.bmodel_path, self.device_id)

    def _init_bmodel(self, model_path: str, device_id: int):
        self.model_info_p = get_model_info_p(model_path, device_id)
        move_to_device(self.model_info_p)
        compile_io_addr(self.model_info_p)
        fill_api_info(self.model_info_p)
        self.model_info_c_p = convert_model_info(self.model_info_p)
        self.bm_handle = self.model_info_c_p.contents.bm_handle
        self._init_idx()
        self._init_tensors()
        self.free_model = free_model
        self.untensor_destroy = untensor_destroy

    def _init_idx(self):
        self.bigvgan_idx = 58
        self.speaker_encoder_idx = 57
        self.final_norm_idx = 56
        self.ln_f2_idx = 55
        self.ln_f_idx = 54
        self.penalty_sample_head_idx = 53
        self.lm_head_idx = 52
        self.conds_encoder_idx = 51
        self.text_embedding_cache_idx = 50
        self.text_embedding_idx = 49
        self.mel_embedding_idx = 48
        self.num_blocks = 24
        self.block_ids = []
        self.block_cache_ids = []
        for i in range(self.num_blocks):
            self.block_ids.append(2 * i)
            self.block_cache_ids.append(2 * i + 1)

    def _init_tensors(self):
        net_num = self.model_info_c_p.contents.net_num
        self.input_tensors = []
        self.output_tensors = []
        for i in range(net_num):
            net_x_input_tensors = []
            net_x_output_tensors = []
            stage_info = self.model_info_c_p.contents.nets[i].stages[0]
            io_alone = stage_info.io_alone
            if io_alone:
                io_device = stage_info.io_device
            else:
                io_device = self.model_info_c_p.contents.neuron_device
            input_num = stage_info.input_num
            output_num = stage_info.output_num
            for i in range(input_num):
                src_tensor = stage_info.input_tensor[i]
                dst_tensor = untensor_create()
                dst_tensor.contents.name = src_tensor.name
                dst_tensor.contents.dtype = src_tensor.data_type
                dst_tensor.contents.dims = src_tensor.dims
                dst_tensor.contents.shape = src_tensor.shape
                dst_tensor.contents.size = src_tensor.size
                dst_tensor.contents.device_id = self.device_id
                dst_tensor.contents.bm_handle = self.bm_handle
                dst_tensor.contents.is_in_device = True
                dst_tensor.contents.addr = stage_info.input_tensor_global_addr[i]
                dst_tensor.contents.device_start = io_device.addr
                dst_tensor.contents.device_size = io_device.size
                dst_tensor.contents.dmabuf_fd = io_device.dmabuf_fd
                dst_tensor.contents.reserved = io_device.reserved
                dst_tensor.contents.rawflags = io_device.rawflags
                dst_tensor.contents.offset = dst_tensor.contents.addr - io_device.addr
                net_x_input_tensors.append(dst_tensor)
            for i in range(output_num):
                src_tensor = stage_info.output_tensor[i]
                dst_tensor = untensor_create()
                dst_tensor.contents.name = src_tensor.name
                dst_tensor.contents.dtype = src_tensor.data_type
                dst_tensor.contents.dims = src_tensor.dims
                dst_tensor.contents.shape = src_tensor.shape
                dst_tensor.contents.size = src_tensor.size
                dst_tensor.contents.device_id = self.device_id
                dst_tensor.contents.bm_handle = self.bm_handle
                dst_tensor.contents.is_in_device = True
                dst_tensor.contents.addr = stage_info.output_tensor_global_addr[i]
                dst_tensor.contents.device_start = io_device.addr
                dst_tensor.contents.device_size = io_device.size
                dst_tensor.contents.dmabuf_fd = io_device.dmabuf_fd
                dst_tensor.contents.reserved = io_device.reserved
                dst_tensor.contents.rawflags = io_device.rawflags
                dst_tensor.contents.offset = dst_tensor.contents.addr - io_device.addr
                net_x_output_tensors.append(dst_tensor)
            self.input_tensors.append(net_x_input_tensors)
            self.output_tensors.append(net_x_output_tensors)

    def s2d_bytes(self, src, data, byte_size):
        untensor_s2d_bytes(src, data, byte_size)

    def d2d_bytes_offset(self, dst, src, dst_offset, src_offset, byte_size):
        untensor_d2d_bytes_offset(
            self.bm_handle, dst, src, dst_offset, src_offset, byte_size
        )

    def run(self, net_idx: int, stage_idx: int = 0):
        run_model(self.model_info_p, net_idx, stage_idx)

    def bigvgan(self, latent: np.ndarray):
        latent_ptr = ctypes.c_void_p(latent.ctypes.data)
        out_tensor_ = self.output_tensors[self.speaker_encoder_idx][0]
        in_tensor_0 = self.input_tensors[self.bigvgan_idx][0]
        in_tensor_1 = self.input_tensors[self.bigvgan_idx][1]
        out_tensor_0 = self.output_tensors[self.bigvgan_idx][0]

        self.s2d_bytes(in_tensor_0, latent_ptr, latent.nbytes)
        self.d2d_bytes_offset(in_tensor_1, out_tensor_, 0, 0, out_tensor_.contents.size)
        self.run(self.bigvgan_idx)

        untensor_sync(out_tensor_0, False, True)
        shape = tuple(out_tensor_0.contents.shape[: out_tensor_0.contents.dims])
        size = out_tensor_0.contents.size // 4  # f32
        buf_type = ctypes.c_float * size
        buf = ctypes.cast(out_tensor_0.contents.data, ctypes.POINTER(buf_type)).contents
        audio = np.frombuffer(buf, dtype=np.float32).reshape(shape)
        audio = torch.from_numpy(audio)
        return audio

    def build_aligned_inputs_and_targets(self, input, start_token, stop_token):
        inp = F.pad(input, (1, 0), value=start_token)
        tar = F.pad(input, (0, 1), value=stop_token)
        return inp, tar

    def set_mel_padding(self, mel_input_tokens, mel_lengths):
        """
        Given mel tokens that are derived from a padded audio clip and the actual lengths of each batch element in
        that audio clip, reformats the tokens with STOP_MEL_TOKEN in place of the zero padding. This is required
        preformatting to create a working TTS model.
        """
        for b in range(len(mel_lengths)):
            # Due to the convolutional nature of how these tokens are generated,
            # it would be best if the model predicts a token past the actual last token.
            actual_end = mel_lengths[b]
            if actual_end < mel_input_tokens.shape[-1]:
                mel_input_tokens[b, actual_end:] = self.stop_mel_token
        return mel_input_tokens

    def set_text_padding(self, text_input_tokens, text_lengths):
        """
        Given mel tokens that are derived from a padded audio clip and the actual lengths of each batch element in
        that audio clip, reformats the tokens with STOP_MEL_TOKEN in place of the zero padding. This is required
        preformatting to create a working TTS model.
        """
        for b in range(len(text_lengths)):
            # Due to the convolutional nature of how these tokens are generated,
            # it would be best if the model predicts a token past the actual last token.
            actual_end = text_lengths[b]
            if actual_end < text_input_tokens.shape[-1]:
                text_input_tokens[b, actual_end:] = self.stop_text_token
        return text_input_tokens

    def gpt(self, text_inputs, text_lengths, mel_codes, wav_lengths):
        mel_codes_lengths = (
            torch.ceil(wav_lengths / self.mel_length_compression).long() + 1
        )
        mel_codes = self.set_mel_padding(mel_codes, mel_codes_lengths)
        mel_codes = F.pad(mel_codes, (0, 1), value=self.stop_mel_token)
        mel_codes, mel_targets = self.build_aligned_inputs_and_targets(
            mel_codes, self.start_mel_token, self.stop_mel_token
        )
        self.L2 = mel_codes.size(1)
        mel_codes = F.pad(
            mel_codes, (0, 256 - mel_codes.size(1)), value=self.stop_mel_token
        )
        mel_codes = mel_codes.cpu().numpy()
        mel_codes_ptr = ctypes.c_void_p(mel_codes.ctypes.data)
        in_tensor = self.input_tensors[self.mel_embedding_idx][0]
        out_tensor = self.output_tensors[self.mel_embedding_idx][0]
        self.s2d_bytes(in_tensor, mel_codes_ptr, in_tensor.contents.size)
        self.run(self.mel_embedding_idx)
        untensor_sync(out_tensor, False, True)
        shape = (1, 256, 1280)
        size = out_tensor.contents.size // 2  # float16
        buf_type = ctypes.c_uint16 * size
        buf = ctypes.cast(out_tensor.contents.data, ctypes.POINTER(buf_type)).contents
        self.mel_emb2 = np.frombuffer(buf, dtype=np.uint16).reshape(shape)[
            :, : self.L2, :
        ]
        emb = np.concatenate([self.mel_emb, self.text_emb, self.mel_emb2], axis=1)
        emb = np.pad(
            emb,
            ((0, 0), (0, 256 - emb.shape[1]), (0, 0)),
        )
        emb_ptr = ctypes.c_void_p(emb.ctypes.data)

        # gpt2 blocks first
        in_tensor = self.input_tensors[self.block_ids[0]][0]
        bytes_size = in_tensor.contents.size // 256

        self.token_length = 32 + self.L1 + self.L2
        attn_mask_ptr = self.get_first_mask_ptr(self.SEQLEN, self.token_length)

        out_tensor = None
        for i in range(self.num_blocks):
            net_id = self.block_ids[i]
            in_tensor_0 = self.input_tensors[net_id][0]

            if i == 0:
                in_tensor_1 = self.input_tensors[net_id][1]
                self.s2d_bytes(in_tensor_1, attn_mask_ptr, in_tensor_1.contents.size)
                self.s2d_bytes(in_tensor_0, emb_ptr, emb.nbytes)
            else:
                self.d2d_bytes_offset(
                    in_tensor_0, out_tensor, 0, 0, in_tensor_0.contents.size
                )
            self.run(net_id)
            out_tensor = self.output_tensors[net_id][0]

        # ln_f
        in_tensor = self.input_tensors[self.ln_f2_idx][0]
        src_offset = bytes_size * 32
        self.d2d_bytes_offset(in_tensor, out_tensor, 0, src_offset, bytes_size * 224)
        self.run(self.ln_f2_idx)
        out_tensor = self.output_tensors[self.ln_f2_idx][0]
        in_tensor = self.input_tensors[self.final_norm_idx][0]
        self.d2d_bytes_offset(in_tensor, out_tensor, 0, 0, in_tensor.contents.size)
        self.run(self.final_norm_idx)

        out_tensor = self.output_tensors[self.final_norm_idx][0]
        untensor_sync(out_tensor, False, True)
        shape = (1, 224, 1280)
        size = out_tensor.contents.size // 4
        buf_type = ctypes.c_float * size
        buf = ctypes.cast(out_tensor.contents.data, ctypes.POINTER(buf_type)).contents
        latent = np.frombuffer(buf, dtype=np.float32).reshape(shape)[
            :, self.L1 : self.L1 + self.L2 - 2, :
        ]  # [1, L1 : L1 + L2 - 2, 1280]
        self.ori_latent_len = latent.shape[1]
        latent = np.pad(
            latent,
            ((0, 0), (0, 224 - latent.shape[1]), (0, 0)),
            mode="constant",
            constant_values=0,
        )
        return latent

    def inference_speech(
        self, text_inputs, top_k=1, top_p=1.0, temperature=1.0, repetition_penalty=10.0
    ):
        text_inputs = F.pad(text_inputs, (0, 1), value=self.stop_text_token)
        text_inputs, _ = self.build_aligned_inputs_and_targets(
            text_inputs, self.start_text_token, self.stop_text_token
        )
        self.L1 = text_inputs.size(1)
        text_inputs = F.pad(
            text_inputs, (0, 256 - text_inputs.size(1)), value=self.stop_text_token
        )
        text_inputs = text_inputs.cpu().numpy()
        text_inputs_ptr = ctypes.c_void_p(text_inputs.ctypes.data)
        position_ids = np.zeros(self.SEQLEN, dtype=np.int32)
        position_ids[: self.L1] = np.arange(self.L1, dtype=np.int32)
        position_ids_ptr = ctypes.c_void_p(position_ids.ctypes.data)

        in_tensor_0 = self.input_tensors[self.text_embedding_idx][0]
        in_tensor_1 = self.input_tensors[self.text_embedding_idx][1]
        out_tensor = self.output_tensors[self.text_embedding_idx][0]
        self.s2d_bytes(in_tensor_0, text_inputs_ptr, in_tensor_0.contents.size)
        self.s2d_bytes(in_tensor_1, position_ids_ptr, in_tensor_1.contents.size)
        self.run(self.text_embedding_idx)
        untensor_sync(out_tensor, False, True)
        shape = (1, 256, 1280)
        buf_type = ctypes.c_uint16 * (out_tensor.contents.size // 2)
        buf = ctypes.cast(out_tensor.contents.data, ctypes.POINTER(buf_type)).contents
        self.text_emb = np.frombuffer(buf, dtype=np.uint16).reshape(shape)[
            :, : self.L1, :
        ]

        codes = self._inference_speech_generate(
            top_k, top_p, temperature, repetition_penalty
        )
        return torch.tensor([codes])

    def _inference_speech_generate(
        self, top_k=1, top_p=0.8, temperature=1.0, penalty=10.0
    ):
        self.token_length = 32 + self.L1
        token = self.start_mel_token
        pos = 0

        first = True
        codes = [1] * self.token_length + [token]

        while self.token_length < self.SEQLEN and token != self.stop_mel_token:
            self.token_length += 1
            pos += 1
            input_ids = np.array([token], dtype=np.int32)
            input_ids_ptr = ctypes.c_void_p(input_ids.ctypes.data)
            position_id = np.array([pos - 1], dtype=np.int32)
            position_id_ptr = ctypes.c_void_p(position_id.ctypes.data)
            in_tensor_0 = self.input_tensors[self.text_embedding_cache_idx][0]
            in_tensor_1 = self.input_tensors[self.text_embedding_cache_idx][1]
            out_tensor = self.output_tensors[self.text_embedding_cache_idx][0]
            self.s2d_bytes(in_tensor_0, input_ids_ptr, in_tensor_0.contents.size)
            self.s2d_bytes(in_tensor_1, position_id_ptr, in_tensor_1.contents.size)
            self.run(self.text_embedding_cache_idx)
            untensor_sync(out_tensor, False, True)
            shape = (1, 1, 1280)
            buf_type = ctypes.c_uint16 * (out_tensor.contents.size // 2)
            buf = ctypes.cast(
                out_tensor.contents.data, ctypes.POINTER(buf_type)
            ).contents
            self.text_cache_emb = np.frombuffer(buf, dtype=np.uint16).reshape(shape)

            # gpt2 blocks
            if first:
                self._gpt_forword_first()
                first = False
                pos += 1  # 第一次的position_id是0, 第二次从2开始。
            else:
                self._gpt_forward_next()
            out_tensor = self.output_tensors[self.ln_f_idx][0]
            # untensor_show(out_tensor, 0, 10, b"d")

            # lm_head
            in_tensor = self.input_tensors[self.lm_head_idx][0]
            self.d2d_bytes_offset(
                in_tensor,
                out_tensor,
                0,
                0,
                out_tensor.contents.size,
            )
            self.run(self.lm_head_idx)
            out_tensor = self.output_tensors[self.lm_head_idx][0]
            # untensor_show(out_tensor, 0, 10, b"d")
            # breakpoint()

            # penalty_sample_head
            in_tensor_0 = self.input_tensors[self.penalty_sample_head_idx][0]
            in_tensor_1 = self.input_tensors[self.penalty_sample_head_idx][1]
            in_tensor_2 = self.input_tensors[self.penalty_sample_head_idx][2]
            in_tensor_3 = self.input_tensors[self.penalty_sample_head_idx][3]
            in_tensor_4 = self.input_tensors[self.penalty_sample_head_idx][4]

            self.d2d_bytes_offset(
                in_tensor_0,
                out_tensor,
                0,
                0,
                in_tensor_0.contents.size,
            )
            input_ids_tensor = np.array([codes], dtype=np.int32)
            input_ids_tensor = np.pad(
                input_ids_tensor,
                ((0, 0), (0, 256 - len(codes))),
                constant_values=codes[-1],
            )
            input_ids_ptr = ctypes.c_void_p(input_ids_tensor.ctypes.data)
            self.s2d_bytes(in_tensor_1, input_ids_ptr, in_tensor_1.contents.size)
            top_p_tensor = np.array([top_p], dtype=np.float32)
            top_p_tensor_ptr = ctypes.c_void_p(top_p_tensor.ctypes.data)
            self.s2d_bytes(in_tensor_2, top_p_tensor_ptr, in_tensor_2.contents.size)
            temperature_tensor = np.array([temperature], dtype=np.float32)
            temperature_tensor_ptr = ctypes.c_void_p(temperature_tensor.ctypes.data)
            self.s2d_bytes(
                in_tensor_3, temperature_tensor_ptr, in_tensor_3.contents.size
            )
            penalty_tensor = np.array([penalty], dtype=np.float32)
            penalty_tensor_ptr = ctypes.c_void_p(penalty_tensor.ctypes.data)
            self.s2d_bytes(in_tensor_4, penalty_tensor_ptr, in_tensor_4.contents.size)
            self.run(self.penalty_sample_head_idx)
            out_tensor = self.output_tensors[self.penalty_sample_head_idx][1]

            # get token
            untensor_sync(out_tensor, False, True)
            shape = (30,)
            size = out_tensor.contents.size // 4  # int32
            buf_type = ctypes.c_int32 * size
            buf = ctypes.cast(
                out_tensor.contents.data, ctypes.POINTER(buf_type)
            ).contents
            tokens = np.frombuffer(buf, dtype=np.int32).reshape(shape)
            # breakpoint()

            token = tokens[0]
            codes.append(token)

        return codes[33 + self.L1 :]

    def _gpt_forword_first(self):
        attn_mask_ptr = self.get_first_mask_ptr(self.SEQLEN, self.token_length)

        emb = np.concatenate([self.mel_emb, self.text_emb, self.text_cache_emb], axis=1)
        emb = np.pad(
            emb,
            ((0, 0), (0, 256 - emb.shape[1]), (0, 0)),
        )

        emb_ptr = ctypes.c_void_p(emb.ctypes.data)

        out_tensor = self.output_tensors[self.block_ids[0]][0]
        bytes_size = out_tensor.contents.size // 256
        for i in range(self.num_blocks):
            net_id = self.block_ids[i]
            cache_id = self.block_cache_ids[i]
            in_tensor_0 = self.input_tensors[net_id][0]

            if i == 0:
                in_tensor_1 = self.input_tensors[net_id][1]
                self.s2d_bytes(in_tensor_1, attn_mask_ptr, in_tensor_1.contents.size)
                self.s2d_bytes(in_tensor_0, emb_ptr, emb.nbytes)
            else:
                self.d2d_bytes_offset(
                    in_tensor_0, out_tensor, 0, 0, in_tensor_0.contents.size
                )

            self.run(net_id)

            out_tensor = self.output_tensors[net_id][0]
            # untensor_show(out_tensor, 0, 10, b"d")
            # breakpoint()
            self.d2d_bytes_offset(
                self.input_tensors[cache_id][2],
                self.output_tensors[net_id][1],
                0,
                0,
                self.output_tensors[net_id][1].contents.size,
            )
            self.d2d_bytes_offset(
                self.input_tensors[cache_id][3],
                self.output_tensors[net_id][2],
                0,
                0,
                self.output_tensors[net_id][2].contents.size,
            )

        in_tensor = self.input_tensors[self.ln_f_idx][0]
        src_offset = bytes_size * (self.token_length - 1)
        self.d2d_bytes_offset(in_tensor, out_tensor, 0, src_offset, bytes_size)
        self.run(self.ln_f_idx)

    def _gpt_forward_next(self):
        attn_mask_ptr = self.get_next_mask_ptr(self.SEQLEN, self.token_length)
        out_tensor = self.output_tensors[self.text_embedding_cache_idx][0]
        block_cache_0 = self.block_cache_ids[0]
        bytes_size = self.output_tensors[block_cache_0][1].contents.size
        dst_offset = bytes_size * (self.token_length - 1)

        for i in range(self.num_blocks):
            id = self.block_cache_ids[i]
            self.d2d_bytes_offset(
                self.input_tensors[id][0], out_tensor, 0, 0, out_tensor.contents.size
            )
            if i == 0:
                self.s2d_bytes(
                    self.input_tensors[id][1],
                    attn_mask_ptr,
                    self.input_tensors[id][1].contents.size,
                )
            else:
                self.d2d_bytes_offset(
                    self.input_tensors[id][1],
                    self.input_tensors[block_cache_0][1],
                    0,
                    0,
                    self.input_tensors[id][1].contents.size,
                )

            self.run(id)
            out_tensor = self.output_tensors[id][0]
            # untensor_show(self.input_tensors[id][0], 0, 10, b"d")
            # untensor_show(self.input_tensors[id][1], 0, 10, b"d")
            # untensor_show(out_tensor, 0, 10, b"d")
            # breakpoint()
            self.d2d_bytes_offset(
                self.input_tensors[id][2],
                self.output_tensors[id][1],
                dst_offset,
                0,
                bytes_size,
            )
            self.d2d_bytes_offset(
                self.input_tensors[id][3],
                self.output_tensors[id][2],
                dst_offset,
                0,
                bytes_size,
            )

        # ln_f
        in_tensor = self.input_tensors[self.ln_f_idx][0]
        self.d2d_bytes_offset(in_tensor, out_tensor, 0, 0, out_tensor.contents.size)
        self.run(self.ln_f_idx)

    def get_first_mask_ptr(self, seq_len, token_len):
        self._attn_mask = np.full((seq_len, seq_len), 0xF0E2, dtype=np.uint16)
        rows = np.arange(token_len).reshape(-1, 1)
        cols = np.arange(seq_len).reshape(1, -1)
        self._attn_mask[:token_len, :] = np.where(
            cols <= rows, 0, self._attn_mask[:token_len, :]
        )
        return ctypes.c_void_p(self._attn_mask.ctypes.data)

    def get_next_mask_ptr(self, seq_len, token_len):
        self._attn_mask = np.zeros(seq_len + 1, dtype=np.uint16)
        self._attn_mask[token_len - 1 : seq_len] = 0xF0E2
        return ctypes.c_void_p(self._attn_mask.ctypes.data)

    def remove_long_silence(
        self, codes: torch.Tensor, silent_token=52, max_consecutive=30
    ):
        code_lens = []
        codes_list = []
        device = codes.device
        dtype = codes.dtype
        isfix = False
        for i in range(0, codes.shape[0]):
            code = codes[i]
            if self.cfg.gpt.stop_mel_token not in code:
                code_lens.append(len(code))
                len_ = len(code)
            else:
                # len_ = code.cpu().tolist().index(8193)+1
                len_ = (code == self.stop_mel_token).nonzero(as_tuple=False)[0] + 1
                len_ = len_ - 2

            count = torch.sum(code == silent_token).item()
            if count > max_consecutive:
                code = code.cpu().tolist()
                ncode = []
                n = 0
                for k in range(0, len_):
                    if code[k] != silent_token:
                        ncode.append(code[k])
                        n = 0
                    elif code[k] == silent_token and n < 10:
                        ncode.append(code[k])
                        n += 1
                    # if (k == 0 and code[k] == 52) or (code[k] == 52 and code[k-1] == 52):
                    #    n += 1
                len_ = len(ncode)
                ncode = torch.LongTensor(ncode)
                codes_list.append(ncode.to(device, dtype=dtype))
                isfix = True
                # codes[i] = self.stop_mel_token
                # codes[i, 0:len_] = ncode
            else:
                codes_list.append(codes[i])
            code_lens.append(len_)

        codes = pad_sequence(codes_list, batch_first=True) if isfix else codes[:, :-2]
        code_lens = torch.LongTensor(code_lens).to(device, dtype=dtype)
        return codes, code_lens

    def bucket_sentences(self, sentences, bucket_max_size=4) -> List[List[Dict]]:
        """
        Sentence data bucketing.
        if ``bucket_max_size=1``, return all sentences in one bucket.
        """
        outputs: List[Dict] = []
        for idx, sent in enumerate(sentences):
            outputs.append({"idx": idx, "sent": sent, "len": len(sent)})

        if len(outputs) > bucket_max_size:
            # split sentences into buckets by sentence length
            buckets: List[List[Dict]] = []
            factor = 1.5
            last_bucket = None
            last_bucket_sent_len_median = 0

            for sent in sorted(outputs, key=lambda x: x["len"]):
                current_sent_len = sent["len"]
                if current_sent_len == 0:
                    print(">> skip empty sentence")
                    continue
                if (
                    last_bucket is None
                    or current_sent_len >= int(last_bucket_sent_len_median * factor)
                    or len(last_bucket) >= bucket_max_size
                ):
                    # new bucket
                    buckets.append([sent])
                    last_bucket = buckets[-1]
                    last_bucket_sent_len_median = current_sent_len
                else:
                    # current bucket can hold more sentences
                    last_bucket.append(sent)  # sorted
                    mid = len(last_bucket) // 2
                    last_bucket_sent_len_median = last_bucket[mid]["len"]
            last_bucket = None
            # merge all buckets with size 1
            out_buckets: List[List[Dict]] = []
            only_ones: List[Dict] = []
            for b in buckets:
                if len(b) == 1:
                    only_ones.append(b[0])
                else:
                    out_buckets.append(b)
            if len(only_ones) > 0:
                # merge into previous buckets if possible
                # print("only_ones:", [(o["idx"], o["len"]) for o in only_ones])
                for i in range(len(out_buckets)):
                    b = out_buckets[i]
                    if len(b) < bucket_max_size:
                        b.append(only_ones.pop(0))
                        if len(only_ones) == 0:
                            break
                # combined all remaining sized 1 buckets
                if len(only_ones) > 0:
                    out_buckets.extend(
                        [
                            only_ones[i : i + bucket_max_size]
                            for i in range(0, len(only_ones), bucket_max_size)
                        ]
                    )
            return out_buckets
        return [outputs]

    def pad_tokens_cat(self, tokens: List[torch.Tensor]) -> torch.Tensor:
        if self.model_version and self.model_version >= 1.5:
            # 1.5版本以上，直接使用stop_text_token 右侧填充，填充到最大长度
            # [1, N] -> [N,]
            tokens = [t.squeeze(0) for t in tokens]
            return pad_sequence(
                tokens,
                batch_first=True,
                padding_value=self.cfg.gpt.stop_text_token,
                padding_side="right",
            )
        max_len = max(t.size(1) for t in tokens)
        outputs = []
        for tensor in tokens:
            pad_len = max_len - tensor.size(1)
            if pad_len > 0:
                n = min(8, pad_len)
                tensor = torch.nn.functional.pad(
                    tensor, (0, n), value=self.cfg.gpt.stop_text_token
                )
                tensor = torch.nn.functional.pad(
                    tensor, (0, pad_len - n), value=self.cfg.gpt.start_text_token
                )
            tensor = tensor[:, :max_len]
            outputs.append(tensor)
        tokens = torch.cat(outputs, dim=0)
        return tokens

    def infer(
        self,
        audio_prompt,
        text,
        output_path,
        verbose=False,
        max_text_tokens_per_sentence=80,
        top_k=1,
        top_p=0.8,
        temperature=1.0,
        repetition_penalty=10.0,
    ):
        print(">> start inference...")
        if verbose:
            print(f"origin text:{text}")
        start_time = time.perf_counter()

        # 如果参考音频改变了，才需要重新生成 cond_mel, 提升速度
        if self.cache_cond_mel is None or self.cache_audio_prompt != audio_prompt:
            audio, sr = torchaudio.load(audio_prompt)
            audio = torch.mean(audio, dim=0, keepdim=True)
            if audio.shape[0] > 1:
                audio = audio[0].unsqueeze(0)
            audio = torchaudio.transforms.Resample(sr, 24000)(audio)
            cond_mel = MelSpectrogramFeatures()(audio).to(self.device)

            max_mel_tokens = 300
            cond_mel = cond_mel[..., :max_mel_tokens]  # 截断多余部分
            if cond_mel.shape[-1] < max_mel_tokens:
                pad_len = max_mel_tokens - cond_mel.shape[-1]
                cond_mel = F.pad(cond_mel, (0, pad_len), value=0.0)  # 用0填充

            cond_mel = cond_mel.transpose(1, 2).contiguous().numpy()

            cond_mel_frame = cond_mel.shape[-1]
            if verbose:
                print(f"cond_mel shape: {cond_mel.shape}", "dtype:", cond_mel.dtype)
                print(f"cond_mel: {cond_mel[0, 0, :10]}")

            self.cache_audio_prompt = audio_prompt
            self.cache_cond_mel = cond_mel
            cond_mel_ptr = ctypes.c_void_p(self.cache_cond_mel.ctypes.data)
            self.cond_mel_ptr = cond_mel_ptr

            in_tensor = self.input_tensors[self.conds_encoder_idx][0]
            self.s2d_bytes(in_tensor, cond_mel_ptr, in_tensor.contents.size)
            mel_len = np.array([max_mel_tokens], dtype=np.int32)
            mel_len_ptr = ctypes.c_void_p(mel_len.ctypes.data)
            in_tensor = self.input_tensors[self.conds_encoder_idx][1]
            out_tensor = self.output_tensors[self.conds_encoder_idx][0]
            self.s2d_bytes(in_tensor, mel_len_ptr, in_tensor.contents.size)
            self.run(self.conds_encoder_idx)
            untensor_sync(out_tensor, False, True)
            shape = (1, 32, 1280)
            size = out_tensor.contents.size // 4  # float32
            buf_type = ctypes.c_float * size
            buf = ctypes.cast(
                out_tensor.contents.data, ctypes.POINTER(buf_type)
            ).contents
            self.mel_emb = (
                np.frombuffer(buf, dtype=np.float32)
                .reshape(shape)
                .astype(np.float16)
                .view(np.uint16)
            )

            in_tensor = self.input_tensors[self.speaker_encoder_idx][0]
            self.s2d_bytes(in_tensor, cond_mel_ptr, in_tensor.contents.size)
            self.run(self.speaker_encoder_idx)
        else:
            cond_mel = self.cache_cond_mel
            cond_mel_frame = cond_mel.shape[-1]
            cond_mel_ptr = self.cond_mel_ptr
            pass

        text_tokens_list = self.tokenizer.tokenize(text)
        sentences = self.tokenizer.split_sentences(
            text_tokens_list, max_text_tokens_per_sentence
        )
        if verbose:
            print("text token count:", len(text_tokens_list))
            print("sentences count:", len(sentences))
            print("max_text_tokens_per_sentence:", max_text_tokens_per_sentence)
            print(*sentences, sep="\n")

        sampling_rate = 24000

        wavs = []
        gpt_gen_time = 0
        gpt_forward_time = 0
        bigvgan_time = 0

        for sent in sentences:
            text_tokens = self.tokenizer.convert_tokens_to_ids(sent)
            text_tokens = torch.tensor(
                text_tokens, dtype=torch.int32, device=self.device
            ).unsqueeze(0)

            if verbose:
                print(text_tokens)
                print(
                    f"text_tokens shape: {text_tokens.shape}, text_tokens type: {text_tokens.dtype}"
                )

            m_start_time = time.perf_counter()
            codes = self.inference_speech(
                text_tokens, top_k, top_p, temperature, repetition_penalty
            )
            gpt_gen_time += time.perf_counter() - m_start_time

            code_lens = torch.tensor([codes.shape[-1]])
            if verbose:
                print(codes, type(codes))
                print(f"codes shape: {codes.shape}, codes type: {codes.dtype}")
                print(f"code len: {code_lens}")

            codes, code_lens = self.remove_long_silence(
                codes, silent_token=52, max_consecutive=30
            )

            if verbose:
                print(codes, type(codes))
                print(f"fix codes shape: {codes.shape}, codes type: {codes.dtype}")
                print(f"code len: {code_lens}")

            m_start_time = time.perf_counter()
            latent = self.gpt(
                text_tokens,
                [text_tokens.shape[-1]],
                codes,
                code_lens * self.mel_length_compression,
            )
            gpt_forward_time += time.perf_counter() - m_start_time

            m_start_time = time.perf_counter()
            wav = self.bigvgan(latent)
            bigvgan_time += time.perf_counter() - m_start_time

            wav = wav[0]
            original_wav_length = int(wav.shape[-1] * self.ori_latent_len / 224)
            wav = wav[:, :original_wav_length]

            wav = torch.clamp(32767 * wav, -32767.0, 32767.0)
            if verbose:
                print(f"wav shape: {wav.shape}", "min:", wav.min(), "max:", wav.max())
            wavs.append(wav.cpu())  # to cpu before saving
        end_time = time.perf_counter()

        wav = torch.cat(wavs, dim=1)
        wav_length = wav.shape[-1] / sampling_rate
        print(
            f">> Reference audio length: {cond_mel_frame * 256 / sampling_rate:.2f} seconds"
        )
        print(f">> gpt_gen_time: {gpt_gen_time:.2f} seconds")
        print(f">> gpt_forward_time: {gpt_forward_time:.2f} seconds")
        print(f">> bigvgan_time: {bigvgan_time:.2f} seconds")
        print(f">> Total inference time: {end_time - start_time:.2f} seconds")
        print(f">> Generated audio length: {wav_length:.2f} seconds")
        print(f">> RTF: {(end_time - start_time) / wav_length:.4f}")

        # save audio
        wav = wav.cpu()  # to cpu
        if output_path:
            # 直接保存音频到指定路径中
            if os.path.isfile(output_path):
                os.remove(output_path)
                print(">> remove old wav file:", output_path)
            if os.path.dirname(output_path) != "":
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
            torchaudio.save(output_path, wav.type(torch.int16), sampling_rate)
            print(">> wav file saved to:", output_path)
            return output_path
        else:
            # 返回以符合Gradio的格式要求
            wav_data = wav.type(torch.int16)
            wav_data = wav_data.numpy().T
            return (sampling_rate, wav_data)


if __name__ == "__main__":
    prompt_wav = "tests/sample_prompt.wav"
    text = """《盗梦空间》是由美国华纳兄弟影片公司出品的电影，由克里斯托弗·诺兰执导并编剧，
莱昂纳多·迪卡普里奥、玛丽昂·歌迪亚、约瑟夫·高登-莱维特、艾利奥特·佩吉、汤姆·哈迪等联袂主演，
2010年7月16日在美国上映，2010年9月1日在中国内地上映，2020年8月28日在中国内地重映。
影片剧情游走于梦境与现实之间，被定义为“发生在意识结构内的当代动作科幻片”，
讲述了由莱昂纳多·迪卡普里奥扮演的造梦师，带领特工团队进入他人梦境，从他人的潜意识中盗取机密，并重塑他人梦境的故事。
""".replace("\n", "")

    tts = IndexTTS(cfg_path="checkpoints/config.yaml", model_dir="checkpoints")
    tts.infer(audio_prompt=prompt_wav, text=text, output_path="gen.wav", verbose=True)
