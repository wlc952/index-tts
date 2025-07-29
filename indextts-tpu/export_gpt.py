import os
import re
import time
from subprocess import CalledProcessError
from typing import List

import numpy as np
import sentencepiece as spm
import torch
import torchaudio
from torch.nn.utils.rnn import pad_sequence
from omegaconf import OmegaConf
from tqdm import tqdm
import torch.nn.functional as F

import warnings

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from indextts.BigVGAN.models import BigVGAN as Generator
from indextts.gpt.model import UnifiedVoice
from indextts.utils.checkpoint import load_checkpoint
from indextts.utils.feature_extractors import MelSpectrogramFeatures

from indextts.utils.front import TextNormalizer, TextTokenizer


class IndexTTS:
    def __init__(
        self,
        cfg_path="checkpoints/config.yaml",
        model_dir="checkpoints",
        is_fp16=True,
        device=None,
        use_cuda_kernel=None,
    ):
        """
        Args:
            cfg_path (str): path to the config file.
            model_dir (str): path to the model directory.
            is_fp16 (bool): whether to use fp16.
            device (str): device to use (e.g., 'cuda:0', 'cpu'). If None, it will be set automatically based on the availability of CUDA or MPS.
            use_cuda_kernel (None | bool): whether to use BigVGan custom fused activation CUDA kernel, only for CUDA device.
        """
        if device is not None:
            self.device = device
            self.is_fp16 = False if device == "cpu" else is_fp16
            self.use_cuda_kernel = (
                use_cuda_kernel is not None
                and use_cuda_kernel
                and device.startswith("cuda")
            )
        elif torch.cuda.is_available():
            self.device = "cuda:0"
            self.is_fp16 = is_fp16
            self.use_cuda_kernel = use_cuda_kernel is None or use_cuda_kernel
        else:
            self.device = "cpu"
            self.is_fp16 = False
            self.use_cuda_kernel = False
            print(">> Be patient, it may take a while to run in CPU mode.")

        self.cfg = OmegaConf.load(cfg_path)
        self.model_dir = model_dir
        self.dtype = torch.float16 if self.is_fp16 else None
        self.stop_mel_token = self.cfg.gpt.stop_mel_token

        self.gpt = UnifiedVoice(**self.cfg.gpt)
        self.gpt_path = os.path.join(self.model_dir, self.cfg.gpt_checkpoint)
        load_checkpoint(self.gpt, self.gpt_path)
        self.gpt = self.gpt.to(self.device)
        if self.is_fp16:
            self.gpt.eval().half()
        else:
            self.gpt.eval()
        print(">> GPT weights restored from:", self.gpt_path)
        if self.is_fp16:
            use_deepspeed = False
            self.gpt.post_init_gpt2_config(
                use_deepspeed=use_deepspeed, kv_cache=True, half=True
            )
        else:
            self.gpt.post_init_gpt2_config(
                use_deepspeed=False, kv_cache=False, half=False
            )

        if self.use_cuda_kernel:
            # preload the CUDA kernel for BigVGAN
            try:
                from indextts.BigVGAN.alias_free_activation.cuda import load

                anti_alias_activation_cuda = load.load()
                print(
                    ">> Preload custom CUDA kernel for BigVGAN",
                    anti_alias_activation_cuda,
                )
            except:
                print(
                    ">> Failed to load custom CUDA kernel for BigVGAN. Falling back to torch."
                )
                self.use_cuda_kernel = False
        self.bigvgan = Generator(self.cfg.bigvgan, use_cuda_kernel=self.use_cuda_kernel)
        self.bigvgan_path = os.path.join(self.model_dir, self.cfg.bigvgan_checkpoint)
        vocoder_dict = torch.load(self.bigvgan_path, map_location="cpu")
        self.bigvgan.load_state_dict(vocoder_dict["generator"])
        self.bigvgan = self.bigvgan.to(self.device)
        # remove weight norm on eval mode
        self.bigvgan.remove_weight_norm()
        self.bigvgan.eval()
        print(">> bigvgan weights restored from:", self.bigvgan_path)
        self.bpe_path = os.path.join(self.model_dir, self.cfg.dataset["bpe_model"])
        self.normalizer = TextNormalizer()
        self.normalizer.load()
        print(">> TextNormalizer loaded")
        self.tokenizer = TextTokenizer(self.bpe_path, self.normalizer)
        print(">> bpe model loaded from:", self.bpe_path)
        # 缓存参考音频mel：
        self.cache_audio_prompt = None
        self.cache_cond_mel = None
        # 进度引用显示（可选）
        self.gr_progress = None

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

    def _set_gr_progress(self, value, desc):
        if self.gr_progress is not None:
            self.gr_progress(value, desc=desc)

    def infer(self, audio_prompt, text, output_path, verbose=False):
        print(">> start inference...")
        self._set_gr_progress(0, "start inference...")
        if verbose:
            print(f"origin text:{text}")
            print()
        start_time = time.perf_counter()

        audio, sr = torchaudio.load(audio_prompt)
        audio = torch.mean(audio, dim=0, keepdim=True)
        if audio.shape[0] > 1:
            audio = audio[0].unsqueeze(0)
        audio = torchaudio.transforms.Resample(sr, 24000)(audio)
        if verbose:
            print(f"audio shape: {audio.shape}", "dtype:", audio.dtype)
            print()
        cond_mel = MelSpectrogramFeatures()(audio).to(self.device)

        max_mel_tokens = 300
        cond_mel = cond_mel[..., :max_mel_tokens]  # 截断多余部分
        if cond_mel.shape[-1] < max_mel_tokens:
            pad_len = max_mel_tokens - cond_mel.shape[-1]
            cond_mel = F.pad(cond_mel, (0, pad_len), value=0.0)  # 用0填充

        cond_mel_frame = cond_mel.shape[-1]
        if verbose:
            print(f"cond_mel shape: {cond_mel.shape}", "dtype:", cond_mel.dtype)
            print()

        self.cache_audio_prompt = audio_prompt
        self.cache_cond_mel = cond_mel

        auto_conditioning = cond_mel
        text_tokens_list = self.tokenizer.tokenize(text)
        sentences = self.tokenizer.split_sentences(text_tokens_list)
        if verbose:
            print("text token count:", len(text_tokens_list))
            print("sentences count:", len(sentences))
            print(*sentences, sep="\n")
            print()
        top_p = 0.8
        top_k = 30
        temperature = 1.0
        autoregressive_batch_size = 1
        length_penalty = 0.0
        num_beams = 1
        repetition_penalty = 10.0
        max_mel_tokens = 600
        sampling_rate = 24000
        # lang = "EN"
        # lang = "ZH"
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
                print(f"text_tokens: {text_tokens}")
                print(
                    f"text_tokens shape: {text_tokens.shape}, text_tokens type: {text_tokens.dtype}"
                )
                # debug tokenizer
                # text_token_syms = self.tokenizer.convert_ids_to_tokens(text_tokens[0].tolist())
                # print("text_token_syms is same as sentence tokens", text_token_syms == sent)
                print()

            # text_len = torch.IntTensor([text_tokens.size(1)], device=text_tokens.device)
            # print(text_len)

            m_start_time = time.perf_counter()
            with torch.no_grad():
                with torch.amp.autocast(
                    text_tokens.device.type,
                    enabled=self.dtype is not None,
                    dtype=self.dtype,
                ):
                    codes = self.gpt.inference_speech(
                        auto_conditioning,
                        text_tokens,
                        cond_mel_lengths=torch.tensor(
                            [auto_conditioning.shape[-1]], device=text_tokens.device
                        ),
                        # text_lengths=text_len,
                        do_sample=True,
                        top_p=top_p,
                        top_k=top_k,
                        temperature=temperature,
                        num_return_sequences=autoregressive_batch_size,
                        length_penalty=length_penalty,
                        num_beams=num_beams,
                        repetition_penalty=repetition_penalty,
                        max_generate_length=max_mel_tokens,
                    )
                gpt_gen_time += time.perf_counter() - m_start_time
                # codes = codes[:, :-2]
                code_lens = torch.tensor(
                    [codes.shape[-1]], device=codes.device, dtype=codes.dtype
                )
                if verbose:
                    print(f"codes: {codes}, {type(codes)}")
                    print(f"codes shape: {codes.shape}, codes type: {codes.dtype}")
                    print(f"code len: {code_lens}")
                    print()

                # remove ultra-long silence if exits
                # temporarily fix the long silence bug.
                codes, code_lens = self.remove_long_silence(
                    codes, silent_token=52, max_consecutive=30
                )
                if verbose:
                    print(codes, type(codes))
                    print(f"fix codes shape: {codes.shape}, codes type: {codes.dtype}")
                    print(f"code len: {code_lens}")
                    print()

                m_start_time = time.perf_counter()
                # latent, text_lens_out, code_lens_out = \
                with torch.amp.autocast(
                    text_tokens.device.type,
                    enabled=self.dtype is not None,
                    dtype=self.dtype,
                ):
                    latent = self.gpt(
                        auto_conditioning,
                        text_tokens,
                        torch.tensor(
                            [text_tokens.shape[-1]], device=text_tokens.device
                        ),
                        codes,
                        code_lens * self.gpt.mel_length_compression,
                        cond_mel_lengths=torch.tensor(
                            [auto_conditioning.shape[-1]], device=text_tokens.device
                        ),
                        return_latent=True,
                        clip_inputs=False,
                    )
                    gpt_forward_time += time.perf_counter() - m_start_time

                    m_start_time = time.perf_counter()
                    wav, _ = self.bigvgan(latent, auto_conditioning.transpose(1, 2))
                    bigvgan_time += time.perf_counter() - m_start_time
                    wav = wav.squeeze(1)

                wav = torch.clamp(32767 * wav, -32767.0, 32767.0)
                print(f"wav shape: {wav.shape}", "min:", wav.min(), "max:", wav.max())
                print()
                # wavs.append(wav[:, :-512])
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


tts = IndexTTS(
    cfg_path="checkpoints/config.yaml",
    model_dir="checkpoints",
    is_fp16=False,
    use_cuda_kernel=False,
    device="cpu"
)

SEQ_LENGTH = 256
HIDDEN_SIZE = 1280
NUM_ATTENTION_HEADS = 20
HEAD_DIM = 64
NUM_BLOCKS = 24

dtype = torch.float32
device = tts.device

folder = "./onnx"
if not os.path.exists(folder):
    os.makedirs(folder)

model1 = tts.gpt.inference_model
gpt = model1.transformer
layers = gpt.h


# Convert gpt2 to ONNX format
class Block(torch.nn.Module):
    def __init__(self, layer_id):
        super().__init__()
        self.layer_id = layer_id
        self.layer = layers[layer_id]
        self.ln_1 = self.layer.ln_1
        self.attn = self.layer.attn
        self.ln_2 = self.layer.ln_2
        self.mlp = self.layer.mlp

    def forward(self, hidden_states, attention_mask):
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)  # type: ignore
        hidden_states, past_kv = self.attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            use_cache=True,
        )  # type: ignore
        hidden_states = hidden_states + residual
        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)  # type: ignore
        hidden_states = self.mlp(hidden_states)  # type: ignore
        hidden_states = hidden_states + residual

        present_k, present_v = past_kv
        return hidden_states.float(), present_k.float(), present_v.float()

class BlockCache(torch.nn.Module):
    def __init__(self, layer_id):
        super().__init__()
        self.layer_id = layer_id
        self.layer = layers[layer_id]
        self.ln_1 = self.layer.ln_1
        self.attn = self.layer.attn
        self.ln_2 = self.layer.ln_2
        self.mlp = self.layer.mlp

    def forward(self, hidden_states, attention_mask, past_k, past_v):
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)  # type: ignore

        hidden_states, past_kv = self.attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            layer_past=(past_k, past_v),
            use_cache=True,
        )  # type: ignore
        hidden_states = hidden_states + residual
        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)  # type: ignore
        hidden_states = self.mlp(hidden_states)  # type: ignore
        hidden_states = hidden_states + residual

        present_k, present_v = past_kv
        return hidden_states.float(), present_k.float(), present_v.float()

def convert_block(layer_id):
    model = Block(layer_id)
    hidden_states = torch.randn((1, SEQ_LENGTH, HIDDEN_SIZE)).to(dtype).to(device)
    attention_mask = torch.ones((1, 1, 1, SEQ_LENGTH)).to(dtype).to(device)
    torch.onnx.export(
        model,
        (hidden_states, attention_mask),
        f"{folder}/block_{layer_id}.onnx",
        verbose=False,
        input_names=["input_states", "attention_mask"],
        output_names=["hidden_states", "past_k", "past_v"],
        do_constant_folding=True,
        opset_version=15,
    )

def convert_block_cache(layer_id):
    model = BlockCache(layer_id)
    hidden_states = torch.randn((1, 1, HIDDEN_SIZE)).to(dtype).to(device)
    attention_mask = torch.ones((1, 1, 1, SEQ_LENGTH+1)).to(dtype).to(device)
    past_k = (
        torch.randn((1, SEQ_LENGTH, NUM_ATTENTION_HEADS, HEAD_DIM)).to(dtype).to(device)
    )
    past_v = (
        torch.randn((1, SEQ_LENGTH, NUM_ATTENTION_HEADS, HEAD_DIM)).to(dtype).to(device)
    )

    torch.onnx.export(
        model,
        (hidden_states, attention_mask, past_k, past_v),
        f"{folder}/block_cache_{layer_id}.onnx",
        verbose=False,
        input_names=["input_states", "attention_mask", "past_k", "past_v"],
        output_names=["hidden_states", "present_k", "present_v"],
        do_constant_folding=True,
        opset_version=15,
    )



def convert_ln_f():
    class ln_f(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.ln_f = gpt.ln_f

        def forward(self, hidden_states):
            hidden_states = self.ln_f(hidden_states)
            return hidden_states

    model = ln_f()
    hidden_states = torch.randn((1, 1, HIDDEN_SIZE)).to(dtype).to(device)

    torch.onnx.export(
        model,
        hidden_states,
        f"{folder}/ln_f.onnx",
        verbose=False,
        input_names=["input_states"],
        output_names=["hidden_states"],
        do_constant_folding=True,
        opset_version=15,
    )
    hidden_states2 = torch.randn((1, 224, HIDDEN_SIZE)).to(dtype).to(device)

    torch.onnx.export(
        model,
        hidden_states2,
        f"{folder}/ln_f2.onnx",
        verbose=False,
        input_names=["input_states"],
        output_names=["hidden_states"],
        do_constant_folding=True,
        opset_version=15,
    )


# Convert the inference model to ONNX format
def convert_inference_model_embedding():
    text_inputs = torch.randint(1,100, (1, 1)).to(device)  # 保持为int64类型，只转换设备

    class GPT2InferenceModelEmbedding(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embeddings = model1.embeddings
            self.text_pos_embedding = model1.text_pos_embedding

        def forward(self, text_inputs):
            text_emb = self.embeddings(text_inputs) # [1,1,1280]
            text_emb = text_emb + self.text_pos_embedding(text_emb)
            return text_emb

    torch.onnx.export(
        GPT2InferenceModelEmbedding(),
        text_inputs,
        f"{folder}/inference_model_embedding.onnx",
        verbose=False,
        input_names=["text_inputs"],
        output_names=["hidden_states"],
        do_constant_folding=True,
        opset_version=15,
    )

def convert_lm_head():
    hidden_states = torch.randn((1, 1, HIDDEN_SIZE))
    torch.onnx.export(
        model1.lm_head,
        hidden_states,
        f"{folder}/lm_head.onnx",
        verbose=False,
        input_names=["hidden_states"],
        output_names=["m_logits"],
        do_constant_folding=True,
        opset_version=15,
    )

def convert_greedy_head_text():
    class GreedyHead(torch.nn.Module):

        def __init__(self):
            super().__init__()

        def forward(self, m_logits):
            _, token = torch.topk(m_logits.float(), 1)
            return token

    model = GreedyHead()
    m_logits = torch.randn(1, 8194)

    torch.onnx.export(
        model,
        (m_logits),
        f"{folder}/greedy_head.onnx",
        verbose=False,
        input_names=["m_logits"],
        output_names=["token"],
        do_constant_folding=True,
        opset_version=15,
    )

def convert_conds_encoder():
    class conds_encoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conditioning_encoder = tts.gpt.conditioning_encoder
            self.perceiver_encoder = tts.gpt.perceiver_encoder
            self.conds_mask = torch.tensor(
                [[True] * (149 + 32)],
                dtype=torch.bool,
                device=model1.device,
            )
            self.cond_mel_lengths = torch.tensor([300], device=model1.device)

        def forward(self, speech_conditioning_input):
            speech_conditioning_input, _ = self.conditioning_encoder(speech_conditioning_input,  self.cond_mel_lengths) # type: ignore
            conds = self.perceiver_encoder(speech_conditioning_input, self.conds_mask) # type: ignore
            return conds

    model = conds_encoder()
    speech_conditioning_input = torch.randn((1, 300, 100)).to(dtype).to(device)

    torch.onnx.export(
        model,
        speech_conditioning_input,
        f"{folder}/conds_encoder.onnx",
        verbose=False,
        input_names=["speech_conditioning_input"],
        output_names=["conds"],
        do_constant_folding=True,
        opset_version=15,
    )

def convert_embedding():
    inputs = torch.randint(10, 1000, (1, SEQ_LENGTH)).to(device)  # 保持为int64类型，只转换设备
    class TextEmbedding(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.text_embedding = tts.gpt.text_embedding
            self.text_pos_embedding = tts.gpt.text_pos_embedding

        def forward(self, text_inputs, text_inputs2):
            text_emb = self.text_embedding(text_inputs) + self.text_pos_embedding(text_inputs2)
            return text_emb

    class MelEmbedding(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.mel_embedding = tts.gpt.mel_embedding
            self.mel_pos_embedding = tts.gpt.mel_pos_embedding

        def forward(self, mel_inputs):
            mel_emb = self.mel_embedding(mel_inputs) + self.mel_pos_embedding(mel_inputs)
            return mel_emb
    
    torch.onnx.export(
        TextEmbedding(),
        (inputs, inputs),
        f"{folder}/text_embedding.onnx",
        verbose=False,
        input_names=["text_inputs", "text_inputs2"],
        output_names=["text_emb"],
        do_constant_folding=True,
        opset_version=15,
    )
    torch.onnx.export(
        MelEmbedding(),
        inputs,
        f"{folder}/mel_embedding.onnx",
        verbose=False,
        input_names=["mel_inputs"],
        output_names=["mel_emb"],
        do_constant_folding=True,
        opset_version=15,
    )

def convert_final_norm():
    class FinalNorm(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.final_norm = model1.final_norm
        def forward(self, enc):
            return self.final_norm(enc)
    
    enc = torch.randn((1, SEQ_LENGTH - 32, HIDDEN_SIZE)).to(dtype).to(device)
    torch.onnx.export(
        FinalNorm(),
        enc,
        f"{folder}/final_norm.onnx",
        verbose=False,
        input_names=["hidden_states"],
        output_names=["logits"],
        do_constant_folding=True,
        opset_version=15,
    )


main
for i in range(NUM_BLOCKS):
    convert_block(i)
    convert_block_cache(i)

convert_ln_f()
convert_inference_model_embedding()
convert_lm_head()
convert_greedy_head_text()

convert_conds_encoder()
convert_embedding()
convert_final_norm()

