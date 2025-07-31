import ctypes
import numpy as np
from untool import (
    unruntime_init,
    unruntime_run,
    unruntime_free,
    unruntime_set_net_stage,
    unruntime_set_input_s2d,
    unruntime_get_io_count,
    unruntime_get_io_tensor,
    convert_to_python,
    type_map)
import torch
import torchaudio
import time
import torch.nn.functional as F
from indextts.utils.feature_extractors import MelSpectrogramFeatures
from tpu_perf.infer import SGInfer
import os
import sophon.sail as sail
import onnxruntime as ort

class engineSAIL:
    def __init__(self, model_path, device_id=0) :
        self.model_path = model_path
        self.device_id = device_id
        try:
            self.model = sail.Engine(model_path, device_id, sail.IOMode.SYSIO)
        except Exception as e:
            print("load model error; please check model path and device status;")
            print(">>>> model_path: ",model_path)
            print(">>>> device_id: ",device_id)
            print(">>>> sail.Engine error: ",e)
            raise e
        sail.set_print_flag(False)
        self.graph_name = self.model.get_graph_names()
        self.input_name = []
        self.output_name = []
        for name in self.graph_name:
            self.input_name.append(self.model.get_input_names(name))
            self.output_name.append(self.model.get_output_names(name))

    def __str__(self):
        return "EngineOV: model_path={}, device_id={}".format(self.model_path,self.device_id)
    
    def __call__(self, args, net_name=None, net_num=0):
        if isinstance(args, list):
            values = args
        elif isinstance(args, dict):
            values = list(args.values())
        else:
            raise TypeError("args is not list or dict")
        args = {}
        if net_name is not None:
            graph_name = net_name
            input_name = self.model.get_input_names(net_name)
            output_name = self.model.get_output_names(net_name)
        elif net_num is not None:
            graph_name = self.graph_name[net_num]
            input_name = self.input_name[net_num]
            output_name = self.output_name[net_num]
        else:
            input_name = self.input_name[0]
            output_name = self.output_name[0]

        for i in range(len(values)):
            args[input_name[i]] = values[i]

        output = self.model.process(graph_name, args)
        res = []
        for name in output_name:
            res.append(output[name])
        return res

class engineSGInfer:
    
    def __init__(self, model_path="", batch=1,device_id=0) :
        # 如果环境变量中没有设置device_id，则使用默认值
        if "DEVICE_ID" in os.environ:
            device_id = int(os.environ["DEVICE_ID"])
        self.model = SGInfer(model_path , batch=batch, devices=[device_id])
        
    def __call__(self, args):
        if isinstance(args, list):
            values = args
        elif isinstance(args, dict):
            values = list(args.values())
        else:
            raise TypeError("args is not list or dict")
        task_id = self.model.put(*values)
        task_id, results, valid = self.model.get()
        return results
    
class EngineOV:
    def __init__(self, model_path="", device_id=0):
        self.model_path = model_path
        self.device_id = device_id
        self.runtime = unruntime_init(model_path, device_id)

    def reset_net_stage(self, net_idx, stage_idx):
        unruntime_set_net_stage(self.runtime, net_idx, stage_idx)
    
    def __str__(self):
        return "EngineOV: model_path={}, device_id={}".format(self.model_path, self.device_id)
    
    def __call__(self, args, bf16=True, to_host=True):
        if isinstance(args, list):
            values = args
        elif isinstance(args, dict):
            values = list(args.values())
        else:
            raise TypeError("args is not list or dict")

        for idx, val in enumerate(values):

            if not isinstance(val, np.ndarray):
                val = np.asarray(val)
            
            unruntime_set_input_s2d(self.runtime, idx, val.ctypes.data_as(ctypes.c_void_p), val.nbytes)
        
        unruntime_run(self.runtime, to_host)

        out_tensor_ptrs = []
        out_tensor_num = unruntime_get_io_count(self.runtime, b'o')
        for i in range(out_tensor_num):
            out_tensor_ptr = unruntime_get_io_tensor(self.runtime, b'o', i)
            out_tensor_ptrs.append(out_tensor_ptr)

        results = []
        for out_tensor_ptr in out_tensor_ptrs:
            data_ptr = out_tensor_ptr.contents
            dims = int(data_ptr.dims)
            shape = tuple(data_ptr.shape[i] for i in range(dims))
            out_dtype = type_map.get(data_ptr.dtype, np.float32)
            results.append(convert_to_python(data_ptr.data, shape, out_dtype, bf16=bf16))
        return results
    
    def close(self):
        if self.runtime:
            unruntime_free(self.runtime)
            self.runtime = None
    
    def __del__(self):
        self.close()




audio_prompt = "tests/sample_prompt.wav"

audio, sr = torchaudio.load(audio_prompt)
audio = torch.mean(audio, dim=0, keepdim=True)
if audio.shape[0] > 1:
    audio = audio[0].unsqueeze(0)
audio = torchaudio.transforms.Resample(sr, 24000)(audio)
cond_mel = MelSpectrogramFeatures()(audio)

max_mel_tokens = 300
cond_mel = cond_mel[..., :max_mel_tokens]  # 截断多余部分
if cond_mel.shape[-1] < max_mel_tokens:
    pad_len = max_mel_tokens - cond_mel.shape[-1]
    cond_mel = F.pad(cond_mel, (0, pad_len), value=0.0)  # 用0填充

cond_mel = cond_mel.transpose(1, 2).contiguous().numpy()
cond_mel_len = np.array([300], dtype=np.int32)

model_path = "checkpoints/conds_encoder.bmodel"
model = EngineOV(model_path)
time1 = time.time()
out = model([cond_mel, cond_mel_len], bf16=True, to_host=True)
time2 = time.time()
print("Inference time:", time2 - time1)
print(out)

model_path = "checkpoints/conds_encoder.onnx"
onnx = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
time1 = time.time()
out = onnx.run(None, 
                {
                    onnx.get_inputs()[0].name: cond_mel,
                    onnx.get_inputs()[1].name: np.array([300], dtype=np.int64)
                })[0]
time2 = time.time()
print("Inference time:", time2 - time1)
print(out)
