import numpy as np
import torch
import torchaudio
from untool import EngineOV
from transformers import LogitsProcessorList, TopPLogitsWarper, TopKLogitsWarper, TemperatureLogitsWarper, RepetitionPenaltyLogitsProcessor
import torch.nn.functional as F
import onnxruntime as ort

# latent = np.load("latent_debug.npz")["latent"]
# ori_ = latent.shape[1]
# latent = np.pad(
#     latent,
#     ((0, 0), (0, 224 - latent.shape[1]), (0, 0)),
#     mode="constant",
#     constant_values=0,
# )

# speaker_embedding = np.load("speaker_embedding.npz")["speaker_embedding"]

# model = EngineOV("checkpoints/bigvgan.bmodel")

# outputs = model([latent, speaker_embedding])[0]
# print(f"outputs shape: {outputs.shape}")
# wav = outputs[0]
# print(f"wav shape: {wav.shape}", "min:", wav.min(), "max:", wav.max())
# print(wav[:10])

# original_wav_length = int(wav.shape[-1] * ori_ / 224)
# wav = wav[:, :original_wav_length]      

# wav = torch.from_numpy(wav)              
# wav = torch.clamp(32767 * wav, -32767.0, 32767.0)

# torchaudio.save("test.wav", wav.type(torch.int16), 24000)




input_states = np.load("checkpoints/emb_pad.npz")["input_states"]
attention_mask = np.load("checkpoints/emb_pad.npz")["attention_mask"]
model = EngineOV("checkpoints/indextts_bm1684x_seq256.bmodel")
print(f"emb shape: {input_states.shape}, mask shape: {attention_mask.shape}")

outputs = model([input_states, attention_mask])[0]
print(f"outputs shape: {outputs.shape}")
print(outputs[0,0,:10])

# model2 = EngineOV("checkpoints/lm_head.bmodel")
# next_token_logits = model2([outputs[:,-1:,:]])[0]
# print(f"next_token_logits shape: {next_token_logits.shape}")
# print(next_token_logits[0,0,:10])

# model3 = EngineOV("checkpoints/greedy_head.bmodel")
# token = model3([next_token_logits])[0]
# print(token)



# import onnxruntime as ort

# ort_session = ort.InferenceSession("checkpoints/gpt2_lite.onnx")
# outputs = ort_session.run(
#     None,
#     {
#         "input_states": emb,
#         "attention_mask": mask,
#     },
# )
# print(outputs[0][0,0,:10])



# next_token_logits = np.load("checkpoints/logits.npz")["logits"][0]

# model = EngineOV("checkpoints/penalty_sample_head.bmodel")

# inputs=[]
# inputs.append(next_token_logits.astype(np.float32))  # Ensure the input is float32
# input_ids = np.ones((1,256), dtype=np.int32)  # Adjust shape as necessary
# input_ids = input_ids * 478
# input_ids[0,0] = 8192
# top_p = 1.0
# temperature = 1.0
# penalty = 10.0
# inputs.append(input_ids.astype(np.int32))  # Ensure the input is int32
# inputs.append(np.array([top_p], dtype=np.float32))  # Ensure the input is float32
# inputs.append(np.array([temperature], dtype=np.float32))  # Ensure the input is float
# inputs.append(np.array([penalty], dtype=np.float32))  # Ensure the input is float32

# # Run the model
# outputs = model(inputs)
# probs = outputs[0][0]
# tokens = outputs[1][0]

# print(tokens)
# print(probs)
# probs = probs.astype(np.float64)
# probs = probs / probs.sum()

# # 2. 根据 probs[p] 的权重，在 [0..N-1] 中选一个下标
# idx = np.random.choice(probs.shape[0], p=probs)

# # 3. 用该下标到 tokens 中取出真实的 token id
# next_token = int(tokens[idx])

# print("sampled token:", next_token)





# # Create proper input_ids tensor
# next_token_logits = torch.from_numpy(next_token_logits)
# print(f"Next token logits shape: {next_token_logits.shape}")
# input_ids = torch.tensor([[8192,478]])  # Assuming a single token input, adjust as necessary
# print(input_ids)

# top_p = 0.8
# top_k = 1
# repetition_penalty = 10.0


# # Initialize logits processors
# logits_processor = LogitsProcessorList()
# if repetition_penalty != 1.0:
#     logits_processor.append(
#         RepetitionPenaltyLogitsProcessor(
#             penalty=repetition_penalty
#         )
#     )

# if top_k > 0:
#     logits_processor.append(TopKLogitsWarper(top_k))
# # if top_p < 1.0:
# #     logits_processor.append(TopPLogitsWarper(top_p))

# # Process logits
# next_token_scores = logits_processor(input_ids, next_token_logits)

# # Sample next token
# probs = F.softmax(next_token_scores, dim=-1)
# next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)

# print(f"Next token: {next_tokens}")

