# 👉🏻 IndexTTS-TPU👈🏻

**IndexTTS** is a GPT-style text-to-speech (TTS) model mainly based on XTTS and Tortoise. It is capable of correcting the pronunciation of Chinese characters using pinyin and controlling pauses at any position through punctuation marks. We enhanced multiple modules of the system, including the improvement of speaker condition feature representation, and the integration of BigVGAN2 to optimize audio quality. Trained on tens of thousands of hours of data, our system achieves state-of-the-art performance, outperforming current popular TTS systems such as XTTS, CosyVoice2, Fish-Speech, and F5-TTS.

**IndexTTS-TPU** is an optimized inference implementation of *IndexTTS* designed for SOPHGO BM1684X TPU hardware acceleration.

## Usage Instructions

### Environment Setup

1. Download this repository:

```bash
git clone https://github.com/wlc952/index-tts.git
```

2. Install dependencies:

```bash
cd index-tts
uv sync
source .venv/bin/activate
```

3. Download models:

```bash
cd checkpoints
python -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/indextts_bm1684x_f16_seq256.bmodel
```

4. Run test script:

```bash
python indextts/infer_f16.py
```

4. Run test web-demo:

```bash
python webui.py
```

Open your browser and visit `http://127.0.0.1:7860` to see the demo.

5. Performance

```sh
>> Reference audio length: 1.07 seconds
>> gpt_gen_time: 18.84 seconds
>> gpt_forward_time: 0.37 seconds
>> bigvgan_time: 3.34 seconds
>> Total inference time: 23.05 seconds
>> Generated audio length: 52.69 seconds
>> RTF: 0.4374
```
