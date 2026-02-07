import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel

model = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    dtype=torch.bfloat16,  # M4/M3/M2対応
    attn_implementation="sdpa",  # FlashAttention → SDPAに変更
    device_map="mps",           # MPS強制
    trust_remote_code=True,
    low_cpu_mem_usage=True
)

ref_audio = "Timeline1.wav"
ref_text  = "なんか作業の自動化をしたり、そういうことができたりします。で、ショートカット・エイリアスについてはまた後で説明します、詳しく。あとは、今回はあまり触れないんですけど、簡単なPythonのスクリプトとかを書いて、それを実行して自動化したりとか、AIとも密接に統合されています。"

wavs, sr = model.generate_voice_clone(
    text="皆さん、こんにちは。今日は音声クローンのデモをお見せします。",
    language="Japanese",
    ref_audio=ref_audio,
    ref_text=ref_text,
)
sf.write("output_voice_clone.wav", wavs[0], sr)