import torch
import soundfile as sf
from dataclasses import asdict
from qwen_tts import Qwen3TTSModel, VoiceClonePromptItem

model = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    dtype=torch.bfloat16,  # M4/M3/M2対応
    attn_implementation="sdpa",  # FlashAttention → SDPAに変更
    device_map="mps",  # MPS強制
    trust_remote_code=True,
    low_cpu_mem_usage=True,
)


def load_voice_model(voice_model_path):
    """保存済みの声モデルをロードする

    Args:
        voice_model_path: 保存済み声モデルファイルのパス (.ptファイル)

    Returns:
        List[VoiceClonePromptItem]: ロードされた声モデル（プロンプトリスト）
    """
    payload = torch.load(voice_model_path, map_location="cpu", weights_only=True)

    if not isinstance(payload, dict) or "items" not in payload:
        raise ValueError("Invalid voice model file format")

    items_raw = payload["items"]
    items = []
    for d in items_raw:
        ref_code = d.get("ref_code", None)
        if ref_code is not None and not torch.is_tensor(ref_code):
            ref_code = torch.tensor(ref_code)

        ref_spk = d.get("ref_spk_embedding", None)
        if ref_spk is None:
            raise ValueError("Missing ref_spk_embedding in voice model")
        if not torch.is_tensor(ref_spk):
            ref_spk = torch.tensor(ref_spk)

        items.append(
            VoiceClonePromptItem(
                ref_code=ref_code,
                ref_spk_embedding=ref_spk,
                x_vector_only_mode=d.get("x_vector_only_mode", False),
                icl_mode=d.get("icl_mode", False),
                ref_text=d.get("ref_text", None),
            )
        )

    print(f"声のモデルをロードしました: {voice_model_path}")
    return items


def text2speech(voice_model, text, out_path):
    """ロード済みの声モデルを使って音声を生成する

    Args:
        voice_model: load_voice_model() でロードした声モデル（プロンプトリスト）
        text: 生成するテキスト
        out_path: 出力音声ファイルのパス
    """
    wavs, sr = model.generate_voice_clone(
        text=text,
        language="Japanese",
        voice_clone_prompt=voice_model,
    )
    sf.write(out_path, wavs[0], sr)
    print(f"音声を生成しました: {out_path}")


def save_voice_model(ref_audio_path, ref_text, output_path):
    """リファレンス音声から声モデルを生成・保存する

    Args:
        ref_audio_path: リファレンス音声ファイルのパス
        ref_text: リファレンス音声のテキスト（必須）
        output_path: 保存先のパス (.ptファイル)
    """
    items = model.create_voice_clone_prompt(
        ref_audio=ref_audio_path,
        ref_text=ref_text,
        x_vector_only_mode=False,  # テキスト情報も含める（品質向上）
    )

    # dataclassをdictに変換して保存
    payload = {
        "items": [asdict(it) for it in items],
    }

    torch.save(payload, output_path)
    print(f"声のモデルを保存しました: {output_path}")


if __name__ == "__main__":
    # save_voice_model(
    #     ref_audio_path="Timeline1.wav",
    #     ref_text="なんか作業の自動化をしたり、そういうことができたりします。で、ショートカット・エイリアスについてはまた後で説明します、詳しく。あとは、今回はあまり触れないんですけど、簡単なPythonのスクリプトとかを書いて、それを実行して自動化したりとか、AIとも密接に統合されています。",
    #     output_path="voice_model.pt",
    # )
    text2speech(
        load_voice_model("voice_model.pt"),
        text="皆さん、こんにちは。今日は音声クローンのデモをお見せします。",
        out_path="output_voice_clone.wav",
    )
