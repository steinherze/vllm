# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import ast
import inspect
import os
import textwrap

import pytest
import torch

from vllm.multimodal.video import EmbeddingGemma2VideoBackend

MODEL = os.environ.get("EG2_MODEL_PATH")
needs_model = pytest.mark.skipif(not MODEL, reason="EG2_MODEL_PATH not set")
TEXT_COS, MM_COS = 0.999, 0.999

LORA_MODEL = "google/embeddinggemma-2"
LORA_REVISION = "914f7f89142e33e77833254d9c9b90c3cef7303b"
LORA_TARGETS = "q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj"


def _cos(a, b):
    return torch.nn.functional.cosine_similarity(
        torch.tensor(a, dtype=torch.float32),
        torch.tensor(b, dtype=torch.float32),
        dim=-1,
    )


@pytest.fixture(scope="module")
def text_lora_reference(tmp_path_factory):
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(LORA_MODEL, revision=LORA_REVISION)
    texts = [
        "task: search result | query: what is vllm",
        "vLLM is a library for language model inference and serving.",
        "Paris is the capital of France.",
        "word " * 520,  # Exercise the sliding-window boundary too.
    ]
    prompts = [{"prompt_token_ids": tokenizer.encode(text)} for text in texts]
    model = AutoModel.from_pretrained(
        LORA_MODEL, revision=LORA_REVISION, dtype=torch.bfloat16
    ).eval()

    def embed():
        embeddings = []
        with torch.inference_mode():
            for prompt in prompts:
                ids = torch.tensor([prompt["prompt_token_ids"]])
                hidden = model(input_ids=ids).last_hidden_state.float()
                embeddings.append(torch.nn.functional.normalize(hidden.mean(1), dim=-1))
        return torch.cat(embeddings).tolist()

    references = [embed()]
    paths: list[str] = []
    config = LoraConfig(
        r=8,
        lora_alpha=8,
        target_modules=rf"language_model\.layers\.\d+\.(self_attn|mlp)\.({LORA_TARGETS})",
        init_lora_weights=False,
    )
    for seed in (42, 43):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            if not paths:
                model = get_peft_model(model, config, adapter_name=str(seed))
            else:
                model.add_adapter(str(seed), config)
            model.set_adapter(str(seed))
        model.eval()
        path = tmp_path_factory.mktemp(f"embedding_gemma2_lora_{seed}")
        model.save_pretrained(
            path, selected_adapters=[str(seed)], save_embedding_layers=False
        )
        paths.append(str(path / str(seed)))
        references.append(embed())
    return texts, prompts, paths, references


def test_text_lora_matches_peft_and_isolates_requests(vllm_runner, text_lora_reference):
    """Nonzero text adapters must match PEFT, including mixed and switched batches."""
    from tests.models.utils import check_embeddings_close
    from vllm.lora.request import LoRARequest

    texts, prompts, paths, references = text_lora_reference
    requests = [
        None,
        LoRARequest("first", 1, paths[0]),
        LoRARequest("second", 2, paths[1]),
    ]
    kwargs = dict(
        runner="pooling",
        revision=LORA_REVISION,
        dtype="bfloat16",
        enforce_eager=True,
        max_model_len=1024,
        max_num_seqs=16,
        max_num_batched_tokens=2048,
        gpu_memory_utilization=0.8,
        limit_mm_per_prompt={"image": 0, "audio": 0, "video": 0},
    )
    with vllm_runner(LORA_MODEL, **kwargs) as vm:
        base = vm.embed(texts)

    # Use the model suite's existing BF16 cosine tolerance (0.999). Different
    # attention/reduction orders accumulate rounding across 24 layers.
    def check(actual, expected):
        tensor = torch.tensor(actual)
        assert tensor.shape == (len(prompts), 768)
        assert torch.isfinite(tensor).all()
        torch.testing.assert_close(tensor.norm(dim=-1), torch.ones(len(prompts)))
        check_embeddings_close(
            embeddings_0_lst=actual,
            embeddings_1_lst=expected,
            name_0="vLLM",
            name_1="reference",
            tol=1 - TEXT_COS,
        )

    with vllm_runner(
        LORA_MODEL, enable_lora=True, max_loras=2, max_lora_rank=8, **kwargs
    ) as vm:
        separate = [vm.embed(prompts, lora_request=request) for request in requests]
        check(separate[0], base)
        check(vm.embed(texts, lora_request=requests[1]), separate[1])
        for actual, expected in zip(separate, references):
            check(actual, expected)
        for left, right in ((0, 1), (0, 2), (1, 2)):
            assert not torch.allclose(
                torch.tensor(separate[left]), torch.tensor(separate[right])
            )

        mixed = vm.embed(
            prompts * 3,
            lora_request=[request for request in requests for _ in prompts],
        )
        for index, expected in enumerate(separate):
            check(mixed[index * len(prompts) : (index + 1) * len(prompts)], expected)
        for index in (2, 1, 0, 1):
            check(vm.embed(prompts, lora_request=requests[index]), separate[index])


def test_text_lora_rejects_partial_unsupported_adapters(
    vllm_runner, text_lora_reference, tmp_path
):
    """An adapter with a valid text delta and an unsupported delta must fail whole."""
    import json
    from pathlib import Path

    from safetensors.torch import load_file, save_file

    from vllm.lora.request import LoRARequest

    _, _, paths, _ = text_lora_reference
    source = Path(paths[0])
    tensors = load_file(source / "adapter_model.safetensors")
    config = json.loads((source / "adapter_config.json").read_text())
    unsupported = [
        "vision_tower.encoder.layers.0.self_attn.q_proj",
        "audio_tower.layers.0.self_attn.q_proj",
        "embed_vision.embedding_projection",
        "embed_audio.embedding_projection",
        "language_model.layers.0.ple_block.per_layer_input_gate",
        "language_model.ple.per_layer_model_projection",
        "language_model.embedding_projection",
        "language_model.embed_tokens",
    ]
    with vllm_runner(
        LORA_MODEL,
        revision=LORA_REVISION,
        runner="pooling",
        enable_lora=True,
        max_lora_rank=8,
        enforce_eager=True,
        max_model_len=1024,
        max_num_seqs=16,
        max_num_batched_tokens=2048,
        gpu_memory_utilization=0.8,
        limit_mm_per_prompt={"image": 0, "audio": 0, "video": 0},
    ) as vm:
        for index, module in enumerate(unsupported, start=1):
            path = tmp_path / str(index)
            path.mkdir()
            (path / "adapter_config.json").write_text(json.dumps(config))
            # Shapes are immaterial: validation must reject these before wrapping.
            extra = {
                f"base_model.model.{module}.lora_{side}.weight": torch.ones(8, 8)
                for side in ("A", "B")
            }
            save_file(tensors | extra, path / "adapter_model.safetensors")
            with pytest.raises(Exception, match=module):
                vm.llm.llm_engine.add_lora(LoRARequest(str(index), index, str(path)))
        for index, module in enumerate(("embedding_projection", "classifier"), start=9):
            path = tmp_path / str(index)
            path.mkdir()
            (path / "adapter_config.json").write_text(
                json.dumps(config | {"modules_to_save": [module]})
            )
            extra = {f"base_model.model.{module}.weight": torch.ones(8, 8)}
            save_file(tensors | extra, path / "adapter_model.safetensors")
            with pytest.raises(Exception, match=module):
                vm.llm.llm_engine.add_lora(LoRARequest(str(index), index, str(path)))
        assert not vm.llm.llm_engine.list_loras()


def _ensure_synthetic_video(
    path: str, duration: float, fps: float, width: int, height: int
) -> str:
    if os.path.exists(path):
        os.remove(path)
    import cv2
    import numpy as np

    fourcc_fn = getattr(cv2, "VideoWriter_fourcc", cv2.VideoWriter.fourcc)
    fourcc = fourcc_fn(*"mp4v")
    num_frames = int(duration * fps)
    writer = cv2.VideoWriter(path, fourcc, fps, (width, height))
    y_coords, x_coords = np.mgrid[0:height, 0:width]
    for i in range(num_frames):
        # Spatially varying color pattern that evolves dynamically over frames
        frame: np.ndarray = np.zeros((height, width, 3), dtype=np.uint8)
        frame[..., 0] = ((x_coords * 255 // max(1, width)) + i * 17) % 256
        frame[..., 1] = ((y_coords * 255 // max(1, height)) + i * 29) % 256
        frame[..., 2] = (x_coords + y_coords + i * 13) % 256
        writer.write(frame)
    writer.release()
    return path


@pytest.fixture(scope="module")
def st_model():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(
        MODEL, device="cuda", model_kwargs={"torch_dtype": torch.bfloat16}
    )


@needs_model
def test_structure(vllm_runner, monkeypatch):
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    with vllm_runner(MODEL, runner="pooling", max_model_len=8192) as vm:

        def probe(model):
            lm = model.language_model
            win = [layer.self_attn.attn.impl.sliding_window for layer in lm.layers]
            return win, model.config.text_config.sliding_window

        win, cfg_w = vm.apply_model(probe)[0]
        assert (cfg_w, cfg_w) in win  # (512, 512) == W-1 with W = 513


@needs_model
@pytest.mark.parametrize(
    "prompt",
    [
        "short query",
        "task: search result | query: what is vllm",
        "word " * 700,  # > 512: crosses sliding window
        "word " * 1500,  # > 1024
    ],
)
def test_text_parity(vllm_runner, st_model, prompt):
    ref = st_model.encode([prompt], normalize_embeddings=True)
    with vllm_runner(MODEL, runner="pooling", max_model_len=8192) as vm:
        out = vm.embed([prompt])
    sim = _cos(out, ref).min().item()
    assert sim >= TEXT_COS, (
        f"Text cosine {sim} < {TEXT_COS} for prompt len {len(prompt)}"
    )


@needs_model
def test_mm_recipe(vllm_runner, st_model):
    from PIL import Image

    img = Image.new("RGB", (224, 224), color=(255, 0, 0))
    prompt = "Describe this image: <|image|>"

    # SentenceTransformer multimodal encode
    ref = st_model.encode([{"image": img, "text": prompt}], normalize_embeddings=True)
    with vllm_runner(MODEL, runner="pooling", max_model_len=8192) as vm:
        out = vm.embed([{"prompt": prompt, "multi_modal_data": {"image": img}}])
    sim = _cos(out, ref).min().item()
    assert sim >= MM_COS, f"Image cosine {sim} < {MM_COS}"


@needs_model
def test_natural_image_parity(vllm_runner, st_model):
    from PIL import Image

    img_path = "/projects/gemma4-vllm/images/cat.jpg"
    if not os.path.exists(img_path):
        pytest.skip(f"Test asset {img_path} not found")
    img = Image.open(img_path)
    prompt = "What is this image? <|image|>"
    ref = st_model.encode([{"image": img, "text": prompt}], normalize_embeddings=True)
    with vllm_runner(MODEL, runner="pooling", max_model_len=8192) as vm:
        out = vm.embed([{"prompt": prompt, "multi_modal_data": {"image": img}}])
    sim = _cos(out, ref).min().item()
    assert sim >= MM_COS, f"Natural image cosine {sim} < {MM_COS}"


@needs_model
def test_audio_parity(vllm_runner, st_model):
    import torchaudio

    audio_path = "/projects/gemma4-vllm/audio/tellMeAboutTheSun.wav"
    if not os.path.exists(audio_path):
        pytest.skip(f"Test asset {audio_path} not found")
    wav, sr = torchaudio.load(audio_path)
    mono_wav = wav.mean(dim=0).numpy()
    prompt = "Describe this audio: <|audio|>"
    ref = st_model.encode(
        [{"audio": audio_path, "text": prompt}], normalize_embeddings=True
    )
    with vllm_runner(MODEL, runner="pooling", max_model_len=8192) as vm:
        out = vm.embed(
            [{"prompt": prompt, "multi_modal_data": {"audio": (mono_wav, sr)}}]
        )
    sim = _cos(out, ref).min().item()
    assert sim >= MM_COS, f"Audio cosine {sim} < {MM_COS}"


@needs_model
def test_mixed_parity(vllm_runner, st_model):
    import torchaudio
    from PIL import Image

    img_path = "/projects/gemma4-vllm/images/cat.jpg"
    audio_path = "/projects/gemma4-vllm/audio/tellMeAboutTheSun.wav"
    if not os.path.exists(img_path) or not os.path.exists(audio_path):
        pytest.skip("Test assets for mixed parity not found")
    img = Image.open(img_path)
    wav, sr = torchaudio.load(audio_path)
    mono_wav = wav.mean(dim=0).numpy()
    prompt = "Describe image <|image|> and audio <|audio|>"
    ref = st_model.encode(
        [{"image": img, "audio": audio_path, "text": prompt}],
        normalize_embeddings=True,
    )
    with vllm_runner(MODEL, runner="pooling", max_model_len=8192) as vm:
        out = vm.embed(
            [
                {
                    "prompt": prompt,
                    "multi_modal_data": {
                        "image": img,
                        "audio": (mono_wav, sr),
                    },
                }
            ]
        )
    sim = _cos(out, ref).min().item()
    assert sim >= MM_COS, f"Mixed cosine {sim} < {MM_COS}"


@needs_model
def test_video_parity_uniform_overflow(vllm_runner, st_model):
    from transformers.video_utils import load_video

    video_path = _ensure_synthetic_video(
        "/tmp/test_40s.mp4", duration=40.0, fps=10.0, width=160, height=120
    )
    frames, meta = load_video(video_path, backend="opencv")
    prompt = "Describe this video: <|video|>"
    ref = st_model.encode(
        [{"video": video_path, "text": prompt}], normalize_embeddings=True
    )
    with vllm_runner(MODEL, runner="pooling", max_model_len=8192) as vm:
        out = vm.embed(
            [{"prompt": prompt, "multi_modal_data": {"video": (frames, meta)}}]
        )
    sim = _cos(out, ref).min().item()
    assert sim >= MM_COS, f"40s video cosine {sim} < {MM_COS}"


@needs_model
def test_video_parity_portrait(vllm_runner, st_model):
    from transformers.video_utils import load_video

    video_path = _ensure_synthetic_video(
        "/tmp/test_portrait.mp4", duration=3.0, fps=10.0, width=120, height=160
    )
    frames, meta = load_video(video_path, backend="opencv")
    prompt = "Describe this video: <|video|>"
    ref = st_model.encode(
        [{"video": video_path, "text": prompt}], normalize_embeddings=True
    )
    with vllm_runner(MODEL, runner="pooling", max_model_len=8192) as vm:
        out = vm.embed(
            [{"prompt": prompt, "multi_modal_data": {"video": (frames, meta)}}]
        )
    sim = _cos(out, ref).min().item()
    assert sim >= MM_COS, f"Portrait video cosine {sim} < {MM_COS}"


@needs_model
def test_tower_skip(vllm_runner, monkeypatch):
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

    def nparams(m):
        return sum(p.numel() for p in m.parameters()) / 1e6

    with vllm_runner(
        MODEL,
        runner="pooling",
        max_model_len=2048,
        limit_mm_per_prompt={"image": 0, "audio": 0, "video": 0},
    ) as vm:
        p_text = vm.apply_model(nparams)[0]
        assert abs(p_text - 271.00) <= 0.05, (
            f"Expected 271.00M text params, got {p_text}M"
        )

    with vllm_runner(MODEL, runner="pooling", max_model_len=2048) as vm:
        p_full = vm.apply_model(nparams)[0]
        assert abs(p_full - 744.37) <= 0.05, (
            f"Expected 744.37M full params, got {p_full}M"
        )


def test_encoder_contract():
    from vllm.model_executor.models.embedding_gemma2 import EmbeddingGemma2Model
    from vllm.model_executor.models.gemma4_mm import (
        Gemma4ForConditionalGeneration as G,
    )

    borrowed_methods = [
        "_parse_and_validate_multimodal_inputs",
        "_parse_and_validate_image_input",
        "_parse_and_validate_video_input",
        "_parse_and_validate_audio_input",
        "_process_image_input",
        "_process_video_input",
        "_process_audio_input",
        "embed_multimodal",
        "_encoder_chunk",
    ]

    for name in borrowed_methods:
        # Check method identity (L-F6 + T-F4)
        assert getattr(EmbeddingGemma2Model, name) is getattr(G, name)

        # AST checks: verify no zero-arg super() calls in borrowed methods
        func = getattr(EmbeddingGemma2Model, name)
        src = textwrap.dedent(inspect.getsource(func))
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "super"
            ):
                assert len(node.args) > 0, (
                    f"Zero-arg super() call found in borrowed method {name}"
                )

    # AST check: all self.<attr> non-callable accesses are assigned in __init__
    init_src = inspect.getsource(EmbeddingGemma2Model.__init__)
    for name in borrowed_methods:
        func = getattr(EmbeddingGemma2Model, name)
        src = textwrap.dedent(inspect.getsource(func))
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name)
                and node.value.id == "self"
            ):
                attr = node.attr
                assert hasattr(EmbeddingGemma2Model, attr) or attr in init_src, (
                    f"Attribute {attr} accessed on self in {name} is neither "
                    f"defined on class nor in __init__"
                )


@needs_model
def test_instance_attribute_access(vllm_runner):
    """Check self.<attr> accesses on a real model instance."""
    from vllm.model_executor.models.embedding_gemma2 import EmbeddingGemma2Model

    borrowed_methods = [
        "_parse_and_validate_multimodal_inputs",
        "_parse_and_validate_image_input",
        "_parse_and_validate_video_input",
        "_parse_and_validate_audio_input",
        "_process_image_input",
        "_process_video_input",
        "_process_audio_input",
        "embed_multimodal",
        "_encoder_chunk",
    ]
    attrs_accessed = set()
    for name in borrowed_methods:
        func = getattr(EmbeddingGemma2Model, name)
        src = textwrap.dedent(inspect.getsource(func))
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name)
                and node.value.id == "self"
            ):
                attrs_accessed.add(node.attr)

    def check_attrs(model):
        for attr in attrs_accessed:
            assert hasattr(model, attr), (
                f"Real instance of EmbeddingGemma2Model is missing attribute '{attr}' "
                f"accessed by borrowed method"
            )
        return True

    with vllm_runner(MODEL, runner="pooling", max_model_len=2048) as vm:
        res = vm.apply_model(check_attrs)
        assert res[0] is True


@needs_model
def test_video_loader_real_video(vllm_runner, st_model):
    """Decode a real MP4 file through EmbeddingGemma2VideoBackend.load_bytes

    and VideoMediaIO / MediaConnector with default num_frames=32.
    """
    from vllm.multimodal.media import MediaConnector
    from vllm.multimodal.media.video import VideoMediaIO
    from vllm.multimodal.video import VIDEO_LOADER_REGISTRY

    video_path = "/projects/gemma4-vllm/videos/test_intel.mp4"
    backend_cls = VIDEO_LOADER_REGISTRY.name2class["embedding_gemma2"]
    assert issubclass(backend_cls, EmbeddingGemma2VideoBackend)
    with open(video_path, "rb") as f:
        data = f.read()

    # Direct backend load
    frames, meta = backend_cls.load_bytes(data)

    # VideoMediaIO with default num_frames=32 (VideoMediaIO passes num_frames=32)
    video_io = VideoMediaIO(image_io=None, video_backend="embedding_gemma2")
    loaded = video_io.load_bytes(data)
    frames_io, meta_io = loaded.media
    assert len(frames_io) == len(frames)
    assert meta_io.get("do_sample_frames") is False

    # MediaConnector with video_processor="EmbeddingGemma2VideoProcessor"
    connector = MediaConnector(allowed_local_media_path="/projects/gemma4-vllm/videos")
    fetched = connector.fetch_video(
        f"file://{video_path}",
        video_processor="EmbeddingGemma2VideoProcessor",
    )
    frames_conn, meta_conn = fetched.media
    assert len(frames_conn) == 32
    assert meta_conn.get("do_sample_frames") is False

    prompt = "Describe this video: <|video|>"
    ref = st_model.encode(
        [{"video": video_path, "text": prompt}], normalize_embeddings=True
    )
    with vllm_runner(MODEL, runner="pooling", max_model_len=8192) as vm:
        out = vm.embed(
            [
                {
                    "prompt": prompt,
                    "multi_modal_data": {"video": (frames_conn, meta_conn)},
                }
            ]
        )
    sim = _cos(out, ref).min().item()
    assert sim >= MM_COS, f"Real video loader cosine {sim} < {MM_COS}"


@needs_model
def test_video_loader_duplicate_expansion_e2e(vllm_runner, st_model):
    """End-to-end duplicate expansion test with low-fps clip."""
    import numpy as np

    from vllm.multimodal.video import VIDEO_LOADER_REGISTRY

    backend_cls = VIDEO_LOADER_REGISTRY.name2class["embedding_gemma2"]
    assert issubclass(backend_cls, EmbeddingGemma2VideoBackend)
    # 4s clip at 0.5 fps = 2 unique frames
    lowfps_path = "/tmp/test_lowfps_05.mp4"
    _ensure_synthetic_video(lowfps_path, duration=4.0, fps=0.5, width=64, height=48)
    with open(lowfps_path, "rb") as f:
        data = f.read()
    # Default fps=1.0 -> 4 frames expanding 2 unique frames to [0, 0, 1, 1]
    frames, meta = backend_cls.load_bytes(data)
    assert frames.shape[0] == 4
    # Compare returned frames directly, not only indices
    assert np.array_equal(frames[0], frames[1])
    assert not np.array_equal(frames[0], frames[2])
    assert np.array_equal(frames[2], frames[3])

    prompt = "Describe this video: <|video|>"
    with vllm_runner(MODEL, runner="pooling", max_model_len=8192) as vm:
        out = vm.embed(
            [{"prompt": prompt, "multi_modal_data": {"video": (frames, meta)}}]
        )
    ref = st_model.encode(
        [{"video": lowfps_path, "text": prompt}], normalize_embeddings=True
    )
    sim = _cos(out, ref).min().item()
    assert sim >= MM_COS, f"Duplicate expansion cosine {sim} < {MM_COS}"
    assert len(out) == 1
    assert len(out[0]) == 768


def test_video_loader_indices():
    from transformers.models.embedding_gemma2.video_processing_embedding_gemma2 import (
        EmbeddingGemma2VideoProcessor,
    )
    from transformers.video_utils import VideoMetadata

    from vllm.multimodal.video import (
        VIDEO_LOADER_REGISTRY,
        VideoSourceMetadata,
        VideoTargetMetadata,
    )

    backend_cls = VIDEO_LOADER_REGISTRY.name2class["embedding_gemma2"]
    assert issubclass(backend_cls, EmbeddingGemma2VideoBackend)
    hf_proc = EmbeddingGemma2VideoProcessor()

    # Case 1: clip shorter than 32s (e.g. 5s @ 24fps = 120 frames)
    src_5s = VideoSourceMetadata(total_frames_num=120, original_fps=24.0, duration=5.0)
    tgt = VideoTargetMetadata(num_frames=-1, fps=1.0, max_duration=300)
    vllm_indices = backend_cls.compute_frames_index_to_sample(src_5s, tgt)
    hf_meta_5s = VideoMetadata(fps=24.0, duration=5.0, total_num_frames=120)
    hf_indices_5s = hf_proc.sample_frames(
        hf_meta_5s, fps=1.0, max_frames=32, overflow_strategy="uniform"
    ).tolist()
    assert vllm_indices == hf_indices_5s

    # Case 2: clip shorter than 1s (e.g. 0.5s @ 24fps = 12 frames)
    src_sub1s = VideoSourceMetadata(
        total_frames_num=12, original_fps=24.0, duration=0.5
    )
    vllm_sub1s = backend_cls.compute_frames_index_to_sample(src_sub1s, tgt)
    hf_meta_sub1s = VideoMetadata(fps=24.0, duration=0.5, total_num_frames=12)
    hf_indices_sub1s = hf_proc.sample_frames(
        hf_meta_sub1s, fps=1.0, max_frames=32, overflow_strategy="uniform"
    ).tolist()
    assert vllm_sub1s == hf_indices_sub1s

    # Case 3: clip longer than 32s (e.g. 40s @ 10fps = 400 frames)
    # -> uniform overflow cap 32 frames
    src_40s = VideoSourceMetadata(
        total_frames_num=400, original_fps=10.0, duration=40.0
    )
    vllm_40s = backend_cls.compute_frames_index_to_sample(src_40s, tgt)
    hf_meta_40s = VideoMetadata(fps=10.0, duration=40.0, total_num_frames=400)
    hf_indices_40s = hf_proc.sample_frames(
        hf_meta_40s, fps=1.0, max_frames=32, overflow_strategy="uniform"
    ).tolist()
    assert vllm_40s == hf_indices_40s
    assert len(vllm_40s) == 32

    # Case 4: default target fps (target.fps <= 0 defaults to 1.0)
    tgt_default_fps = VideoTargetMetadata(num_frames=-1, fps=-1.0, max_duration=300)
    vllm_def = backend_cls.compute_frames_index_to_sample(src_5s, tgt_default_fps)
    assert vllm_def == hf_indices_5s

    # Case 5 (NEW-2): 60 fps / 300 s (18000 frames) vs HF sample_frames
    src_300s = VideoSourceMetadata(
        total_frames_num=18000, original_fps=60.0, duration=300.0
    )
    v_300s = backend_cls.compute_frames_index_to_sample(
        src_300s, VideoTargetMetadata(num_frames=-1, fps=60.0, max_duration=300)
    )
    h_300s = hf_proc.sample_frames(
        VideoMetadata(fps=60.0, duration=300.0, total_num_frames=18000),
        fps=60.0,
        max_frames=32,
        overflow_strategy="uniform",
    ).tolist()
    assert v_300s == h_300s, f"60fps/300s mismatch: {v_300s[-3:]} vs {h_300s[-3:]}"

    # Case 6 (NEW-2): 1 fps / 4 h (432000 frames @ 30 native fps) vs HF sample_frames
    src_4h = VideoSourceMetadata(
        total_frames_num=432000, original_fps=30.0, duration=14400.0
    )
    v_4h = backend_cls.compute_frames_index_to_sample(
        src_4h, VideoTargetMetadata(num_frames=-1, fps=1.0, max_duration=300)
    )
    h_4h = hf_proc.sample_frames(
        VideoMetadata(fps=30.0, duration=14400.0, total_num_frames=432000),
        fps=1.0,
        max_frames=32,
        overflow_strategy="uniform",
    ).tolist()
    assert v_4h == h_4h, f"1fps/4h mismatch: {v_4h[-2:]} vs {h_4h[-2:]}"


def test_video_loader_duplicate_expansion():
    from transformers.models.embedding_gemma2.video_processing_embedding_gemma2 import (
        EmbeddingGemma2VideoProcessor,
    )
    from transformers.video_utils import VideoMetadata

    from vllm.multimodal.video import (
        VIDEO_LOADER_REGISTRY,
        VideoSourceMetadata,
        VideoTargetMetadata,
    )

    backend_cls = VIDEO_LOADER_REGISTRY.name2class["embedding_gemma2"]
    assert issubclass(backend_cls, EmbeddingGemma2VideoBackend)
    hf_proc = EmbeddingGemma2VideoProcessor()

    # 4s clip @ 0.5fps = 2 frames total. Requested target fps = 1.0 -> 4 frames.
    # Source has fewer frames than requested sampled frames, requiring duplicates.
    src_lowfps = VideoSourceMetadata(total_frames_num=2, original_fps=0.5, duration=4.0)
    tgt = VideoTargetMetadata(num_frames=-1, fps=1.0, max_duration=300)
    vllm_indices = backend_cls.compute_frames_index_to_sample(src_lowfps, tgt)
    hf_meta = VideoMetadata(fps=0.5, duration=4.0, total_num_frames=2)
    hf_indices = hf_proc.sample_frames(
        hf_meta, fps=1.0, max_frames=32, overflow_strategy="uniform"
    ).tolist()
    assert vllm_indices == hf_indices
    assert len(vllm_indices) == 4
    assert vllm_indices == [0, 0, 1, 1]


def test_video_media_io_embedding_gemma2():
    """Verify VideoMediaIO with default num_frames=32 loads video bytes for

    EmbeddingGemma2 without error.
    """
    import os
    from unittest.mock import MagicMock

    from vllm.multimodal.media import MediaConnector
    from vllm.multimodal.media.video import VideoMediaIO

    video_path = "/projects/gemma4-vllm/videos/test_intel.mp4"
    if not os.path.exists(video_path):
        return

    with open(video_path, "rb") as f:
        data = f.read()

    # VideoMediaIO defaults to num_frames=32
    video_io = VideoMediaIO(image_io=MagicMock(), video_backend="embedding_gemma2")
    loaded = video_io.load_bytes(data)
    frames, meta = loaded.media
    assert len(frames) == 32
    assert meta.get("do_sample_frames") is False

    # MediaConnector path
    connector = MediaConnector(allowed_local_media_path="/projects/gemma4-vllm/videos")
    fetched = connector.fetch_video(
        f"file://{video_path}",
        video_processor="EmbeddingGemma2VideoProcessor",
    )
    frames_conn, meta_conn = fetched.media
    assert len(frames_conn) == 32
    assert meta_conn.get("do_sample_frames") is False


def test_config_capping_and_explicit_limits():
    """Verify EmbeddingGemma2ModelConfig backend selection, default capping,
    and preservation of explicit limits.
    """
    from types import SimpleNamespace
    from unittest.mock import patch

    from vllm.model_executor.models.config import (
        EmbeddingGemma2ModelConfig,
        Gemma4Config,
    )
    from vllm.v1.attention.backends.registry import AttentionBackendEnum

    # (a) When attention backend is None, default to TRITON_ATTN,
    # and default capping: max_model_len > 8192 without explicit limit capped to 8192
    cfg_default = SimpleNamespace(
        attention_config=SimpleNamespace(backend=None),
        model_config=SimpleNamespace(
            max_model_len=262144,
            original_max_model_len=None,
        ),
    )
    with patch.object(Gemma4Config, "verify_and_update_config"):
        EmbeddingGemma2ModelConfig.verify_and_update_config(cfg_default)

    assert cfg_default.attention_config.backend == AttentionBackendEnum.TRITON_ATTN
    assert cfg_default.model_config.max_model_len == 8192

    # (b) Default capping with original_max_model_len == -1
    cfg_default_minus1 = SimpleNamespace(
        attention_config=SimpleNamespace(backend=None),
        model_config=SimpleNamespace(
            max_model_len=131072,
            original_max_model_len=-1,
        ),
    )
    with patch.object(Gemma4Config, "verify_and_update_config"):
        EmbeddingGemma2ModelConfig.verify_and_update_config(cfg_default_minus1)

    assert cfg_default_minus1.model_config.max_model_len == 8192

    # (c) Explicit limit 4096 preserved
    cfg_explicit_4k = SimpleNamespace(
        attention_config=SimpleNamespace(backend=None),
        model_config=SimpleNamespace(
            max_model_len=4096,
            original_max_model_len=4096,
        ),
    )
    with patch.object(Gemma4Config, "verify_and_update_config"):
        EmbeddingGemma2ModelConfig.verify_and_update_config(cfg_explicit_4k)

    assert cfg_explicit_4k.model_config.max_model_len == 4096

    # (d) Explicit limit 16384 preserved and explicit backend preserved
    cfg_explicit_16k = SimpleNamespace(
        attention_config=SimpleNamespace(backend=AttentionBackendEnum.FLASH_ATTN),
        model_config=SimpleNamespace(
            max_model_len=16384,
            original_max_model_len=16384,
        ),
    )
    with patch.object(Gemma4Config, "verify_and_update_config"):
        EmbeddingGemma2ModelConfig.verify_and_update_config(cfg_explicit_16k)

    assert cfg_explicit_16k.attention_config.backend == AttentionBackendEnum.FLASH_ATTN
    assert cfg_explicit_16k.model_config.max_model_len == 16384


def test_sentence_transformer_tokenizer_config_max_seq_length(tmp_path):
    """Verify get_sentence_transformer_tokenizer_config accepts config
    with only max_seq_length.
    """
    import json

    from vllm.transformers_utils.config import (
        get_sentence_transformer_tokenizer_config,
    )

    # 1. Synthetic model directory with only max_seq_length in sentence_bert_config.json
    cfg_file = tmp_path / "sentence_bert_config.json"
    cfg_file.write_text(json.dumps({"max_seq_length": 8192}))
    loaded = get_sentence_transformer_tokenizer_config(str(tmp_path))
    assert loaded is not None
    assert loaded["max_seq_length"] == 8192
