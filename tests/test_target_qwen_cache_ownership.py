from types import SimpleNamespace

from mlx_lm.models import cache as cache_mod

from dflash_mlx.engine.target_qwen_gdn import (
    DFlashTargetKVCache,
    DFlashTargetRotatingKVCache,
    QwenGdnTargetOps,
    _install_full_attention_gqa_hook,
)


def _target_with_full_attention_layer():
    text_model = SimpleNamespace(
        embed_tokens=object(),
        layers=[SimpleNamespace(is_linear=False)],
    )
    return SimpleNamespace(language_model=SimpleNamespace(model=text_model))


def test_full_attention_hook_leaves_stock_cache_on_original_path():
    class FakeAttention:
        def __call__(self, x, mask=None, cache=None):
            return "stock"

    attention = FakeAttention()
    _install_full_attention_gqa_hook(attention)

    stock_cache = cache_mod.KVCache()
    stock_cache.offset = 2048

    assert attention(object(), cache=stock_cache) == "stock"


def test_make_cache_marks_unquantized_full_attention_cache_as_dflash_owned():
    caches = QwenGdnTargetOps().make_cache(
        _target_with_full_attention_layer(),
        enable_speculative_linear_cache=True,
        quantize_kv_cache=False,
        target_fa_window=0,
    )

    assert len(caches) == 1
    assert isinstance(caches[0], DFlashTargetKVCache)
    assert isinstance(caches[0], cache_mod.KVCache)


def test_make_cache_marks_rotating_full_attention_cache_as_dflash_owned():
    caches = QwenGdnTargetOps().make_cache(
        _target_with_full_attention_layer(),
        enable_speculative_linear_cache=True,
        quantize_kv_cache=False,
        target_fa_window=128,
    )

    assert len(caches) == 1
    assert isinstance(caches[0], DFlashTargetRotatingKVCache)
    assert isinstance(caches[0], cache_mod.RotatingKVCache)


def test_make_cache_keeps_quantized_full_attention_cache_stock():
    caches = QwenGdnTargetOps().make_cache(
        _target_with_full_attention_layer(),
        enable_speculative_linear_cache=True,
        quantize_kv_cache=True,
        target_fa_window=0,
    )

    assert len(caches) == 1
    assert type(caches[0]) is cache_mod.QuantizedKVCache
