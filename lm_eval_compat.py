"""
    Import this before `lm_eval` when running under transformers 5.

    lm-eval 0.4.5 builds its multimodal class at module scope:

        class HFMultimodalLM(HFLM):
            AUTO_MODEL_CLASS = transformers.AutoModelForVision2Seq

    transformers 5 removed that name, so merely importing lm_eval raises
    AttributeError. Qwen3.8-27B is a Qwen3.5-family model whose calibration origin pins
    transformers 5.16.1, so it is the one model here that hits this.

    The alias is restored rather than moving to a newer lm-eval on purpose: the three models in
    MIXFP4_REPORT.md are meant to be compared with each other, and swapping the evaluation
    harness for one of them would break that. The name is only ever used for vision-to-text
    models; every task in this panel is text, so nothing reads through the alias.

        import lm_eval_compat  # noqa: F401  -- must precede `import lm_eval`
        import lm_eval
"""

import transformers

_MAJOR = int(transformers.__version__.split('.')[0])

if _MAJOR >= 5 and not hasattr(transformers, 'AutoModelForVision2Seq'):
    _successor = (getattr(transformers, 'AutoModelForImageTextToText', None)
                  or transformers.AutoModel)
    transformers.AutoModelForVision2Seq = _successor
    print(f'[lm_eval_compat] transformers {transformers.__version__}: '
          f'AutoModelForVision2Seq -> {_successor.__name__}', flush=True)
