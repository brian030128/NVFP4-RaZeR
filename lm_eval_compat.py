"""
    Import this before `lm_eval` when running under transformers 5.

    lm-eval 0.4.5 resolves a name at module scope that transformers 5 removed:

        class HFMultimodalLM(HFLM):
            AUTO_MODEL_CLASS = transformers.AutoModelForVision2Seq

    so merely importing lm_eval raises AttributeError. Qwen3.8-27B is the one model in
    MIXFP4_REPORT.md that hits this, because its calibration origin pins transformers 5.16.1.

    Restoring the name rather than moving to a newer lm-eval is deliberate: the three models in
    that report are compared with each other, and swapping the evaluation harness for one of
    them would break the comparison. The alias is only read for vision-to-text models, and every
    task in this panel is text.

    The obvious implementation does not work. Setting the attribute on the module object is
    silently undone, because transformers 5 REPLACES sys.modules['transformers'] with a fresh
    _LazyModule while lm_eval is importing:

        set ok: True  id: 22825783416032
        FAILED: module transformers has no attribute AutoModelForVision2Seq
        module swapped: True  new id: 22818474336288
        attr still in __dict__: False

    So the fallback is installed on the _LazyModule CLASS instead, where it survives any number
    of module swaps, and defers to the original __getattr__ for every other name.
"""

import transformers

_MAJOR = int(transformers.__version__.split('.')[0])
_MISSING = 'AutoModelForVision2Seq'
_SUCCESSOR = 'AutoModelForImageTextToText'


def _install():
    cls = type(transformers)
    original = getattr(cls, '__getattr__', None)
    if original is None or getattr(cls, '_mixfp4_vision2seq_shim', False):
        return False

    def __getattr__(self, name):
        if name == _MISSING:
            try:
                return original(self, name)
            except AttributeError:
                value = original(self, _SUCCESSOR)
                setattr(self, name, value)
                return value
        return original(self, name)

    cls.__getattr__ = __getattr__
    cls._mixfp4_vision2seq_shim = True
    return True


if _MAJOR >= 5 and not hasattr(transformers, _MISSING):
    if _install():
        print(f'[lm_eval_compat] transformers {transformers.__version__}: '
              f'{_MISSING} -> {_SUCCESSOR} (installed on {type(transformers).__name__}, '
              f'which survives the module swap during lm_eval import)', flush=True)
    else:
        print(f'[lm_eval_compat] WARNING: could not install the {_MISSING} fallback; '
              f'importing lm_eval will probably fail', flush=True)
