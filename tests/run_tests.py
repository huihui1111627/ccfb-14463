"""Tiny test runner (no third-party dependencies).

Supports the subset of pytest used in this repository: ``test_*`` functions,
``@pytest.mark.parametrize`` and ``pytest.skip``.  Run with::

    python3 tests/run_tests.py [-v] [pattern]
"""

from __future__ import annotations

import importlib
import inspect
import itertools
import os
import sys
import traceback

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


class _Skip(Exception):
    pass


class _PytestShim:
    class mark:
        @staticmethod
        def parametrize(argnames, argvalues):
            names = [a.strip() for a in argnames.split(",")]

            def deco(fn):
                fn._parametrize = (names, list(argvalues))
                return fn

            return deco

    @staticmethod
    def skip(reason=""):
        raise _Skip(reason)

    @staticmethod
    def raises(exc):
        class _CM:
            def __enter__(self):
                return self

            def __exit__(self, et, ev, tb):
                if et is None:
                    raise AssertionError(f"did not raise {exc.__name__}")
                if not issubclass(et, exc):
                    return False
                return True

        return _CM()


def _install_shim():
    import types

    mod = types.ModuleType("pytest")
    mod.mark = _PytestShim.mark
    mod.skip = _PytestShim.skip
    mod.raises = _PytestShim.raises
    sys.modules["pytest"] = mod


def _cases(fn):
    if hasattr(fn, "_parametrize"):
        names, values = fn._parametrize
        for v in values:
            if not isinstance(v, (tuple, list)):
                v = (v,)
            yield tuple(v)
    else:
        yield ()


def main(argv):
    _install_shim()
    verbose = "-v" in argv
    args = [a for a in argv if not a.startswith("-")]
    pattern = args[0] if args else ""
    tests_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, tests_dir)

    passed = failed = skipped = 0
    failures = []
    for fname in sorted(os.listdir(tests_dir)):
        if not (fname.startswith("test_") and fname.endswith(".py")):
            continue
        modname = fname[:-3]
        if pattern and pattern not in modname:
            continue
        mod = importlib.import_module(modname)
        for name in sorted(dir(mod)):
            if not name.startswith("test_"):
                continue
            fn = getattr(mod, name)
            if not callable(fn):
                continue
            for values in _cases(fn):
                label = f"{modname}::{name}" + (f"[{values}]" if values else "")
                try:
                    fn(*values)
                    passed += 1
                    if verbose:
                        print(f"PASS {label}")
                except _Skip as e:
                    skipped += 1
                    if verbose:
                        print(f"SKIP {label}: {e}")
                except Exception as e:  # noqa: BLE001
                    failed += 1
                    failures.append((label, traceback.format_exc()))
                    print(f"FAIL {label}: {e}")
    print(f"\n{passed} passed, {failed} failed, {skipped} skipped")
    if failures:
        print("\n" + "=" * 70)
        for label, tb in failures:
            print(label)
            print(tb)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
