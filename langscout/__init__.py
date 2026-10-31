"""LangScout - multilingual language detection (17 languages)."""

__version__ = "1.0.0"


def __getattr__(name):  # lazy import keeps `import langscout` light
    if name == "LanguageDetector":
        from .detector import LanguageDetector
        return LanguageDetector
    raise AttributeError(name)
