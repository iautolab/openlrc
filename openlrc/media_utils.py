#  Copyright (C) 2025. Hao Zheng
#  All rights reserved.

"""Media-related utility functions that depend on heavy external libraries.

Functions in this module import packages such as ``ffmpeg``, ``filetype``,
``audioread``, ``torch``, and ``spacy`` inside their bodies.  Keeping them
separate from :mod:`openlrc.utils` ensures that the lightweight translation
path never triggers those imports — critical for Nuitka ``--nofollow-import-to``
builds where the heavy packages are intentionally excluded.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from spacy.language import Language as SpacyLanguage

from openlrc.logger import logger
from openlrc.utils import detect_lang


def extract_audio(path: Path) -> Path:
    """
    Extract audio from video.
    :return: Audio path
    """
    import ffmpeg

    file_type = get_file_type(path)
    if file_type == "audio":
        return path

    probe = ffmpeg.probe(path)
    audio_streams = next((stream for stream in probe["streams"] if stream["codec_type"] == "audio"), None)
    if audio_streams is None:
        raise RuntimeError(f"No audio stream found in {path}")
    sample_rate = audio_streams["sample_rate"]
    logger.info(f"File {path}: Audio sample rate: {sample_rate}")

    audio, err = (
        ffmpeg.input(path)
        .output("pipe:", format="wav", acodec="pcm_s16le", ar=sample_rate, loglevel="quiet")
        .run(capture_stdout=True)
    )

    if err:
        raise RuntimeError(f"ffmpeg error: {err}")

    audio_path = path.with_suffix(".wav")
    with open(audio_path, "wb") as f:
        f.write(audio)

    return audio_path


def get_file_type(path: Path) -> str:
    import filetype

    if path.suffix == ".ts":
        return "video"

    try:
        guess = filetype.guess(path)
        if guess is None:
            raise RuntimeError(f"File {path} is not a valid file.")
        file_type = guess.mime.split("/")[0]
    except (TypeError, AttributeError) as e:
        raise RuntimeError(f"File {path} is not a valid file.") from e

    if file_type not in ["audio", "video"]:
        raise RuntimeError(f"File {path} is not a valid file. Should be audio or video file.")

    return file_type


def get_audio_duration(path: str | Path) -> float:
    import audioread

    with audioread.audio_open(str(path)) as audio:
        return audio.duration


def release_memory(model: Any) -> None:
    try:
        import torch
    except ImportError:
        return

    if isinstance(model, torch.nn.Module):
        torch.cuda.empty_cache()


def get_spacy_lib(lang):
    special_case = {"core_web": ["zh", "en"], "ent_wiki": ["xx"]}

    mid_str = "core_news"
    for k, v in special_case.items():
        if lang in v:
            mid_str = k

    return f"{lang}_{mid_str}_sm"


def spacy_load(lang) -> SpacyLanguage:
    import spacy
    import spacy.cli

    lib_name = get_spacy_lib(lang)
    try:
        nlp = spacy.load(lib_name)
    except (ImportError, OSError):
        logger.warning(f"Spacy model {lib_name} missed, downloading")
        spacy.cli.download(lib_name)  # pyright: ignore[reportPrivateImportUsage]
        nlp = spacy.load(lib_name)

    return nlp


def get_similarity(text1, text2):
    lang1 = detect_lang(text1)
    lang2 = detect_lang(text2)

    if lang1 != lang2:
        raise ValueError(f'language of "{text1}" ({lang1}) is not the same as "{text2}" ({lang2})')

    nlp = spacy_load(lang1)

    doc1 = nlp(text1)
    doc2 = nlp(text2)

    return doc1.similarity(doc2)


def merge_subtitle(video_path, subtitle_path, output_path):
    # check ffmpeg
    try:
        subprocess.check_output(["ffmpeg", "-version"])
    except FileNotFoundError:
        raise RuntimeError("ffmpeg is not installed. Please install ffmpeg first.") from None

    subtitle_abs = str(Path(subtitle_path).absolute())
    style = "FontSize=24,Bold=1"
    subprocess.call(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(video_path),
            "-vf",
            f"subtitles={subtitle_abs}:force_style='{style}'",
            str(output_path),
        ]
    )

    logger.info(f"Subtitled video saved to {output_path}")
