# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

"""Regression tests for the video generation tool (background music muxing)."""

import tempfile
from pathlib import Path
from unittest import mock

from trae_agent.tools.video_gen_tool import VideoGenTool


def _tmp(suffix: str) -> Path:
    f = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    f.close()
    return Path(f.name)


def _make_tool() -> VideoGenTool:
    return VideoGenTool(model_provider="doubao", api_key="test-key")


def test_parameters_include_background_music():
    names = {p.name for p in _make_tool().get_parameters()}
    assert "background_music" in names
    assert "loop_music" in names
    assert "prompt" in names
    assert "duration" in names
    assert "ratio" in names


def test_mux_audio_command_loop():
    tool = _make_tool()
    video = _tmp(".mp4")
    audio = _tmp(".mp3")
    out = Path(tempfile.gettempdir()) / "out_loop.mp4"

    captured: dict = {}

    def fake_run(cmd, *a, **k):
        captured["cmd"] = cmd
        return mock.Mock(returncode=0, stderr=b"")

    with mock.patch("shutil.which", return_value="/usr/bin/ffmpeg"), mock.patch(
        "subprocess.run", side_effect=fake_run
    ):
        tool._mux_audio(video, audio, out, loop=True)

    cmd = captured["cmd"]
    assert str(video) in cmd
    assert str(audio) in cmd
    assert str(out) in cmd
    assert any("aloop=loop=-1" in str(c) for c in cmd)
    assert "-c:v" in cmd and "copy" in cmd
    assert "-c:a" in cmd and "aac" in cmd


def test_mux_audio_command_no_loop():
    tool = _make_tool()
    video = _tmp(".mp4")
    audio = _tmp(".mp3")
    out = Path(tempfile.gettempdir()) / "out_noloop.mp4"

    captured: dict = {}

    def fake_run(cmd, *a, **k):
        captured["cmd"] = cmd
        return mock.Mock(returncode=0, stderr=b"")

    with mock.patch("shutil.which", return_value="/usr/bin/ffmpeg"), mock.patch(
        "subprocess.run", side_effect=fake_run
    ):
        tool._mux_audio(video, audio, out, loop=False)

    cmd = captured["cmd"]
    assert any("apad" in str(c) for c in cmd)
    assert not any("aloop" in str(c) for c in cmd)


def test_mux_audio_missing_ffmpeg_raises():
    tool = _make_tool()
    video = _tmp(".mp4")
    audio = _tmp(".mp3")
    out = Path(tempfile.gettempdir()) / "out_no_ffmpeg.mp4"
    with mock.patch("shutil.which", return_value=None):
        try:
            tool._mux_audio(video, audio, out, False)
            assert False, "expected ToolError"
        except Exception as e:  # noqa: BLE001
            assert "ffmpeg" in str(e)


def test_mux_audio_missing_audio_file_raises():
    tool = _make_tool()
    video = _tmp(".mp4")
    out = Path(tempfile.gettempdir()) / "out_no_audio.mp4"
    with mock.patch("shutil.which", return_value="/usr/bin/ffmpeg"):
        try:
            tool._mux_audio(video, Path("/nonexistent/bgm.mp3"), out, False)
            assert False, "expected ToolError"
        except Exception as e:  # noqa: BLE001
            assert "不存在" in str(e) or "not exist" in str(e).lower()
