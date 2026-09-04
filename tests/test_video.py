import numpy as np

import video


def test_progress_video_uses_post_reset_frames_and_logs_once(
        tmp_path, monkeypatch):
    recorder = video.VideoRecorder(tmp_path, use_wandb=True)
    recorder.enabled = True
    recorder.frames = [
        np.full((8, 8, 3), value, dtype=np.uint8)
        for value in (1, 2, 3)
    ]
    reward_frames = [
        np.full((8, 8, 3), value, dtype=np.uint8)
        for value in (10, 20, 30)
    ]
    saved = {}

    def fake_save(frames, progress, success, is_ood, output_path, fps):
        saved["frames"] = frames
        saved["progress"] = progress
        saved["success"] = success
        saved["is_ood"] = is_ood
        saved["path"] = output_path
        saved["fps"] = fps
        return output_path

    logs = []
    monkeypatch.setattr(video, "save_progress_composite_video", fake_save)
    monkeypatch.setattr(video.wandb, "run", object())
    monkeypatch.setattr(
        video.wandb, "Video", lambda path, fps, format: (path, fps, format))
    monkeypatch.setattr(
        video.wandb, "log", lambda data, commit: logs.append((data, commit)))

    path = recorder.save_progress(
        "120.mp4", reward_frames,
        progress=[0.1, 0.7], success=[0.0, 1.0], is_ood=[False, True])

    assert [int(frame[0, 0, 0]) for frame in saved["frames"]] == [20, 30]
    np.testing.assert_allclose(saved["progress"], [0.1, 0.7])
    np.testing.assert_allclose(saved["success"], [0.0, 1.0])
    np.testing.assert_array_equal(saved["is_ood"], [False, True])
    assert path == tmp_path / "eval_video" / "120_progress.mp4"
    assert saved["fps"] == 20
    assert len(logs) == 1
    assert list(logs[0][0]) == ["eval/progress_video"]
    assert logs[0][1] is False
