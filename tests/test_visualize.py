"""
scripts/visualize.py の純粋ロジック単体テスト (numba を起動しない)。

検証対象:
  - TrajectorySnapshot / RunRecord の不変性
  - snapshots_to_arrays: スナップショット列の転置
  - histogram_bins: 等幅ヒストグラム (空/単一値/通常ケース)
  - summary_stats: 集計統計
  - build_plot_data: 統合データ辞書の構造
  - render_html: HTML 出力の必須要素 & JSON 埋め込み & エスケープ
  - write_html: ファイル書き出し (tmp_path 使用)
"""

from __future__ import annotations

import json
import re
from dataclasses import FrozenInstanceError

import pytest

from scripts.visualize import (
    RunRecord,
    SpinFrame,
    TrajectorySnapshot,
    build_plot_data,
    grid_shape_for,
    histogram_bins,
    render_html,
    snapshots_to_arrays,
    spin_frames_to_arrays,
    summary_stats,
    write_html,
)


# ============================================================
#  Fixtures
# ============================================================
def _make_snap(
    step: int = 0, cut: int = 10000, best_cut: int = 10000,
    beta_reset: bool = False, improvement: bool = False,
) -> TrajectorySnapshot:
    return TrajectorySnapshot(
        step=step, cut=cut, best_cut=best_cut,
        mean_abs_x=1.0, std_abs_x=0.1,
        mean_e=1.0, std_e=0.05,
        beta_inj=0.01, a_t=3.0, num_positive=1000,
        beta_reset=beta_reset, improvement=improvement,
    )


def _make_record(
    trajectory=(), final_cuts=(13000, 13100, 13200)
) -> RunRecord:
    return RunRecord(
        method="CAC", graph_name="G22",
        n_spins=2000, n_edges=19990,
        num_trials=len(final_cuts), num_outer_steps=50000,
        config={"alpha": 3.0, "rho": 1.0},
        final_cuts=final_cuts,
        trajectory=trajectory,
        wall_time_sec=12.3,
        timestamp="2026-04-17T12:34:56",
        target_cut=13359,
    )


# ============================================================
#  Immutability
# ============================================================
class TestImmutability:
    def test_snapshot_is_frozen(self) -> None:
        snap = _make_snap()
        with pytest.raises(FrozenInstanceError):
            snap.cut = 99  # type: ignore[misc]

    def test_run_record_is_frozen(self) -> None:
        rec = _make_record()
        with pytest.raises(FrozenInstanceError):
            rec.method = "SA"  # type: ignore[misc]


# ============================================================
#  snapshots_to_arrays
# ============================================================
class TestSnapshotsToArrays:
    def test_transposes_into_columns(self) -> None:
        snaps = [
            _make_snap(step=0, cut=100),
            _make_snap(step=10, cut=200),
        ]
        arr = snapshots_to_arrays(snaps)
        assert arr["step"] == [0, 10]
        assert arr["cut"] == [100, 200]
        assert len(arr["beta_inj"]) == 2

    def test_filters_reset_steps(self) -> None:
        snaps = [
            _make_snap(step=0, beta_reset=False),
            _make_snap(step=100, beta_reset=True),
            _make_snap(step=200, beta_reset=False),
            _make_snap(step=300, beta_reset=True),
        ]
        arr = snapshots_to_arrays(snaps)
        assert arr["reset_steps"] == [100, 300]

    def test_filters_improvement_steps_with_cuts(self) -> None:
        snaps = [
            _make_snap(step=0, best_cut=100, improvement=True),
            _make_snap(step=50, best_cut=100, improvement=False),
            _make_snap(step=100, best_cut=200, improvement=True),
        ]
        arr = snapshots_to_arrays(snaps)
        assert arr["improve_steps"] == [0, 100]
        assert arr["improve_cuts"] == [100, 200]

    def test_empty_list_returns_empty_arrays(self) -> None:
        arr = snapshots_to_arrays([])
        for key in ("step", "cut", "best_cut", "reset_steps",
                    "improve_steps"):
            assert arr[key] == []


# ============================================================
#  histogram_bins
# ============================================================
class TestHistogramBins:
    def test_empty_values(self) -> None:
        h = histogram_bins([])
        assert h["counts"] == []
        assert h["edges"] == []
        assert h["centers"] == []

    def test_single_value_no_zero_width(self) -> None:
        h = histogram_bins([42, 42, 42], num_bins=10)
        # lo == hi の場合は幅 1 の単一ビンで全件集約
        assert h["counts"] == [3]
        assert h["edges"] == [42.0, 43.0]
        assert h["centers"] == [42.5]

    def test_normal_distribution_preserves_count(self) -> None:
        values = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        h = histogram_bins(values, num_bins=5)
        assert sum(h["counts"]) == len(values)
        assert len(h["counts"]) == 5
        assert len(h["edges"]) == 6
        assert len(h["centers"]) == 5

    def test_max_value_goes_into_last_bin(self) -> None:
        """hi == lo + num_bins * width のとき idx が out-of-range にならないこと"""
        values = [0, 10]
        h = histogram_bins(values, num_bins=2)
        assert sum(h["counts"]) == 2  # 両方ビンに入る


# ============================================================
#  summary_stats
# ============================================================
class TestSummaryStats:
    def test_empty_cuts(self) -> None:
        s = summary_stats([], target_cut=13359)
        assert s == {"mean": 0.0, "max": 0, "min": 0, "num_optimal": 0}

    def test_normal_case(self) -> None:
        s = summary_stats([13300, 13359, 13359, 13200], target_cut=13359)
        assert s["mean"] == pytest.approx((13300 + 13359 + 13359 + 13200) / 4)
        assert s["max"] == 13359
        assert s["min"] == 13200
        assert s["num_optimal"] == 2

    def test_no_optimal_hits(self) -> None:
        s = summary_stats([100, 200, 300], target_cut=999)
        assert s["num_optimal"] == 0


# ============================================================
#  build_plot_data
# ============================================================
class TestBuildPlotData:
    def test_includes_all_expected_keys(self) -> None:
        rec = _make_record(trajectory=(_make_snap(),))
        data = build_plot_data(rec)
        expected_keys = {
            "method", "graph_name", "n_spins", "n_edges",
            "num_trials", "num_outer_steps", "target_cut",
            "wall_time_sec", "timestamp", "config",
            "final_cuts", "trajectory", "histogram", "stats",
        }
        assert expected_keys.issubset(data.keys())

    def test_trajectory_and_histogram_populated(self) -> None:
        rec = _make_record(
            trajectory=(_make_snap(step=0), _make_snap(step=100)),
            final_cuts=(13000, 13200, 13100),
        )
        data = build_plot_data(rec)
        assert data["trajectory"]["step"] == [0, 100]
        assert sum(data["histogram"]["counts"]) == 3

    def test_config_contains_original_keys(self) -> None:
        rec = _make_record()
        data = build_plot_data(rec)
        assert data["config"]["alpha"] == 3.0
        assert data["config"]["rho"] == 1.0


# ============================================================
#  render_html
# ============================================================
class TestRenderHtml:
    def test_contains_method_and_graph(self) -> None:
        rec = _make_record(trajectory=(_make_snap(),))
        html = render_html(rec)
        assert "CAC" in html
        assert "G22" in html
        assert "N=2000" in html
        assert "K=19990" in html

    def test_contains_plotly_cdn(self) -> None:
        rec = _make_record()
        html = render_html(rec)
        assert "cdn.plot.ly/plotly" in html

    def test_embeds_valid_json_data(self) -> None:
        """埋め込まれた JSON が valid でパース可能なこと"""
        rec = _make_record(
            trajectory=(_make_snap(step=5),),
            final_cuts=(13000, 13100),
        )
        html = render_html(rec)
        # <script id="viz-data" type="application/json">...</script> を抽出
        match = re.search(
            r'<script id="viz-data"[^>]*>(.*?)</script>',
            html, re.DOTALL,
        )
        assert match is not None, "viz-data script tag not found"
        raw = match.group(1)
        # \u003c など Unicode エスケープは json.loads がそのまま処理する
        data = json.loads(raw)
        assert data["method"] == "CAC"
        assert data["trajectory"]["step"] == [5]
        assert data["final_cuts"] == [13000, 13100]

    def test_all_chart_divs_present(self) -> None:
        rec = _make_record(trajectory=(_make_snap(),))
        html = render_html(rec)
        for chart_id in (
            "chart_cut", "chart_x", "chart_e",
            "chart_beta", "chart_sign", "chart_hist",
        ):
            assert f'id="{chart_id}"' in html

    def test_escapes_method_name_with_html_special_chars(self) -> None:
        """method に <script> が入っても HTML/JSON の両方でエスケープされる"""
        rec = RunRecord(
            method="<script>alert('xss')</script>",
            graph_name="G22", n_spins=100, n_edges=200,
            num_trials=1, num_outer_steps=10, config={},
            final_cuts=(1,), trajectory=(),
            wall_time_sec=1.0, timestamp="2026-01-01",
            target_cut=1,
        )
        html = render_html(rec)
        # 生の <script>alert は HTML のどこにも出現してはいけない
        # (HTML エスケープ: &lt;, JSON エスケープ: \u003c のいずれか)
        assert "<script>alert" not in html
        # h1/title では HTML エスケープ
        assert "&lt;script&gt;alert" in html
        # JSON 埋め込みでは \u003c エスケープ
        assert "\\u003cscript\\u003ealert" in html

    def test_json_embed_escapes_closing_script_tag(self) -> None:
        """JSON 内に </script> があっても script タグを切らない"""
        rec = _make_record()
        # config に </script> を含む値を埋め込む
        rec2 = RunRecord(
            method="CAC", graph_name="G22",
            n_spins=2000, n_edges=19990,
            num_trials=1, num_outer_steps=1,
            config={"note": "<!-- </script> -->"},
            final_cuts=(1,), trajectory=(),
            wall_time_sec=1.0, timestamp="t",
            target_cut=1,
        )
        html = render_html(rec2)
        # テンプレート本来の </script> 数 (Plotly CDN, viz-data, メインスクリプト)
        # = 3。config に </script> を含む値があっても増えてはいけない
        # (JSON の途中で script タグが閉じて XSS が発生するため)
        script_close_count = html.count("</script>")
        assert script_close_count == 3
        # 生の </script> が JSON blob 内部にはないこと
        # (viz-data の中身を抽出して確認)
        match = re.search(
            r'<script id="viz-data"[^>]*>(.*?)</script>',
            html, re.DOTALL,
        )
        assert match is not None
        assert "</script>" not in match.group(1)

    def test_shows_summary_statistics(self) -> None:
        rec = _make_record(final_cuts=(13000, 13359, 13359))
        html = render_html(rec)
        assert "13359" in html  # target_cut
        assert "2 / 3" in html or "2/3" in html  # optimal hits (spacing varies)

    def test_includes_config_rows(self) -> None:
        rec = _make_record()
        html = render_html(rec)
        assert "alpha" in html
        assert "rho" in html


# ============================================================
#  write_html (tmp_path)
# ============================================================
class TestWriteHtml:
    def test_creates_file_with_expected_content(self, tmp_path) -> None:
        rec = _make_record(trajectory=(_make_snap(),))
        out = tmp_path / "viz" / "run.html"
        written = write_html(rec, out)
        assert written == out
        assert out.exists()
        content = out.read_text(encoding="utf-8")
        assert "<!DOCTYPE html>" in content
        assert "CAC" in content

    def test_creates_parent_directory(self, tmp_path) -> None:
        rec = _make_record()
        out = tmp_path / "deep" / "nested" / "x.html"
        write_html(rec, out)
        assert out.exists()


# ============================================================
#  SpinFrame / grid_shape_for / AHC プレーヤー
# ============================================================
class TestSpinFrame:
    def test_spin_frame_is_frozen(self) -> None:
        f = SpinFrame(step=0, cut=10, scale=1.0, x_q=(1, 2, 3))
        with pytest.raises(FrozenInstanceError):
            f.step = 1  # type: ignore[misc]

    def test_spin_frames_to_arrays_transposes(self) -> None:
        frames = [
            SpinFrame(step=0, cut=100, scale=0.1, x_q=(10, -5)),
            SpinFrame(step=50, cut=200, scale=0.2, x_q=(30, -60)),
        ]
        arr = spin_frames_to_arrays(frames)
        assert arr["step"] == [0, 50]
        assert arr["cut"] == [100, 200]
        assert arr["scale"] == [0.1, 0.2]
        assert arr["x_q"] == [[10, -5], [30, -60]]

    def test_spin_frames_to_arrays_empty(self) -> None:
        arr = spin_frames_to_arrays([])
        assert arr == {"step": [], "cut": [], "scale": [], "x_q": []}


class TestGridShape:
    def test_grid_shape_for_2000(self) -> None:
        cols, rows = grid_shape_for(2000)
        # N=2000, 縦横比 1.25 → cols ≒ 50, rows ≒ 40
        assert cols * rows >= 2000
        # 縦横比が 0.5..2.0 の範囲に入る (妥当)
        assert 0.5 <= cols / rows <= 2.0

    def test_grid_shape_for_1(self) -> None:
        cols, rows = grid_shape_for(1)
        assert cols * rows >= 1

    def test_grid_shape_for_0(self) -> None:
        cols, rows = grid_shape_for(0)
        assert cols >= 1 and rows >= 1


class TestBuildPlotDataWithSpinFrames:
    def test_includes_spin_frames_and_grid(self) -> None:
        frames = (
            SpinFrame(step=0, cut=100, scale=0.01, x_q=(0,) * 4),
            SpinFrame(step=10, cut=200, scale=0.02, x_q=(10, 20, -30, -40)),
        )
        rec = RunRecord(
            method="CAC", graph_name="G22",
            n_spins=4, n_edges=6,
            num_trials=1, num_outer_steps=10,
            config={}, final_cuts=(100, 200),
            trajectory=(), spin_frames=frames,
            wall_time_sec=1.0, timestamp="t", target_cut=6,
        )
        data = build_plot_data(rec)
        assert "spin_frames" in data
        assert data["spin_frames"]["step"] == [0, 10]
        assert data["spin_frames"]["x_q"] == [[0, 0, 0, 0], [10, 20, -30, -40]]
        assert "grid" in data
        assert data["grid"]["cols"] >= 1 and data["grid"]["rows"] >= 1
        assert data["grid"]["cols"] * data["grid"]["rows"] >= 4

    def test_default_spin_frames_is_empty(self) -> None:
        """spin_frames フィールドは既定値 () のため既存呼び出しが壊れない"""
        rec = _make_record()
        data = build_plot_data(rec)
        assert data["spin_frames"] == {
            "step": [], "cut": [], "scale": [], "x_q": []
        }


class TestRenderHtmlPlayer:
    def test_html_contains_player_controls(self) -> None:
        rec = _make_record()
        html = render_html(rec)
        # AHC 風プレーヤー UI の主要要素
        assert 'id="slider"' in html
        assert 'id="btn-play"' in html
        assert 'id="btn-prev"' in html
        assert 'id="btn-next"' in html
        assert 'id="spin-canvas"' in html

    def test_html_embeds_spin_frames(self) -> None:
        frames = (
            SpinFrame(step=0, cut=100, scale=0.01, x_q=(50, -50)),
        )
        rec = RunRecord(
            method="CAC", graph_name="G22",
            n_spins=2, n_edges=1,
            num_trials=1, num_outer_steps=1,
            config={}, final_cuts=(100,),
            trajectory=(), spin_frames=frames,
            wall_time_sec=1.0, timestamp="t", target_cut=1,
        )
        html = render_html(rec)
        # viz-data 内に spin_frames の中身が含まれる
        match = re.search(
            r'<script id="viz-data"[^>]*>(.*?)</script>',
            html, re.DOTALL,
        )
        assert match is not None
        data = json.loads(match.group(1))
        assert data["spin_frames"]["x_q"] == [[50, -50]]
