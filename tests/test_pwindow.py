"""Fused-silica pressure window (layout.Window with gap > 0, cosmic origin 'window', cosmic_rates.py --pwindow-t)."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from xerich import cosmic, sim
from xerich.constants import AEROGEL_N_AIR, MIRROR_D, MIRROR_R
from xerich.layout import Aerogel, Mirror, Phase1Layout, Sensor, Window
from xerich.media import air, fused_silica, n_fused_silica, xenon

T, GAP, HALF = 5.0, 1.0, 40.0


def load_driver(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parents[1]/'studies'/('%s.py' % name))
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


def layout(tile=True):
    win = Window(thickness=T, gap=GAP, half=HALF, glass=fused_silica(), outer=air())
    return Phase1Layout(Mirror(MIRROR_R, MIRROR_D), Sensor(25.4), xenon(2.0), Aerogel(AEROGEL_N_AIR) if tile else None,
                        window=win, gap=2*GAP+T)


def test_fused_silica_index():
    """Malitson's fit: n_d = 1.4585 at 587.6 nm; n = 1.4701 at 400 nm."""
    assert n_fused_silica(587.6) == pytest.approx(1.4585, abs=1e-4)
    assert n_fused_silica(400.0) == pytest.approx(1.4701, abs=1e-4)


def test_scene_slabs():
    """Tile, window, outer air slab, in this order and stacked: tile top, GAP of gas, window, GAP of air, sensor plane."""
    lay = layout(); sl = lay.scene().slabs; z = lay.z_sensor
    assert len(sl) == 3
    assert sl[0].hi[2] == pytest.approx(z-2*GAP-T)
    assert (sl[1].lo[2], sl[1].hi[2]) == pytest.approx((z-GAP-T, z-GAP)) and sl[1].hi[0] == HALF
    assert (sl[2].lo[2], sl[2].hi[2]) == pytest.approx((z-GAP, z))
    assert sl[1].medium.n(np.array([400.0]))[0] == pytest.approx(n_fused_silica(400.0))
    assert sl[2].medium.n(np.array([400.0]))[0] == pytest.approx(air().n(np.array([400.0]))[0])
    with pytest.raises(ValueError):
        Phase1Layout(lay.mirror, lay.sensor, lay.gas, lay.tile, window=lay.window, gap=GAP+T).scene()


@pytest.mark.parametrize('tile', [True, False])
def test_window_origin(tile):
    """Window light starts inside the window slab, in its region (slab index 1 with a tile, 0 without)."""
    lay = layout(tile); z = lay.z_sensor
    ph = cosmic.emit_batch(lay, np.random.default_rng(1), np.array([[1.0, 2.0, z]]), np.array([[0.0, 0.0, -1.0]]), np.ones(1), 500,
                           ('window',))
    assert np.all(ph['reg'] == int(tile)) and np.all(ph['origin'] == cosmic.ORIGINS.index('window'))
    assert np.all((ph['p'][:, 2] > z-GAP-T) & (ph['p'][:, 2] < z-GAP))


def test_track_segments_outer():
    """A vertical track from the sensor plane: GAP in the outer slab, T in the window, no gas there."""
    lay = layout(); z = lay.z_sensor
    seg = sim.track_segments(lay, np.array([0.0, 0.0, z]), np.array([0.0, 0.0, -1.0]))
    assert [(b-a) for a, b in seg['outer']] == pytest.approx([GAP]) and [(b-a) for a, b in seg['window']] == pytest.approx([T])
    assert all(a >= GAP+T-1e-9 for a, _ in seg['gas'])


def test_driver_window_layout_and_refusals(tmp_path, monkeypatch):
    """cosmic_rates.py: SiPM refocused by T (1 - 1/n(400 nm)) with the window; --pwindow-t refused outside c1, c5."""
    cr = load_driver('cosmic_rates')
    from xerich import cli
    cr._G.update(inputs=cli.preset('baseline-board'), window=Window(thickness=T, gap=GAP, half=HALF, glass=fused_silica(), outer=air()))
    run = {r['name']: r for r in cr.runs(0.01, window=True)}['c5_xe2_plain_geo']
    lay = cr.layout_of(run)
    assert lay.z_sensor == pytest.approx(lay.mirror.f+T*(1-1/n_fused_silica(cr.PWINDOW_LAM)))
    assert lay.z_tile_top == pytest.approx(lay.z_sensor-2*GAP-T)
    for argv in (['--pwindow-t', '5'], ['--pwindow-t', '5', '--groups', 'c5', 'c4'], ['--pwindow-t', '-1', '--groups', 'c5'],
                 ['--groups', 'c5', '--only', 'no-such-run']):
        monkeypatch.setattr(sys, 'argv', ['cosmic_rates.py', '--outdir', str(tmp_path)]+argv)
        with pytest.raises(SystemExit):
            cr.main()
