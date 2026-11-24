"""Single cosmic-muon event displays on the baseline board (cli.BASELINE), configurations 1-5 of docs/design_options.md.

Configurations 1, 2, 3, 5: layout.Phase1Layout + cosmic.emit_batch + tracer.trace + cosmic.board_reflections, as
cosmic_rates.py, with the cli.BASELINE detector inputs: S13361 PDE, nominal mirror reflectivity, board pixel map
(layout.Board.pad_index), single-pole aerogel n = 1.03 at 400 nm (plain tile), board reflectance, Xe at P_REF (air in
configuration 2), Rayleigh scattering in the tile. Configurations 2 and 3 with the f = 50.8 mm mirror (Edmund #43-543),
3 also with f = 76.2 mm (#43-553, tag c3_f76).
Configuration 4 (tag c4_g50): proximity_config4.ProxLayout (mirror-image frame), gap 50 mm, no window slab, no board
reflection; emission = emit_batch's formula with prox_segments (check_emit() checks it bit for bit against emit_batch
on a mirror layout).
Muon: p = 5 GeV/c, in the x-z plane through the centre of the telescope acceptance at its zenith angle, moved to the
nearest pad centre. One event = Poisson photon counts per pad and light class from one traced photon sample (expected
counts mu per pad, the model of analysis.pad_hits); a pad fires if any count > 0; class of a fired pad = class with the
most drawn photons. Per tag and angle --nseed independent events; the displayed event is the lowest seed whose number
of fired pads (muon pad excluded) equals the lower median over the seeds. The random streams depend on the configuration
number, not the tag: c3 and c3_f76 see the same muons and photon samples.
Outputs in --outdir: event_displays.json (counts per seed, chosen seed, fired pads and classes), event_displays_locus.npz
(landing points of unscattered light on the sensor plane, seed 0); figures: event_displays_plots.py.
"""
import os
for _v in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_v] = '1'

import json  # noqa: E402
import multiprocessing as mp  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

from xerich import cli, cosmic, provenance, tracer  # noqa: E402
from xerich.cherenkov import beta as beta_of, qe_band, response  # noqa: E402
from xerich.constants import (AEROGEL_N_AIR, ALPHA, HBARC_EV_CM, HC, M_MU, MIRROR_D, MIRROR_R, MIRROR_SHORT_F, P_REF,  # noqa: E402
                              SENSOR_HALF_2IN)
from xerich.layout import Aerogel, Board, Mirror, Phase1Layout, Sensor  # noqa: E402
from xerich.media import Rayleigh, aerogel_medium, air, xenon  # noqa: E402
import proximity_config4 as pc  # noqa: E402
from cosmic_rates import BOT_HALF, TOP_HALF  # noqa: E402

B = cli.BASELINE
RESP = response(B['response'])
BOARD = Board()
P_MU = 5.0                      # muon momentum [GeV/c]
BETA = float(beta_of(np.array([P_MU]), M_MU)[0])
# tag: (configuration, mirror key of configurations 2-3, configuration-4 gap [mm]), display zenith angles [deg]
SPEC = {'c1': (1, None, None), 'c2': (2, '43-543', None), 'c3': (3, '43-543', None), 'c3_f76': (3, '43-553', None),
        'c4_g50': (4, None, 50.0), 'c5': (5, None, None)}
ANGLES = {'c1': (0.0, 4.0, 7.0, 10.0), 'c2': (0.0, 15.0, 30.0, 39.0), 'c3': (0.0, 15.0, 30.0, 39.0),
          'c3_f76': (0.0, 15.0, 30.0, 39.0), 'c4_g50': (0.0, 20.0, 30.0, 38.0), 'c5': (0.0, 5.0, 8.0, 14.0)}
GAS, TILE, OTHER = 0, 1, 2      # light classes: unscattered gas / aerogel light; other (Rayleigh-scattered, board-reflected)


def snap(x):
    """Nearest pad centre to x [mm]: the acceptance centre (x = 0 for near-vertical muons) lies on the dead strip
    between the packages, so the muon is moved by at most half a pitch to cross the middle of a pad."""
    c = BOARD.pad_lo+BOARD.pad/2
    return float(c[np.argmin(np.abs(c-x))])


def tile_medium():
    return aerogel_medium(B['aerogel_index'], AEROGEL_N_AIR, B['aerogel_lambda_ref'])


def make(tag):
    """Layout of tag at Xe P_REF (air in configuration 2)."""
    cfg, mirror, gap = SPEC[tag]
    xe = xenon(P_REF)
    if cfg in (1, 5):
        tile = Aerogel(AEROGEL_N_AIR) if cfg == 5 else None
        return Phase1Layout(Mirror(MIRROR_R, MIRROR_D, reflectivity=B['mirror_r']), Sensor(SENSOR_HALF_2IN, model=B['sensor_model']), xe, tile,
                            tile_medium=tile_medium() if tile else None, tile_scatter=Rayleigh() if tile else None)
    if cfg in (2, 3):
        f, D = MIRROR_SHORT_F[mirror]
        return Phase1Layout(Mirror(2*f, D, reflectivity=B['mirror_r']), Sensor(SENSOR_HALF_2IN, model=B['sensor_model']), air() if cfg == 2 else xe,
                            Aerogel(AEROGEL_N_AIR), tile_medium=tile_medium(), tile_scatter=Rayleigh())
    return pc.ProxLayout(xe, gap, tile_medium(), Rayleigh(), None, sensor_half=BOARD.half)


def muon(tag, zen_deg):
    """Entry point P0 and direction U (tracer frame) of the muon at zenith zen_deg in the x-z plane, through the centre of
    the telescope acceptance at that angle (real frame: tilted towards +x while going down), and its impact on the sensor
    plane; None outside the telescope acceptance."""
    t = np.tan(np.radians(zen_deg)); c = np.cos(np.radians(zen_deg)); s = np.sin(np.radians(zen_deg))
    lay = make(tag); L = lay.z_sensor
    if SPEC[tag][0] != 4:   # top counter at the SiPM plane, bottom counter at the mirror vertex plane, x_b = x_s + L t
        lo, hi = max(-TOP_HALF, -BOT_HALF['mirror']-L*t), min(TOP_HALF, BOT_HALF['mirror']-L*t)
        xs = snap(0.5*(lo+hi)); y = snap(0.0)
        P0 = np.array([xs, y, L]); U = np.array([s, 0.0, -c]); imp = (xs, y)
    else:                   # real frame: top counter at the tile top, bottom counter at the SiPM plane; tracer frame z -> -z
        lo, hi = max(-TOP_HALF, -BOT_HALF['prox']-L*t), min(TOP_HALF, BOT_HALF['prox']-L*t)
        xi = snap(0.5*(lo+hi)+L*t); y = snap(0.0); x0 = xi-L*t
        P0 = np.array([x0, y, 0.0]); U = np.array([s, 0.0, c]); imp = (xi, y)
    if hi < lo:
        return None
    return P0, U, imp


def emit_generic(seg_fn, index, rng, P0, U, beta, n_per, origins=('gas', 'tile')):
    """cosmic.emit_batch's formula (same random calls in the same order) with a segment function seg_fn(p, u)."""
    E_lo, E_hi = qe_band()
    region = {'gas': -1, 'tile': 0}
    segs = [seg_fn(p, u) for p, u in zip(P0, U)]
    out = []
    MS = cosmic.MAX_SEGMENTS
    for oi, org in enumerate(cosmic.ORIGINS):
        if org not in origins or index[org] is None:
            continue
        n_org = n_per[org] if isinstance(n_per, dict) else n_per
        a0 = np.zeros((len(P0), MS)); ln = np.zeros((len(P0), MS))
        for m, sg in enumerate(segs):
            for k, (a, b) in enumerate(sg[org]):
                a0[m, k] = a; ln[m, k] = b-a
        Lt = ln.sum(1); mu = np.flatnonzero(Lt > 0)
        if not len(mu):
            continue
        idx = np.repeat(mu, n_org); n = len(idx)
        cum = np.cumsum(ln[idx], 1)/Lt[idx, None]; r = rng.random(n)
        k = np.minimum((r[:, None] > cum).sum(1), MS-1)
        s = a0[idx, k]+rng.random(n)*ln[idx, k]
        E = rng.uniform(E_lo, E_hi, n); lam = HC/E; nn = index[org](lam)
        ct = 1/(beta[idx]*nn); ok = ct < 1; ct = np.minimum(ct, 1.0); st = np.sqrt(1-ct*ct)
        phi = 2*np.pi*(np.tile(np.arange(n_org), len(mu))+rng.random(n))/n_org
        e1, e2 = tracer.basis(U[idx])
        d = ct[:, None]*U[idx]+st[:, None]*(np.cos(phi)[:, None]*e1+np.sin(phi)[:, None]*e2)
        w = np.where(ok, (E_hi-E_lo)*ALPHA/HBARC_EV_CM*(1-ct*ct)*Lt[idx]/10/n_org, 0.0)
        out.append(dict(p=P0[idx]+s[:, None]*U[idx], d=d, lam=lam, reg=np.full(n, region[org], np.int16), muon=idx,
                        origin=np.full(n, oi, np.int8), phi=phi, w_ft=w, qe=RESP(E)))
    if not out:
        return None
    return {k: np.concatenate([o[k] for o in out]) for k in out[0]}


def emit(tag, lay, rng, P0, U, n_per):
    cfg = SPEC[tag][0]; beta = np.full(1, BETA)
    if cfg != 4:
        org = ('gas',) if cfg == 1 else ('tile',) if cfg == 2 else ('gas', 'tile')
        return cosmic.emit_batch(lay, rng, P0[None], U[None], beta, n_per, org, response=RESP)
    idx = {'gas': lay.gas.n, 'tile': lay.tile_medium.n}
    return emit_generic(lambda p, u: pc.prox_segments(lay, p, u), idx, rng, P0[None], U[None], beta, n_per)


def check_emit():
    """emit_generic with track_segments reproduces cosmic.emit_batch bit for bit (configuration 5)."""
    from xerich.sim import track_segments
    lay = make('c5'); P0, U, _ = muon('c5', 3.0)
    a = cosmic.emit_batch(lay, np.random.default_rng(1), P0[None], U[None], np.full(1, BETA), 500, ('gas', 'tile'), response=RESP)
    b = emit_generic(lambda p, u: track_segments(lay, p, u), {'gas': lay.gas.n, 'tile': lay.tile_index.n}, np.random.default_rng(1),
                     P0[None], U[None], np.full(1, BETA), 500)
    return all(np.array_equal(a[k], b[k]) for k in a)


def pad_mu(tag, zen_deg, seed, master, n_per, locus=None):
    """Expected photons per pad and light class (3, npads) for one muon from one traced photon sample, and the muon's pad.
    Board reflection in configurations 1, 2, 3, 5 (own random stream)."""
    cfg = SPEC[tag][0]
    lay = make(tag); P0, U, imp = muon(tag, zen_deg)
    rng = np.random.default_rng([master, cfg, int(round(zen_deg*10)), seed])
    rrng = np.random.default_rng([master, cfg, int(round(zen_deg*10)), seed, 1])
    ph = emit(tag, lay, rng, P0, U, {'gas': int(0.4*n_per), 'tile': n_per})
    mu = np.zeros((3, BOARD.npads))
    mupad = int(BOARD.pad_index(np.array([imp[0]]), np.array([imp[1]]))[0])
    if ph is None:
        return mu, mupad, imp
    live = ph['w_ft'] > 0
    sel = {k: v[live] for k, v in ph.items()}
    if not live.any():
        return mu, mupad, imp
    scene = lay.scene()
    e = tracer.trace(scene, sel['p'], sel['d'], sel['lam'], sel['reg'], None, 2, rng=rng)
    det = e['reason'] == tracer.REASONS.index('detected'); i = e['i'][det]
    w_in = e['w'][det]*sel['w_ft'][i]
    a = w_in*sel['qe'][i]
    if locus is not None:   # landing points on the sensor plane (on or off the board) of unscattered main branches, by origin
        on_plane = np.isin(e['reason'], [tracer.REASONS.index('detected'), tracer.REASONS.index('outside sensor')])
        main = on_plane&(e['cls'] != tracer.CLASSES.index('scattered'))&(e['w'] >= 0.3)
        for o in (GAS, TILE):
            m = main&(sel['origin'][e['i']] == o)
            q = e['p'][m, :2]; locus[o] = q[np.random.default_rng(0).permutation(len(q))[:3000]]
    pad = BOARD.pad_index(e['p'][det, 0], e['p'][det, 1])
    scat = e['cls'][det] == tracer.CLASSES.index('scattered')
    cls = np.where(scat, OTHER, sel['origin'][i]).astype(int)
    on = pad >= 0
    for c in (GAS, TILE, OTHER):
        m = on&(cls == c); mu[c] += np.bincount(pad[m], a[m], BOARD.npads)
    if cfg != 4:
        src, pe, le, we = cosmic.board_reflections(scene, rrng, e['p'][det], e['d'][det], sel['lam'][i], w_in, B['board_r_spec'], B['board_r_diff'])
        pd = BOARD.pad_index(pe[:, 0], pe[:, 1]); m = pd >= 0
        mu[OTHER] += np.bincount(pd[m], we[m]*RESP(HC/le[m]), BOARD.npads)
    return mu, mupad, imp


def draw_event(mu, mupad, rng):
    """One realisation: Poisson counts per class and pad; fired pads and their class (most drawn photons; ties: gas, tile,
    other). The muon's pad is excluded (fired by the muon itself)."""
    n = rng.poisson(mu)
    fired = n.sum(0) > 0
    cls = np.argmax(n, 0)
    if mupad >= 0:
        fired[mupad] = False
    return fired, cls


def counts(fired, cls):
    return {nm: int((fired&(cls == c)).sum()) for nm, c in (('gas', GAS), ('tile', TILE), ('other', OTHER))}


def job(a):
    tag, z, s, master, n_per = a
    cfg = SPEC[tag][0]
    loc = {} if s == 0 else None
    mu, mp_, imp = pad_mu(tag, z, s, master, n_per, locus=loc)
    fired, cls = draw_event(mu, mp_, np.random.default_rng([master, cfg, int(round(z*10)), s, 2]))
    keep = np.ones(mu.shape[1], bool)
    if mp_ >= 0:
        keep[mp_] = False
    return tag, z, s, dict(fired=np.flatnonzero(fired).tolist(), cls=cls[fired].tolist(), n=counts(fired, cls), mupad=mp_,
                           imp=[float(v) for v in imp], expected_fired=float((1-np.exp(-mu[:, keep].sum(0))).sum()),
                           expected_photons_by_class=mu[:, keep].sum(1).tolist()), loc


def main():
    p = cli.parser(__doc__)
    p.add_argument('--nseed', type=int, default=21, help='events per tag and angle (default 21; the displayed one has the lower median of fired pads)')
    p.add_argument('--nphot', type=int, default=20000, help='sampled tile photons per muon, 0.4 x as many gas photons (default 20000)')
    p.add_argument('--seed', type=int, default=20261015, help='master seed (default 20261015)')
    p.add_argument('--nproc', type=int, default=24, help='worker processes (default 24)')
    args = p.parse_args(); out = cli.outdir(args)
    if not check_emit():
        raise SystemExit('emit_generic differs from cosmic.emit_batch')
    t0 = time.time()
    jobs = [(t, z, s, args.seed, args.nphot) for t in SPEC for z in ANGLES[t] for s in range(args.nseed)]
    with mp.get_context('fork').Pool(args.nproc) as pool:
        res = pool.map(job, jobs, chunksize=1)
    ev, loci = {}, {}
    for t, z, s, r, loc in res:
        ev.setdefault(t, {}).setdefault('%g' % z, {})[s] = r
        if loc is not None:
            for o, v in loc.items():
                loci['%s_z%g_o%d' % (t, z, o)] = v
    summary = {}
    for t, d in ev.items():
        for z, seeds in d.items():
            tot = np.array([sum(seeds[s]['n'].values()) for s in range(args.nseed)])
            med = int(np.sort(tot)[(len(tot)-1)//2]); pick = int(np.flatnonzero(tot == med)[0])
            summary.setdefault(t, {})[z] = dict(chosen_seed=pick, median_fired=med, fired_per_seed=tot.tolist(),
                                                mean_fired=float(tot.mean()), event=seeds[pick],
                                                expected_fired_seed0=seeds[0]['expected_fired'],
                                                expected_photons_by_class_seed0=seeds[0]['expected_photons_by_class'])
            print('%-6s zenith %5s  seed %2d  fired %s  (lower median %d, mean %.2f over %d seeds; expected %.2f)' % (
                t, z, pick, seeds[pick]['n'], med, tot.mean(), args.nseed, seeds[0]['expected_fired']), flush=True)
    meta = dict(p_GeV=P_MU, beta=BETA, spec={t: list(v) for t, v in SPEC.items()}, angles=ANGLES, nseed=args.nseed, events=summary)
    with open(out/'event_displays.json', 'w') as f:
        json.dump(provenance.with_record(meta), f, indent=1)
    np.savez_compressed(out/'event_displays_locus.npz', **loci, **provenance.npz())
    print('wall %.0f s, %d workers' % (time.time()-t0, args.nproc))


if __name__ == '__main__':
    main()
