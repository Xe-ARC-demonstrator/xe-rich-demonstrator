"""Event-display figures from event_displays.py: per tag, a to-scale side view and the board at four zenith angles.

Reads event_displays.json and event_displays_locus.npz from --input (default --outdir). Side view: gas (Xe or air) volume,
mirror (bold: clear aperture), SiPM board, aerogel tile, trigger counters, the muons at the four angles (bold: the one
whose in-plane Cherenkov rays at 400 nm are traced, Rayleigh scattering off; x: ray lost at the mirror aperture or
escaped), scale bar. Board: fired pads coloured by light class (gas, aerogel, scattered = Rayleigh-scattered or
board-reflected), the muon's pad in black, the landing points of unscattered light (expected ring) as dots.
Figure width 5.512 in, so the font sizes are the printed sizes; common scale for the
short-mirror and proximity views (c2, c3, c3_f76, c4_g50) and for the long-mirror views (c1, c5).
Outputs in --outdir: event_displays_<tag>.png (c1, c2, c3, c3_f76, c4_g50, c5) and event_displays_c3_mirrors.png (c3 over
c3_f76: the f = 50.8 and 76.2 mm mirrors).
"""
import os
for _v in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_v] = '1'

import dataclasses  # noqa: E402
import json  # noqa: E402
import multiprocessing as mp  # noqa: E402
from pathlib import Path  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402

from xerich import cli, plotting, tracer  # noqa: E402
from xerich.sim import track_segments  # noqa: E402
import event_displays as L  # noqa: E402

RAY_ANGLE = {1: 4.0, 2: 15.0, 3: 15.0, 4: 20.0, 5: 8.0}     # per configuration, the muon whose rays are drawn (bold)
C_GAS, C_AERO, C_OTH, C_MU, C_OFF = '#2563eb', '#d97706', '#CC79A7', '#111111', '#ececec'
C_XE, C_AIR, C_SIPM, C_TILE, C_MIR = '#dce8fa', '#ecebe6', '#1f2933', '#f8deb4', '#475569'
C_TRIG, C_TRIG_E = '#10b981', '#047857'
CLS_COL = {L.GAS: C_GAS, L.TILE: C_AERO, L.OTHER: C_OTH}
F_TITLE, F_ANG, F_LEG, F_SCALE, F_ROW = 10.0, 9.6, 9.6, 9.6, 11.0   # font sizes [pt]
XW = 46.0                      # half width of the side views [mm]
LIM = 31.0                     # half width of the board displays [mm]
W_FIG = 5.512                  # figure width [in]
plt.rcParams.update({'font.family': 'DejaVu Sans'})


def cfg_of(tag):
    return L.SPEC[tag][0]


def setup(tag, RUN):
    """Display angles of tag; the muon impact points recomputed from the geometry must equal those of the run."""
    angles = [float(z) for z in RUN['angles'][tag]]
    for z in angles:
        e = RUN['events'][tag]['%g' % z]['event']
        imp = L.muon(tag, z)[2]
        assert np.allclose(imp, e['imp'], atol=1e-6), (tag, z, imp, e['imp'])
        cnt = {nm: sum(1 for c in e['cls'] if c == k) for nm, k in (('gas', L.GAS), ('tile', L.TILE), ('other', L.OTHER))}
        assert cnt == e['n'] and len(e['fired']) == sum(e['n'].values()) and e['mupad'] not in e['fired'], (tag, z)
    return angles


def rays(tag, zen):
    """In-plane Cherenkov rays at 400 nm of the muon at zen (Rayleigh off): vertex lists in the real frame."""
    cfg = cfg_of(tag)
    lay = dataclasses.replace(L.make(tag), tile_scatter=None)
    P0, U, _ = L.muon(tag, zen)
    seg = track_segments(lay, P0, U) if cfg != 4 else L.pc.prox_segments(lay, P0, U)
    nperp = np.array([U[2], 0.0, -U[0]]); nperp /= np.linalg.norm(nperp)
    pts, dirs, regs, org = [], [], [], []
    lam = 400.0
    for o, key, med, nk in ((L.GAS, 'gas', lay.gas, 5), (L.TILE, 'tile', getattr(lay, 'tile_index', None) if cfg != 4 else lay.tile_medium, 2)):
        if not seg.get(key) or med is None or (cfg == 1 and key == 'tile') or (cfg == 2 and key == 'gas'):
            continue
        n = float(med.n(np.array([lam]))[0]); ct = 1/(L.BETA*n)
        if ct >= 1:
            continue
        st = np.sqrt(1-ct*ct)
        for a, b in seg[key]:
            for s in np.linspace(a, b, nk+2)[1:-1] if key == 'gas' else ((a+b)/2,):
                for sg in (1, -1):
                    pts.append(P0+s*U); dirs.append(ct*U+sg*st*nperp); regs.append(-1 if key == 'gas' else 0); org.append(o)
    p = np.array(pts); d = np.array(dirs)
    e = tracer.trace(lay.scene(), p, d, np.full(len(p), lam), np.array(regs, np.int16), None, 0, record=True)
    out = []
    for k in range(len(e['i'])):
        if e['w'][k] < 0.3 or e['cls'][k] != 0:
            continue
        v = np.asarray(e['path'][k])
        if cfg == 4:
            v = v*np.array([1, 1, -1])
        out.append((org[e['i'][k]], v, tracer.REASONS[e['reason'][k]]))
    return out


def frame(tag):
    """Side-view extent (z_lo, z_hi) and the trigger-counter planes zt, zb [mm]."""
    cfg = cfg_of(tag); f = L.make(tag).z_sensor
    if cfg != 4:
        zt, zb = f+0.035*f+3, -0.035*f-3
    else:
        zt, zb = 3.0, -f-5.0
    return zb-4.0, zt+(12.0 if cfg in (1, 5) else 25.0), zt, zb     # room above the top counter for the scale bar


def sketch(ax, tag, angles):
    """To-scale side view; returns the bottom ends of the muon tracks (data coordinates) with angle and highlight flag."""
    cfg = cfg_of(tag); lay = L.make(tag); bh = L.BOARD.half
    z_lo, z_hi, zt, zb = frame(tag)
    ax.add_patch(Rectangle((-XW, z_lo), 2*XW, z_hi-z_lo, fc=C_AIR if cfg == 2 else C_XE, ec='none', zorder=0))
    if cfg != 4:
        f, R, D, a = lay.z_sensor, lay.mirror.R, lay.mirror.D, lay.mirror.clear_radius
        x = np.linspace(-D/2, D/2, 300); ax.plot(x, R-np.sqrt(R*R-x*x), color=C_MIR, lw=0.8, zorder=3)
        x = np.linspace(-a, a, 300); ax.plot(x, R-np.sqrt(R*R-x*x), color=C_MIR, lw=2.0, zorder=3)
        ax.add_patch(Rectangle((-bh, f), 2*bh, max(2.0, 0.012*f), fc=C_SIPM, ec='none', zorder=4))
        if lay.tile is not None:
            ax.add_patch(Rectangle((-lay.tile.half, lay.z_tile_bottom), 2*lay.tile.half, lay.tile.thickness, fc=C_TILE, ec=C_AERO, lw=0.6, zorder=2))
        top_h, bot_h = L.TOP_HALF, L.BOT_HALF['mirror']
    else:
        T, f = lay.tile_t, lay.z_sensor
        ax.add_patch(Rectangle((-lay.tile_half, -T), 2*lay.tile_half, T, fc=C_TILE, ec=C_AERO, lw=0.6, zorder=2))
        ax.add_patch(Rectangle((-bh, -f-2.0), 2*bh, 2.0, fc=C_SIPM, ec='none', zorder=4))
        top_h, bot_h = L.TOP_HALF, L.BOT_HALF['prox']
    hb = 2.0 if cfg != 1 and cfg != 5 else 4.0
    for zc, h in ((zt, top_h), (zb, bot_h)):
        ax.add_patch(Rectangle((-h, zc-hb/2), 2*h, hb, fc=C_TRIG, ec=C_TRIG_E, lw=0.4, zorder=4))
    for o, v, why in rays(tag, RAY_ANGLE[cfg]):
        col = C_GAS if o == L.GAS else C_AERO
        ax.plot(v[:, 0], v[:, 2], color=col, lw=0.6, zorder=5, alpha=0.9)
        if why in ('mirror aperture', 'escaped'):
            ax.plot(v[-1, 0], v[-1, 2], marker='x', color=col, ms=3.5, mew=0.9, zorder=6)
    ends = []
    for z in angles:
        P0, U, _ = L.muon(tag, z)
        if cfg == 4:
            P0 = P0*np.array([1, 1, -1]); U = U*np.array([1, 1, -1])
        s_in = (z_hi-P0[2])/U[2]; s_out = (z_lo-P0[2])/U[2]                 # U[2] < 0: s_in < s_out
        for xb in (-XW, XW):                                                 # clip to the side edges
            if abs(U[0]) > 1e-12:
                s = (xb-P0[0])/U[0]
                if (U[0] > 0) == (xb > 0):
                    s_out = min(s_out, s)
                else:
                    s_in = max(s_in, s)
        q1, q2 = P0+s_in*U, P0+s_out*U
        hl = z == RAY_ANGLE[cfg]
        ax.plot([q1[0], q2[0]], [q1[2], q2[2]], color='k', lw=1.3 if hl else 0.7, ls='-' if hl else (0, (3, 1.5)), zorder=7)
        ax.annotate('', xy=(q2[0], q2[2]), xytext=(q2[0]-2.5*U[0], q2[2]-2.5*U[2]),
                    arrowprops=dict(arrowstyle='-|>', color='k', lw=0.6, mutation_scale=5), zorder=7)
        ends.append((q2[0], q2[2], z, hl))
    sb = 20 if cfg in (1, 5) else 10
    xr = XW-1.5; y0 = zt+20.0 if cfg not in (1, 5) else zt-14.0
    ax.plot([xr-sb, xr], [y0, y0], color='k', lw=1.2, zorder=9, solid_capstyle='butt')
    if cfg in (1, 5):     # narrow view: text right of the view
        ax.text(XW+1.5, y0, '%g cm' % (sb/10), ha='left', va='center', fontsize=F_SCALE, zorder=9, clip_on=False)
    else:                 # text right-aligned under the bar, in the free strip right of the tile
        ax.text(xr, y0-2.0, '%g cm' % (sb/10), ha='right', va='top', fontsize=F_SCALE, zorder=9, clip_on=False)
    ax.set_xlim(-XW, XW); ax.set_ylim(z_lo, z_hi); ax.axis('off')
    return ends


def board(ax, tag, zen, RUN, LOC):
    e = RUN['events'][tag]['%g' % zen]['event']
    lo = L.BOARD.pad_lo; pw = L.BOARD.pad; npd = 2*L.BOARD.npad
    fired = dict(zip(e['fired'], e['cls']))
    for k in range(L.BOARD.npads):
        ix, iy = divmod(k, npd)
        fc = C_MU if k == e['mupad'] else CLS_COL[fired[k]] if k in fired else C_OFF
        ax.add_patch(Rectangle((lo[ix], lo[iy]), pw, pw, fc=fc, ec='#bdbdbd' if fc == C_OFF else 'none', lw=0.25, zorder=3 if fc == C_MU else 1))
    for o, col in ((L.GAS, C_GAS), (L.TILE, C_AERO)):
        key = '%s_z%g_o%d' % (tag, zen, o)
        if key in LOC.files and len(LOC[key]):
            q = LOC[key]
            ax.scatter(q[:, 0], q[:, 1], s=0.12, color=col, alpha=0.45, lw=0, zorder=2, rasterized=True)
    h = L.BOARD.half
    ax.add_patch(Rectangle((-h, -h), 2*h, 2*h, fill=False, ec='#555555', lw=0.6, zorder=3))
    if e['mupad'] < 0:
        ax.plot(*e['imp'], marker='*', color=C_MU, ms=5, zorder=4)
    ax.set_xlim(-LIM, LIM); ax.set_ylim(-LIM, LIM); ax.set_aspect('equal')
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    return e['n']


# ---- layout [in]
LEG_H, LAB_H, TITLE_H, PAD = 0.25, 0.22, 0.21, 0.02
GAP_B, GAP_SB, ROW_GAP, ROWLAB_W = 0.06, 0.08, 0.07, 0.24
GAP_SB2, ROWLAB_W2 = 0.34, 0.34   # two-row figure: more room between side view and boards, and for the row labels
GAP_B2 = 0.30                  # board gap of the two-row figure: its legend is wider than four boards
H_TWO_ROWS = 2.15              # height of the two-row figure [in]


def family(tag):
    return 'long' if cfg_of(tag) in (1, 5) else 'short'


def frame_heights():
    """Tallest side view per family [mm]: sets the common scale of the family."""
    out = {'short': 0.0, 'long': 0.0}
    for tag in L.SPEC:
        z_lo, z_hi, _, _ = frame(tag)
        out[family(tag)] = max(out[family(tag)], z_hi-z_lo)
    return out


def layout(tags, rl, fh):
    """Board size b, figure size W x H, board-row bottoms, side-view bottoms per row, scale [in/mm] per family, column
    width, board gap."""
    nr = len(tags)
    if nr == 1:     # side-view height b + TITLE_H + LEG_H - LAB_H; column width 2 XW k; four boards fill the width
        c = 2*XW/fh['short']; A = W_FIG-2*PAD-rl-GAP_SB-3*GAP_B
        b = (A-c*(TITLE_H+LEG_H-LAB_H))/(4+c)
        H = LEG_H+b+TITLE_H+PAD; row_y = [LEG_H]; bots = [LAB_H]
        k = {f: (H-PAD-LAB_H)/fh[f] for f in fh}
    else:           # side views stacked in their column, each with its label row below; one scale for all rows
        H = H_TWO_ROWS; b = (H-LEG_H-(nr-1)*ROW_GAP-TITLE_H-PAD)/nr
        row_y = [LEG_H+(nr-1-r)*(b+ROW_GAP) for r in range(nr)]
        hs = []
        for t in tags:
            z_lo, z_hi, _, _ = frame(t); hs.append(z_hi-z_lo)
        kk = (H-PAD-nr*LAB_H-0.03*(nr-1))/sum(hs); k = {'short': kk, 'long': kk}
        bots = []; y = LAB_H
        for h in hs[::-1]:
            bots.insert(0, y); y += h*kk+0.03+LAB_H
    col_w = 2*XW*k['short']; gb = GAP_B if nr == 1 else GAP_B2
    W = W_FIG if nr == 1 else PAD+rl+col_w+GAP_SB2+4*b+3*gb+PAD
    return b, W, H, row_y, bots, k, col_w, gb


def make_figure(tags, name, row_labels, RUN, LOC, out):
    """One row per tag: side view + four board displays; one legend row; angle titles above the top row."""
    nr = len(tags); rl = (ROWLAB_W2 if nr == 2 else ROWLAB_W) if row_labels else 0.0
    b, W, H, row_y, bots, k_mm, col_w, gb = layout(tags, rl, frame_heights())
    x_col = PAD+rl; xb0 = x_col+col_w+(GAP_SB if nr == 1 else GAP_SB2)
    fig = plt.figure(figsize=(W, H), dpi=400)      # text extents measured at the output resolution
    fx = lambda x: x/W; fy = lambda y: y/H
    present = {L.GAS: False, L.TILE: False, L.OTHER: False}
    for r, tag in enumerate(tags):
        cfg = cfg_of(tag); angles = setup(tag, RUN)
        z_lo, z_hi, _, _ = frame(tag)
        k = k_mm[family(tag)]; wf, hf = 2*XW*k, (z_hi-z_lo)*k; y_f = bots[r]
        assert y_f+hf <= H-PAD+1e-9, (name, tag)
        x_f = x_col+(col_w-wf)/2-(0.08 if family(tag) == 'long' else 0.0)     # long views: room for the scale text
        sk = fig.add_axes([fx(x_f), fy(y_f), fx(wf), fy(hf)])
        ends = sketch(sk, tag, angles)
        ends.sort()      # angle labels in a row below the side view, leader lines to the track ends
        xl = np.linspace(x_col+0.10, x_col+col_w-0.10, len(ends)) if nr == 1 else np.linspace(x_col-0.05, x_col+col_w+0.30, len(ends))
        for (xe, ze, z, hl), xt in zip(ends, xl):
            X, Y = fig.transFigure.inverted().transform(sk.transData.transform((xe, ze)))
            yt = y_f-0.065
            fig.add_artist(Line2D([X, fx(xt)], [Y, fy(yt)], transform=fig.transFigure, color='#666666', lw=0.4))
            fig.text(fx(xt), fy(yt-0.008), '%g°' % z, ha='center', va='top', fontsize=F_ANG, fontweight='bold' if hl else 'normal')
        for j, z in enumerate(angles):
            ax = fig.add_axes([fx(xb0+j*(b+gb)), fy(row_y[r]), fx(b), fy(b)])
            n = board(ax, tag, z, RUN, LOC)
            present[L.OTHER] |= n['other'] > 0
            if r == 0:
                ax.set_title('θ = %g°' % z, fontsize=F_TITLE, pad=2.5)
        present[L.GAS] |= cfg != 2; present[L.TILE] |= cfg != 1
        if row_labels:
            fig.text(fx(PAD), fy(y_f+hf/2), row_labels[r], ha='left', va='center', fontsize=F_ROW, fontweight='bold')
    hd, lab = [], []
    for c, txt, col in ((L.GAS, 'gas', C_GAS), (L.TILE, 'aerogel', C_AERO), (L.OTHER, 'scattered', C_OTH)):
        if present[c]:
            hd.append(Patch(fc=col, ec='none')); lab.append(txt)
    hd.append(Patch(fc=C_MU, ec='none')); lab.append('muon')
    hd.append(Line2D([0], [0], marker='o', ls='', mfc='none', mec='#777777', mew=0.8, ms=5)); lab.append('expected ring')
    lg = fig.legend(hd, lab, loc='lower center', bbox_to_anchor=(fx(xb0+(4*b+3*gb)/2), 0.0), ncol=len(hd), fontsize=F_LEG,
                    frameon=False, handlelength=0.9, handleheight=0.9, columnspacing=0.8, handletextpad=0.4, borderaxespad=0.1)
    fig.canvas.draw()
    e = lg.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    assert e.x0 >= xb0-0.05 and e.x1 <= W-PAD/2, (name, 'legend', e.x0, e.x1, xb0, W)     # under the boards, inside the figure
    plotting.save(out, name+'.png', dpi=400, fig=fig)
    plt.close(fig)
    return name, W, H, b


JOBS = [(['c1'], 'event_displays_c1', None), (['c2'], 'event_displays_c2', None), (['c3'], 'event_displays_c3', None),
        (['c3_f76'], 'event_displays_c3_f76', None), (['c4_g50'], 'event_displays_c4_g50', None), (['c5'], 'event_displays_c5', None),
        (['c3', 'c3_f76'], 'event_displays_c3_mirrors', ['2"', '3"'])]
_CTX = {}


def job(a):
    with np.load(_CTX['inp']/'event_displays_locus.npz') as loc:     # opened per worker (a shared handle shares its offset)
        return make_figure(*a, _CTX['run'], loc, _CTX['out'])


def main():
    p = cli.parser(__doc__)
    p.add_argument('--input', type=Path, default=None, help='directory of event_displays.json and event_displays_locus.npz (default --outdir)')
    p.add_argument('--nproc', type=int, default=len(JOBS), help='worker processes, one figure each (default %(default)d)')
    args = p.parse_args(); out = cli.outdir(args); inp = args.input or out
    _CTX.update(run=json.load(open(inp/'event_displays.json')), inp=inp, out=out)
    with mp.get_context('fork').Pool(min(args.nproc, len(JOBS))) as pool:
        for r in pool.map(job, JOBS, chunksize=1):
            print('%s.png: %.3f x %.3f in, board %.3f in' % r, flush=True)


if __name__ == '__main__':
    main()
