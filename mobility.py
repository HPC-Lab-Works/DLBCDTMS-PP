"""Vectorised synthetic mobility models used by the evaluation.

All models return positions with shape (T+1, N, 2) in metres, sampled every dt=1 s.
  - manhattan: urban grid (Bai et al., IMPORTANT framework); streets every `block` m,
    turning probabilities 0.5 straight / 0.25 left / 0.25 right at intersections.
  - highway:   bidirectional highway, 3 lanes per direction, toroidal length.
  - rwp:       random waypoint with uniform pause time in [0, 5] s.
"""
import numpy as np


def _speeds(rng, n, vmin, vmax):
    return rng.uniform(vmin, vmax, n)


def manhattan(rng, n, T, L=2000.0, block=250.0, vmin=8.0, vmax=16.0, dt=1.0):
    nlines = int(L // block) + 1
    pos = np.zeros((T + 1, n, 2), dtype=np.float32)
    # place each vehicle on a random street
    horiz = rng.random(n) < 0.5
    line = rng.integers(0, nlines, n) * block
    along = rng.uniform(0, L, n)
    x = np.where(horiz, along, line); y = np.where(horiz, line, along)
    # direction unit vectors (dx, dy) in {(1,0),(-1,0),(0,1),(0,-1)}
    sgn = np.where(rng.random(n) < 0.5, 1.0, -1.0)
    dx = np.where(horiz, sgn, 0.0); dy = np.where(horiz, 0.0, sgn)
    vbase = _speeds(rng, n, vmin, vmax)
    pos[0, :, 0] = x; pos[0, :, 1] = y
    for t in range(1, T + 1):
        v = vbase * rng.uniform(0.9, 1.1, n) * dt
        # distance to next intersection along the direction of motion
        coord = np.where(dx != 0, x, y); d = np.where(dx != 0, dx, dy)
        nxt = np.where(d > 0, np.floor(coord / block + 1e-9) * block + block,
                       np.ceil(coord / block - 1e-9) * block - block)
        dist = np.abs(nxt - coord)
        cross = v >= dist
        # move vehicles that do not cross
        x = x + np.where(cross, 0, dx * v); y = y + np.where(cross, 0, dy * v)
        if cross.any():
            idx = np.where(cross)[0]
            # snap to the intersection
            xi = np.where(dx[idx] != 0, nxt[idx], x[idx]); yi = np.where(dx[idx] != 0, y[idx], nxt[idx])
            rem = v[idx] - dist[idx]
            u = rng.random(idx.size)
            odx, ody = dx[idx].copy(), dy[idx].copy()
            # straight / left / right
            ndx = np.where(u < 0.5, odx, np.where(u < 0.75, -ody, ody))
            ndy = np.where(u < 0.5, ody, np.where(u < 0.75, odx, -odx))
            # keep inside the area: reverse if heading outside
            out = ((xi + ndx * 1.0) < 0) | ((xi + ndx * 1.0) > L) | ((yi + ndy * 1.0) < 0) | ((yi + ndy * 1.0) > L)
            ndx = np.where(out, -odx, ndx); ndy = np.where(out, -ody, ndy)
            out2 = ((xi + ndx) < 0) | ((xi + ndx) > L) | ((yi + ndy) < 0) | ((yi + ndy) > L)
            ndx = np.where(out2, -ndx, ndx); ndy = np.where(out2, -ndy, ndy)
            x[idx] = np.clip(xi + ndx * rem, 0, L); y[idx] = np.clip(yi + ndy * rem, 0, L)
            dx[idx] = ndx; dy[idx] = ndy
        pos[t, :, 0] = x; pos[t, :, 1] = y
    return pos


def highway(rng, n, T, L=5000.0, lanes=3, vmin=22.0, vmax=33.0, dt=1.0):
    pos = np.zeros((T + 1, n, 2), dtype=np.float32)
    direction = np.where(rng.random(n) < 0.5, 1.0, -1.0)
    lane = rng.integers(0, lanes, n)
    y = np.where(direction > 0, 10.0 + 3.5 * lane, -10.0 - 3.5 * lane)
    x = rng.uniform(0, L, n)
    vbase = _speeds(rng, n, vmin, vmax)
    pos[0, :, 0] = x; pos[0, :, 1] = y
    for t in range(1, T + 1):
        x = (x + direction * vbase * rng.uniform(0.95, 1.05, n) * dt) % L
        pos[t, :, 0] = x; pos[t, :, 1] = y
    return pos


def rwp(rng, n, T, L=2000.0, vmin=5.0, vmax=20.0, pmax=5.0, dt=1.0):
    pos = np.zeros((T + 1, n, 2), dtype=np.float32)
    p = rng.uniform(0, L, (n, 2)); tgt = rng.uniform(0, L, (n, 2))
    v = _speeds(rng, n, vmin, vmax); pause = np.zeros(n)
    pos[0] = p
    for t in range(1, T + 1):
        d = tgt - p; dist = np.linalg.norm(d, axis=1) + 1e-9
        moving = pause <= 0
        step = np.minimum(v * dt, dist)
        p = p + np.where(moving[:, None], d / dist[:, None] * step[:, None], 0)
        arrived = moving & (step >= dist - 1e-6)
        if arrived.any():
            k = arrived.sum()
            tgt[arrived] = rng.uniform(0, L, (k, 2)); v[arrived] = rng.uniform(vmin, vmax, k)
            pause[arrived] = rng.uniform(0, pmax, k)
        pause = pause - dt * (~moving)
        pos[t] = p
    return pos


def rsu_positions(model, m, L=2000.0, Lh=5000.0):
    if model == "highway":
        xs = (np.arange(m) + 0.5) * Lh / m
        return np.stack([xs, np.zeros(m)], 1)
    k = int(round(np.sqrt(m)))
    g = (np.arange(k) + 0.5) * L / k
    gx, gy = np.meshgrid(g, g)
    return np.stack([gx.ravel(), gy.ravel()], 1)


def generate(model, rng, n, T, speed=None):
    if model == "manhattan":
        if speed is not None:
            return manhattan(rng, n, T, vmin=speed * 0.9, vmax=speed * 1.1)
        return manhattan(rng, n, T)
    if model == "highway":
        return highway(rng, n, T)
    if model == "rwp":
        return rwp(rng, n, T)
    raise ValueError(model)
