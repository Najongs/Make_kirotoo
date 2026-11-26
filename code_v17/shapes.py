"""Rounded profiles: polygons whose corners carry fillet radii, as (u, v, r) points."""
import math

ARC_STEPS = 8          # samples per fillet arc (same count for every arc -> offset loops stay index-aligned)


def _norm(x, y):
    n = math.hypot(x, y)
    return x / n, y / n


def area(pts):
    n = len(pts)
    return sum(pts[i][0] * pts[(i + 1) % n][1] - pts[(i + 1) % n][0] * pts[i][1] for i in range(n)) / 2


def orient(pts, ccw=True):
    """Return the (u,v,r) loop with the requested orientation."""
    a = area([(p[0], p[1]) for p in pts])
    return list(pts) if (a > 0) == ccw else list(pts)[::-1]


def segments(pts):
    """Closed loop of (u,v,r) -> list of segments.
    ('L', p0, p1) or ('A', centre, radius, p0, p1, ccw). Zero-length lines are dropped."""
    n = len(pts)
    corner = []
    for i in range(n):
        px, py, r = pts[i]
        ax, ay = pts[i - 1][0], pts[i - 1][1]
        bx, by = pts[(i + 1) % n][0], pts[(i + 1) % n][1]
        if r <= 0:
            corner.append(((px, py), (px, py), None))
            continue
        din = _norm(px - ax, py - ay)
        dout = _norm(bx - px, by - py)
        cosphi = max(-1.0, min(1.0, -(din[0] * dout[0] + din[1] * dout[1])))
        phi = math.acos(cosphi)                       # interior angle
        t = r / math.tan(phi / 2)
        lin, lout = math.hypot(px - ax, py - ay), math.hypot(bx - px, by - py)
        assert t <= min(lin, lout) / 2 + 1e-6 or t <= min(lin, lout) + 1e-6, ("fillet too large", pts[i], t)
        s = (px - din[0] * t, py - din[1] * t)
        e = (px + dout[0] * t, py + dout[1] * t)
        bis = _norm(dout[0] - din[0], dout[1] - din[1])
        d = r / math.sin(phi / 2)
        c = (px + bis[0] * d, py + bis[1] * d)
        ccw = (din[0] * dout[1] - din[1] * dout[0]) > 0
        corner.append((s, e, (c, r, ccw)))
    segs = []
    for i in range(n):
        s, e, arc = corner[i]
        prev_end = corner[i - 1][1]
        if math.hypot(s[0] - prev_end[0], s[1] - prev_end[1]) > 1e-6:
            segs.append(("L", prev_end, s))
        if arc:
            segs.append(("A", arc[0], arc[1], s, e, arc[2]))
    # rotate so that the loop starts with the first segment (order already closed)
    return segs


def arc_points(c, r, p0, p1, ccw, steps=ARC_STEPS):
    a0 = math.atan2(p0[1] - c[1], p0[0] - c[0])
    a1 = math.atan2(p1[1] - c[1], p1[0] - c[0])
    if ccw:
        while a1 <= a0:
            a1 += 2 * math.pi
    else:
        while a1 >= a0:
            a1 -= 2 * math.pi
    return [(c[0] + r * math.cos(a0 + (a1 - a0) * k / steps), c[1] + r * math.sin(a0 + (a1 - a0) * k / steps))
            for k in range(steps + 1)]


def sample(pts):
    """Polygon approximation of a rounded loop (for rendering and interference checks)."""
    out = []
    for sg in segments(pts):
        if sg[0] == "L":
            out.append(sg[1])
        else:
            out.extend(arc_points(sg[1], sg[2], sg[3], sg[4], sg[5])[:-1])
    return out


def rrect(u0, u1, v0, v1, r):
    r = min(r, (u1 - u0) / 2, (v1 - v0) / 2)
    return [(u0, v0, r), (u1, v0, r), (u1, v1, r), (u0, v1, r)]


def annulus_sector(cu, cv, r_in, r_out, a_start, a_end, n=None):
    """C-shaped ring (u,v,r loop) from angle a_start to a_end (deg, CCW), centred on (cu, cv).
    Built from polygon corners whose fillets all share the centre -> exact circular arcs."""
    a0, a1 = math.radians(a_start), math.radians(a_end)
    span = a1 - a0
    n = n or max(2, math.ceil(math.degrees(span) / 60))
    d = span / n
    pts = [(cu + r_out * math.cos(a0), cv + r_out * math.sin(a0), 0)]
    for k in range(n):
        a = a0 + (k + 0.5) * d
        R = r_out / math.cos(d / 2)
        pts.append((cu + R * math.cos(a), cv + R * math.sin(a), r_out))
    pts.append((cu + r_out * math.cos(a1), cv + r_out * math.sin(a1), 0))
    pts.append((cu + r_in * math.cos(a1), cv + r_in * math.sin(a1), 0))
    for k in reversed(range(n)):
        a = a0 + (k + 0.5) * d
        R = r_in / math.cos(d / 2)
        pts.append((cu + R * math.cos(a), cv + R * math.sin(a), r_in))
    pts.append((cu + r_in * math.cos(a0), cv + r_in * math.sin(a0), 0))
    return pts


def circle_loop(cu, cv, r):
    return rrect(cu - r, cu + r, cv - r, cv + r, r)


def circle_cap(cu, cv, r, a_start, a_end, n=None):
    """Circle segment: exact arc from a_start to a_end (deg, CCW, measured from +u toward +v) closed by its chord."""
    a0, a1 = math.radians(a_start), math.radians(a_end)
    span = a1 - a0
    n = n or max(2, math.ceil(math.degrees(span) / 60))
    d = span / n
    pts = [(cu + r * math.cos(a0), cv + r * math.sin(a0), 0)]
    for k in range(n):
        a = a0 + (k + 0.5) * d
        R = r / math.cos(d / 2)
        pts.append((cu + R * math.cos(a), cv + R * math.sin(a), r))
    pts.append((cu + r * math.cos(a1), cv + r * math.sin(a1), 0))
    return pts
