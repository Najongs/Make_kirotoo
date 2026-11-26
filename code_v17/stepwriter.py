"""Minimal STEP AP214 writer: box and cylinder B-rep solids, colours, multi-level assemblies."""
import math


def _f(v):
    s = repr(float(v))
    if "e" in s or "E" in s:
        s = "%.10f" % v
    if "." not in s:
        s += "."
    return s


def _vec(v):
    return "(" + ",".join(_f(c) for c in v) + ")"


def _norm(v):
    n = math.sqrt(sum(c * c for c in v))
    return tuple(c / n for c in v)


def _sub(a, b):
    return tuple(x - y for x, y in zip(a, b))


def _cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def _dot(a, b):
    return sum(x * y for x, y in zip(a, b))


class StepWriter:
    def __init__(self, name):
        self.name = name
        self.lines = []
        self.n = 0
        self.app = self.add("APPLICATION_CONTEXT('core data for automotive mechanical design processes')")
        self.add(f"APPLICATION_PROTOCOL_DEFINITION('international standard','automotive_design',2000,#{self.app})")
        self.pc = self.add(f"PRODUCT_CONTEXT('',#{self.app},'mechanical')")
        self.pdc = self.add(f"PRODUCT_DEFINITION_CONTEXT('part definition',#{self.app},'design')")
        u1 = self.add("( LENGTH_UNIT() NAMED_UNIT(*) SI_UNIT(.MILLI.,.METRE.) )")
        u2 = self.add("( NAMED_UNIT(*) PLANE_ANGLE_UNIT() SI_UNIT($,.RADIAN.) )")
        u3 = self.add("( NAMED_UNIT(*) SI_UNIT($,.STERADIAN.) SOLID_ANGLE_UNIT() )")
        unc = self.add(f"UNCERTAINTY_MEASURE_WITH_UNIT(LENGTH_MEASURE(1.E-07),#{u1},'distance_accuracy_value','confusion accuracy')")
        self.ctx = self.add(
            f"( GEOMETRIC_REPRESENTATION_CONTEXT(3) GLOBAL_UNCERTAINTY_ASSIGNED_CONTEXT((#{unc})) "
            f"GLOBAL_UNIT_ASSIGNED_CONTEXT((#{u1},#{u2},#{u3})) REPRESENTATION_CONTEXT('Context #1','3D Context with UNIT and UNCERTAINTY') )")
        self._colours = {}

    def add(self, text):
        self.n += 1
        self.lines.append(f"#{self.n}={text};")
        return self.n

    # ---------- basic geometry ----------
    def pt(self, p):
        return self.add(f"CARTESIAN_POINT('',{_vec(p)})")

    def dir(self, d):
        return self.add(f"DIRECTION('',{_vec(_norm(d))})")

    def axis(self, origin=(0, 0, 0), z=(0, 0, 1), x=(1, 0, 0)):
        return self.add(f"AXIS2_PLACEMENT_3D('',#{self.pt(origin)},#{self.dir(z)},#{self.dir(x)})")

    def box(self, x0, x1, y0, y1, z0, z1):
        xs, ys, zs = sorted((x0, x1)), sorted((y0, y1)), sorted((z0, z1))
        P = [(xs[i & 1], ys[(i >> 1) & 1], zs[(i >> 2) & 1]) for i in range(8)]
        V = [self.add(f"VERTEX_POINT('',#{self.pt(p)})") for p in P]
        E = {}
        for i in range(8):
            for b in (1, 2, 4):
                if not i & b:
                    j = i | b
                    d = _sub(P[j], P[i])
                    ln = math.sqrt(_dot(d, d))
                    vec = self.add(f"VECTOR('',#{self.dir(d)},{_f(ln)})")
                    line = self.add(f"LINE('',#{self.pt(P[i])},#{vec})")
                    E[(i, j)] = self.add(f"EDGE_CURVE('',#{V[i]},#{V[j]},#{line},.T.)")
        quads = [  # (vertex indices, outward normal)
            ([0, 2, 6, 4], (-1, 0, 0)), ([1, 3, 7, 5], (1, 0, 0)),
            ([0, 1, 5, 4], (0, -1, 0)), ([2, 3, 7, 6], (0, 1, 0)),
            ([0, 1, 3, 2], (0, 0, -1)), ([4, 5, 7, 6], (0, 0, 1)),
        ]
        faces = []
        for q, nrm in quads:
            n_est = _cross(_sub(P[q[1]], P[q[0]]), _sub(P[q[2]], P[q[1]]))
            if _dot(n_est, nrm) < 0:
                q = q[::-1]
            oes = []
            for k in range(4):
                a, b = q[k], q[(k + 1) % 4]
                if (a, b) in E:
                    oes.append(self.add(f"ORIENTED_EDGE('',*,*,#{E[(a, b)]},.T.)"))
                else:
                    oes.append(self.add(f"ORIENTED_EDGE('',*,*,#{E[(b, a)]},.F.)"))
            loop = self.add("EDGE_LOOP(''," + "(" + ",".join(f"#{o}" for o in oes) + "))")
            bound = self.add(f"FACE_OUTER_BOUND('',#{loop},.T.)")
            ref = (0, 0, 1) if nrm[2] == 0 else (1, 0, 0)
            plane = self.add(f"PLANE('',#{self.axis(P[q[0]], nrm, ref)})")
            faces.append(self.add(f"ADVANCED_FACE('',(#{bound}),#{plane},.T.)"))
        shell = self.add("CLOSED_SHELL('',(" + ",".join(f"#{f}" for f in faces) + "))")
        return self.add(f"MANIFOLD_SOLID_BREP('',#{shell})")

    def cylinder(self, c, a, r, h):
        a = _norm(a)
        ref = (0, 0, 1) if abs(a[2]) < 0.9 else (1, 0, 0)
        ref = _norm(_sub(ref, tuple(_dot(ref, a) * x for x in a)))
        top_c = tuple(ci + ai * h for ci, ai in zip(c, a))
        p0 = tuple(ci + r * ri for ci, ri in zip(c, ref))
        p1 = tuple(ci + r * ri for ci, ri in zip(top_c, ref))
        v0 = self.add(f"VERTEX_POINT('',#{self.pt(p0)})")
        v1 = self.add(f"VERTEX_POINT('',#{self.pt(p1)})")
        c0 = self.add(f"CIRCLE('',#{self.axis(c, a, ref)},{_f(r)})")
        c1 = self.add(f"CIRCLE('',#{self.axis(top_c, a, ref)},{_f(r)})")
        e0 = self.add(f"EDGE_CURVE('',#{v0},#{v0},#{c0},.T.)")
        e1 = self.add(f"EDGE_CURVE('',#{v1},#{v1},#{c1},.T.)")
        vec = self.add(f"VECTOR('',#{self.dir(a)},{_f(h)})")
        seam_l = self.add(f"LINE('',#{self.pt(p0)},#{vec})")
        seam = self.add(f"EDGE_CURVE('',#{v0},#{v1},#{seam_l},.T.)")
        # bottom face: outward normal -a  -> plane normal a, same_sense .F., circle reversed
        oe = self.add(f"ORIENTED_EDGE('',*,*,#{e0},.F.)")
        lp = self.add(f"EDGE_LOOP('',(#{oe}))")
        fb = self.add(f"FACE_OUTER_BOUND('',#{lp},.T.)")
        pl = self.add(f"PLANE('',#{self.axis(c, a, ref)})")
        f_bot = self.add(f"ADVANCED_FACE('',(#{fb}),#{pl},.F.)")
        # top face: outward normal +a
        oe = self.add(f"ORIENTED_EDGE('',*,*,#{e1},.T.)")
        lp = self.add(f"EDGE_LOOP('',(#{oe}))")
        fb = self.add(f"FACE_OUTER_BOUND('',#{lp},.T.)")
        pl = self.add(f"PLANE('',#{self.axis(top_c, a, ref)})")
        f_top = self.add(f"ADVANCED_FACE('',(#{fb}),#{pl},.T.)")
        # side face
        o1 = self.add(f"ORIENTED_EDGE('',*,*,#{e0},.T.)")
        o2 = self.add(f"ORIENTED_EDGE('',*,*,#{seam},.T.)")
        o3 = self.add(f"ORIENTED_EDGE('',*,*,#{e1},.F.)")
        o4 = self.add(f"ORIENTED_EDGE('',*,*,#{seam},.F.)")
        lp = self.add(f"EDGE_LOOP('',(#{o1},#{o2},#{o3},#{o4}))")
        fb = self.add(f"FACE_OUTER_BOUND('',#{lp},.T.)")
        cs = self.add(f"CYLINDRICAL_SURFACE('',#{self.axis(c, a, ref)},{_f(r)})")
        f_side = self.add(f"ADVANCED_FACE('',(#{fb}),#{cs},.T.)")
        shell = self.add(f"CLOSED_SHELL('',(#{f_bot},#{f_side},#{f_top}))")
        return self.add(f"MANIFOLD_SOLID_BREP('',#{shell})")

    # in-plane (u, v) axes for an extrusion along k, chosen so that u x v = +k
    UV = {"x": ((0, 1, 0), (0, 0, 1)), "y": ((0, 0, 1), (1, 0, 0)), "z": ((1, 0, 0), (0, 1, 0))}
    K = {"x": (1, 0, 0), "y": (0, 1, 0), "z": (0, 0, 1)}

    def prism(self, axis, a0, a1, poly, holes=()):
        """Extrude a CCW polygon (u,v) along +axis from a0 to a1, with circular through-holes (u,v,r)."""
        U, Vd = self.UV[axis]
        k = self.K[axis]
        if a1 < a0:
            a0, a1 = a1, a0
        # ensure CCW
        area = sum(poly[i][0] * poly[(i + 1) % len(poly)][1] - poly[(i + 1) % len(poly)][0] * poly[i][1]
                   for i in range(len(poly)))
        if area < 0:
            poly = poly[::-1]
        P3 = lambda u, v, a: tuple(u * U[i] + v * Vd[i] + a * k[i] for i in range(3))
        n = len(poly)
        pb = [P3(u, v, a0) for u, v in poly]
        pt_ = [P3(u, v, a1) for u, v in poly]
        vb = [self.add(f"VERTEX_POINT('',#{self.pt(p)})") for p in pb]
        vt = [self.add(f"VERTEX_POINT('',#{self.pt(p)})") for p in pt_]

        def seg(pa, pb_, va, vb_):
            d = _sub(pb_, pa)
            vec = self.add(f"VECTOR('',#{self.dir(d)},{_f(math.sqrt(_dot(d, d)))})")
            ln = self.add(f"LINE('',#{self.pt(pa)},#{vec})")
            return self.add(f"EDGE_CURVE('',#{va},#{vb_},#{ln},.T.)")

        eb = [seg(pb[i], pb[(i + 1) % n], vb[i], vb[(i + 1) % n]) for i in range(n)]
        et = [seg(pt_[i], pt_[(i + 1) % n], vt[i], vt[(i + 1) % n]) for i in range(n)]
        ev = [seg(pb[i], pt_[i], vb[i], vt[i]) for i in range(n)]
        oe = lambda e, s: self.add(f"ORIENTED_EDGE('',*,*,#{e},.{'T' if s else 'F'}.)")
        loop = lambda oes: self.add("EDGE_LOOP('',(" + ",".join(f"#{o}" for o in oes) + "))")
        faces = []
        # holes: circles at bottom and top, seam
        hole_data = []
        for (hu, hv, r) in holes:
            cb, ct = P3(hu, hv, a0), P3(hu, hv, a1)
            sb, st = P3(hu + r, hv, a0), P3(hu + r, hv, a1)
            v0 = self.add(f"VERTEX_POINT('',#{self.pt(sb)})")
            v1 = self.add(f"VERTEX_POINT('',#{self.pt(st)})")
            c0 = self.add(f"CIRCLE('',#{self.axis(cb, k, U)},{_f(r)})")
            c1 = self.add(f"CIRCLE('',#{self.axis(ct, k, U)},{_f(r)})")
            e0 = self.add(f"EDGE_CURVE('',#{v0},#{v0},#{c0},.T.)")
            e1 = self.add(f"EDGE_CURVE('',#{v1},#{v1},#{c1},.T.)")
            sm = seg(sb, st, v0, v1)
            hole_data.append((cb, r, e0, e1, sm))
        # top face (normal +k)
        cu = sum(p[0] for p in poly) / n
        cv = sum(p[1] for p in poly) / n
        bounds = [self.add(f"FACE_OUTER_BOUND('',#{loop([oe(e, True) for e in et])},.T.)")]
        bounds += [self.add(f"FACE_BOUND('',#{loop([oe(h[3], False)])},.T.)") for h in hole_data]
        pl = self.add(f"PLANE('',#{self.axis(P3(cu, cv, a1), k, U)})")
        faces.append(self.add("ADVANCED_FACE('',(" + ",".join(f"#{b}" for b in bounds) + f"),#{pl},.T.)"))
        # bottom face (normal -k)
        bounds = [self.add(f"FACE_OUTER_BOUND('',#{loop([oe(e, False) for e in eb[::-1]])},.T.)")]
        bounds += [self.add(f"FACE_BOUND('',#{loop([oe(h[2], True)])},.T.)") for h in hole_data]
        pl = self.add(f"PLANE('',#{self.axis(P3(cu, cv, a0), tuple(-c for c in k), U)})")
        faces.append(self.add("ADVANCED_FACE('',(" + ",".join(f"#{b}" for b in bounds) + f"),#{pl},.T.)"))
        # side faces
        for i in range(n):
            j = (i + 1) % n
            d = _sub(pb[j], pb[i])
            nrm = _cross(d, k)
            lp = loop([oe(eb[i], True), oe(ev[j], True), oe(et[i], False), oe(ev[i], False)])
            fb = self.add(f"FACE_OUTER_BOUND('',#{lp},.T.)")
            pl = self.add(f"PLANE('',#{self.axis(pb[i], nrm, k)})")
            faces.append(self.add(f"ADVANCED_FACE('',(#{fb}),#{pl},.T.)"))
        # hole faces (material outside the cylinder -> same_sense .F.)
        for cb, r, e0, e1, sm in hole_data:
            lp = loop([oe(sm, True), oe(e1, True), oe(sm, False), oe(e0, False)])
            fb = self.add(f"FACE_OUTER_BOUND('',#{lp},.T.)")
            cs = self.add(f"CYLINDRICAL_SURFACE('',#{self.axis(cb, k, U)},{_f(r)})")
            faces.append(self.add(f"ADVANCED_FACE('',(#{fb}),#{cs},.F.)"))
        shell = self.add("CLOSED_SHELL('',(" + ",".join(f"#{f}" for f in faces) + "))")
        return self.add(f"MANIFOLD_SOLID_BREP('',#{shell})")

    def rprism(self, axis, a0, a1, outer, inners=(), holes=()):
        """Extrude rounded loops along +axis. outer/inners: lists of (u, v, r) corners (r = fillet radius).
        Fillets become true cylindrical faces; circular through-holes as in prism()."""
        from shapes import segments, orient
        U, Vd = self.UV[axis]
        k = self.K[axis]
        if a1 < a0:
            a0, a1 = a1, a0
        P3 = lambda u, v, a: tuple(u * U[i] + v * Vd[i] + a * k[i] for i in range(3))
        oe = lambda e, s: self.add(f"ORIENTED_EDGE('',*,*,#{e},.{'T' if s else 'F'}.)")
        loop = lambda oes: self.add("EDGE_LOOP('',(" + ",".join(f"#{o}" for o in oes) + "))")

        def seg_edge(sg, a, va, vb):
            if sg[0] == "L":
                p0, p1 = P3(*sg[1], a), P3(*sg[2], a)
                d = _sub(p1, p0)
                vec = self.add(f"VECTOR('',#{self.dir(d)},{_f(math.sqrt(_dot(d, d)))})")
                ln = self.add(f"LINE('',#{self.pt(p0)},#{vec})")
                return self.add(f"EDGE_CURVE('',#{va},#{vb},#{ln},.T.)")
            c, r, ccw = sg[1], sg[2], sg[5]
            circ = self.add(f"CIRCLE('',#{self.axis(P3(c[0], c[1], a), k, U)},{_f(r)})")
            return self.add(f"EDGE_CURVE('',#{va},#{vb},#{circ},.{'T' if ccw else 'F'}.)")

        loops = [orient(outer, True)] + [orient(l, False) for l in inners]
        built = []   # per loop: (segs, bottom edges, top edges, vertical edges)
        for lp in loops:
            segs = segments(lp)
            starts = [sg[1] if sg[0] == "L" else sg[3] for sg in segs]
            vb = [self.add(f"VERTEX_POINT('',#{self.pt(P3(u, v, a0))})") for u, v in starts]
            vt = [self.add(f"VERTEX_POINT('',#{self.pt(P3(u, v, a1))})") for u, v in starts]
            n = len(segs)
            eb = [seg_edge(segs[i], a0, vb[i], vb[(i + 1) % n]) for i in range(n)]
            et = [seg_edge(segs[i], a1, vt[i], vt[(i + 1) % n]) for i in range(n)]
            ev = []
            for i in range(n):
                p0, p1 = P3(*starts[i], a0), P3(*starts[i], a1)
                vec = self.add(f"VECTOR('',#{self.dir(k)},{_f(a1 - a0)})")
                ln = self.add(f"LINE('',#{self.pt(p0)},#{vec})")
                ev.append(self.add(f"EDGE_CURVE('',#{vb[i]},#{vt[i]},#{ln},.T.)"))
            built.append((segs, eb, et, ev))
        hole_data = []
        for (hu, hv, r) in holes:
            cb, ct = P3(hu, hv, a0), P3(hu, hv, a1)
            v0 = self.add(f"VERTEX_POINT('',#{self.pt(P3(hu + r, hv, a0))})")
            v1 = self.add(f"VERTEX_POINT('',#{self.pt(P3(hu + r, hv, a1))})")
            c0 = self.add(f"CIRCLE('',#{self.axis(cb, k, U)},{_f(r)})")
            c1 = self.add(f"CIRCLE('',#{self.axis(ct, k, U)},{_f(r)})")
            e0 = self.add(f"EDGE_CURVE('',#{v0},#{v0},#{c0},.T.)")
            e1 = self.add(f"EDGE_CURVE('',#{v1},#{v1},#{c1},.T.)")
            vec = self.add(f"VECTOR('',#{self.dir(k)},{_f(a1 - a0)})")
            ln = self.add(f"LINE('',#{self.pt(P3(hu + r, hv, a0))},#{vec})")
            sm = self.add(f"EDGE_CURVE('',#{v0},#{v1},#{ln},.T.)")
            hole_data.append((cb, r, e0, e1, sm))
        faces = []
        cu, cv = outer[0][0], outer[0][1]
        # top face (+k): every loop in its traversal direction
        bounds = []
        for li, (segs, eb, et, ev) in enumerate(built):
            kind = "FACE_OUTER_BOUND" if li == 0 else "FACE_BOUND"
            bounds.append(self.add(f"{kind}('',#{loop([oe(e, True) for e in et])},.T.)"))
        bounds += [self.add(f"FACE_BOUND('',#{loop([oe(h[3], False)])},.T.)") for h in hole_data]
        pl = self.add(f"PLANE('',#{self.axis(P3(cu, cv, a1), k, U)})")
        faces.append(self.add("ADVANCED_FACE('',(" + ",".join(f"#{b}" for b in bounds) + f"),#{pl},.T.)"))
        # bottom face (-k): loops reversed
        bounds = []
        for li, (segs, eb, et, ev) in enumerate(built):
            kind = "FACE_OUTER_BOUND" if li == 0 else "FACE_BOUND"
            bounds.append(self.add(f"{kind}('',#{loop([oe(e, False) for e in eb[::-1]])},.T.)"))
        bounds += [self.add(f"FACE_BOUND('',#{loop([oe(h[2], True)])},.T.)") for h in hole_data]
        pl = self.add(f"PLANE('',#{self.axis(P3(cu, cv, a0), tuple(-c for c in k), U)})")
        faces.append(self.add("ADVANCED_FACE('',(" + ",".join(f"#{b}" for b in bounds) + f"),#{pl},.T.)"))
        # side faces
        for segs, eb, et, ev in built:
            n = len(segs)
            for i in range(n):
                j = (i + 1) % n
                lp = loop([oe(eb[i], True), oe(ev[j], True), oe(et[i], False), oe(ev[i], False)])
                fb = self.add(f"FACE_OUTER_BOUND('',#{lp},.T.)")
                sg = segs[i]
                if sg[0] == "L":
                    d = _sub(P3(*sg[2], a0), P3(*sg[1], a0))
                    surf = self.add(f"PLANE('',#{self.axis(P3(*sg[1], a0), _cross(d, k), k)})")
                    faces.append(self.add(f"ADVANCED_FACE('',(#{fb}),#{surf},.T.)"))
                else:
                    c, r, ccw = sg[1], sg[2], sg[5]
                    surf = self.add(f"CYLINDRICAL_SURFACE('',#{self.axis(P3(c[0], c[1], a0), k, U)},{_f(r)})")
                    faces.append(self.add(f"ADVANCED_FACE('',(#{fb}),#{surf},.{'T' if ccw else 'F'}.)"))
        for cb, r, e0, e1, sm in hole_data:
            lp = loop([oe(sm, True), oe(e1, True), oe(sm, False), oe(e0, False)])
            fb = self.add(f"FACE_OUTER_BOUND('',#{lp},.T.)")
            cs = self.add(f"CYLINDRICAL_SURFACE('',#{self.axis(cb, k, U)},{_f(r)})")
            faces.append(self.add(f"ADVANCED_FACE('',(#{fb}),#{cs},.F.)"))
        shell = self.add("CLOSED_SHELL('',(" + ",".join(f"#{f}" for f in faces) + "))")
        return self.add(f"MANIFOLD_SOLID_BREP('',#{shell})")

    def polyhedron(self, P, faces):
        """Closed planar-faced solid: P vertices, faces = outward-oriented vertex index loops (planar)."""
        V = {}
        def vtx(i):
            if i not in V:
                V[i] = self.add(f"VERTEX_POINT('',#{self.pt(P[i])})")
            return V[i]
        E = {}
        def edge(a, b):
            if (a, b) in E:
                return E[(a, b)], ".T."
            if (b, a) in E:
                return E[(b, a)], ".F."
            d = _sub(P[b], P[a])
            ln = math.sqrt(_dot(d, d))
            vec = self.add(f"VECTOR('',#{self.dir(d)},{_f(ln)})")
            line = self.add(f"LINE('',#{self.pt(P[a])},#{vec})")
            E[(a, b)] = self.add(f"EDGE_CURVE('',#{vtx(a)},#{vtx(b)},#{line},.T.)")
            return E[(a, b)], ".T."
        fs = []
        for f in faces:
            oes = []
            for k in range(len(f)):
                e, sense = edge(f[k], f[(k + 1) % len(f)])
                oes.append(self.add(f"ORIENTED_EDGE('',*,*,#{e},{sense})"))
            loop = self.add("EDGE_LOOP(''," + "(" + ",".join(f"#{o}" for o in oes) + "))")
            bound = self.add(f"FACE_OUTER_BOUND('',#{loop},.T.)")
            n = (0.0, 0.0, 0.0)
            for k in range(len(f)):
                n = tuple(a + b for a, b in zip(n, _cross(P[f[k]], P[f[(k + 1) % len(f)]])))
            n = _norm(n)
            ref = _sub(P[f[1]], P[f[0]])
            ref = _norm(_sub(ref, tuple(_dot(ref, n) * x for x in n)))
            plane = self.add(f"PLANE('',#{self.axis(P[f[0]], n, ref)})")
            fs.append(self.add(f"ADVANCED_FACE('',(#{bound}),#{plane},.T.)"))
        shell = self.add("CLOSED_SHELL('',(" + ",".join(f"#{f}" for f in fs) + "))")
        return self.add(f"MANIFOLD_SOLID_BREP('',#{shell})")

    def loft(self, axis, levels, closed):
        import loft as _l
        V, F = _l.mesh((axis, levels, closed))
        return self.polyhedron([tuple(float(c) for c in v) for v in V], F)

    def sphere_cap(self, c, a, r_out, r_in, d):
        """Hollow spherical cap: shell between radii r_in..r_out on the side a.(p-c) >= d of a cutting plane.
        Faces: outer sphere (one circular loop), inner sphere (one loop), planar annulus at the cut."""
        a = _norm(a)
        ref = (0, 0, 1) if abs(a[2]) < 0.9 else (1, 0, 0)
        ref = _norm(_sub(ref, tuple(_dot(ref, a) * x for x in a)))
        pc = tuple(ci + d * ai for ci, ai in zip(c, a))
        ro, ri = math.sqrt(r_out ** 2 - d ** 2), math.sqrt(r_in ** 2 - d ** 2)
        vo = self.add(f"VERTEX_POINT('',#{self.pt(tuple(p + ro * q for p, q in zip(pc, ref)))})")
        vi = self.add(f"VERTEX_POINT('',#{self.pt(tuple(p + ri * q for p, q in zip(pc, ref)))})")
        co = self.add(f"CIRCLE('',#{self.axis(pc, a, ref)},{_f(ro)})")
        ci_ = self.add(f"CIRCLE('',#{self.axis(pc, a, ref)},{_f(ri)})")
        eo = self.add(f"EDGE_CURVE('',#{vo},#{vo},#{co},.T.)")
        ei = self.add(f"EDGE_CURVE('',#{vi},#{vi},#{ci_},.T.)")
        oe = lambda e, s: self.add(f"ORIENTED_EDGE('',*,*,#{e},.{'T' if s else 'F'}.)")
        lp = lambda e, s: self.add(f"EDGE_LOOP('',(#{oe(e, s)}))")
        so = self.add(f"SPHERICAL_SURFACE('',#{self.axis(c, a, ref)},{_f(r_out)})")
        si = self.add(f"SPHERICAL_SURFACE('',#{self.axis(c, a, ref)},{_f(r_in)})")
        f_out = self.add(f"ADVANCED_FACE('',(#{self.add(f'FACE_OUTER_BOUND({chr(39)}{chr(39)},#{lp(eo, True)},.T.)')}),#{so},.T.)")
        f_in = self.add(f"ADVANCED_FACE('',(#{self.add(f'FACE_OUTER_BOUND({chr(39)}{chr(39)},#{lp(ei, False)},.T.)')}),#{si},.F.)")
        b1 = self.add(f"FACE_OUTER_BOUND('',#{lp(eo, False)},.T.)")
        b2 = self.add(f"FACE_BOUND('',#{lp(ei, True)},.T.)")
        pl = self.add(f"PLANE('',#{self.axis(pc, tuple(-x for x in a), ref)})")
        f_cut = self.add(f"ADVANCED_FACE('',(#{b1},#{b2}),#{pl},.T.)")
        shell = self.add(f"CLOSED_SHELL('',(#{f_out},#{f_in},#{f_cut}))")
        return self.add(f"MANIFOLD_SOLID_BREP('',#{shell})")

    def _style(self, rgb):
        if rgb not in self._colours:
            col = self.add(f"COLOUR_RGB('',{_f(rgb[0])},{_f(rgb[1])},{_f(rgb[2])})")
            fsc = self.add(f"FILL_AREA_STYLE_COLOUR('',#{col})")
            fas = self.add(f"FILL_AREA_STYLE('',(#{fsc}))")
            ssfa = self.add(f"SURFACE_STYLE_FILL_AREA(#{fas})")
            sss = self.add(f"SURFACE_SIDE_STYLE('',(#{ssfa}))")
            ssu = self.add(f"SURFACE_STYLE_USAGE(.BOTH.,#{sss})")
            self._colours[rgb] = self.add(f"PRESENTATION_STYLE_ASSIGNMENT((#{ssu}))")
        return self._colours[rgb]

    # ---------- product structure ----------
    def _product(self, name):
        p = self.add(f"PRODUCT('{name}','{name}','',(#{self.pc}))")
        pdf = self.add(f"PRODUCT_DEFINITION_FORMATION('','',#{p})")
        pd = self.add(f"PRODUCT_DEFINITION('design','',#{pdf},#{self.pdc})")
        self.add(f"PRODUCT_RELATED_PRODUCT_CATEGORY('part',$,(#{p}))")
        pds = self.add(f"PRODUCT_DEFINITION_SHAPE('','',#{pd})")
        return pd, pds

    def part(self, name, solids):
        """solids: list of (kind, args, rgb). Returns handle dict."""
        pd, pds = self._product(name)
        origin = self.axis()
        breps, styled = [], []
        for kind, args, rgb in solids:
            b = {"box": self.box, "cyl": self.cylinder, "prism": self.prism, "rprism": self.rprism,
                 "scap": self.sphere_cap, "loft": self.loft}[kind](*args)
            breps.append(b)
            styled.append(self.add(f"STYLED_ITEM('color',(#{self._style(rgb)}),#{b})"))
        rep = self.add("ADVANCED_BREP_SHAPE_REPRESENTATION('" + name + "',(" +
                       ",".join(f"#{i}" for i in [origin] + breps) + f"),#{self.ctx})")
        self.add(f"SHAPE_DEFINITION_REPRESENTATION(#{pds},#{rep})")
        self.add("MECHANICAL_DESIGN_GEOMETRIC_PRESENTATION_REPRESENTATION(''," + "(" +
                 ",".join(f"#{s}" for s in styled) + f"),#{self.ctx})")
        return {"pd": pd, "rep": rep, "origin": origin, "name": name}

    def assembly(self, name, children):
        """children: list of (handle, translation). Returns handle dict."""
        pd, pds = self._product(name)
        origin = self.axis()
        places = [self.axis(t) for _, t in children]
        rep = self.add("SHAPE_REPRESENTATION('" + name + "',(" +
                       ",".join(f"#{i}" for i in [origin] + places) + f"),#{self.ctx})")
        self.add(f"SHAPE_DEFINITION_REPRESENTATION(#{pds},#{rep})")
        for k, ((child, _), place) in enumerate(zip(children, places)):
            nauo = self.add(f"NEXT_ASSEMBLY_USAGE_OCCURRENCE('{name}_{k + 1}','{child['name']}','',#{pd},#{child['pd']},$)")
            ppds = self.add(f"PRODUCT_DEFINITION_SHAPE('Placement','Placement of an item',#{nauo})")
            idt = self.add(f"ITEM_DEFINED_TRANSFORMATION('','',#{child['origin']},#{place})")
            rr = self.add(f"( REPRESENTATION_RELATIONSHIP('','',#{child['rep']},#{rep}) "
                          f"REPRESENTATION_RELATIONSHIP_WITH_TRANSFORMATION(#{idt}) SHAPE_REPRESENTATION_RELATIONSHIP() )")
            self.add(f"CONTEXT_DEPENDENT_SHAPE_REPRESENTATION(#{rr},#{ppds})")
        return {"pd": pd, "rep": rep, "origin": origin, "name": name}

    def text(self):
        head = ("ISO-10303-21;\nHEADER;\n"
                "FILE_DESCRIPTION(('biped robot concept assembly'),'2;1');\n"
                f"FILE_NAME('{self.name}','2026-11-25T00:00:00',(''),(''),'python stepwriter','','');\n"
                "FILE_SCHEMA(('AUTOMOTIVE_DESIGN { 1 0 10303 214 1 1 1 1 }'));\nENDSEC;\nDATA;\n")
        return head + "\n".join(self.lines) + "\nENDSEC;\nEND-ISO-10303-21;\n"
