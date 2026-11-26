"""결과 표(마크다운): python robust/tables.py → robust/results/tables.md"""
import os, sys, json
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from rb_common import RES


def cell_walk(o):
    if "error" in o:
        return "오류", "", "", "", "", ""
    st = "성공" if not o["fallen"] else f"**{o['steps_done']}걸음에서 넘어짐**"
    if not o.get("u_p99"):
        return st, "", "", "", "", ""
    p5 = o["cop_margin_mm"]["p5"]; spk = o.get("swing_peak_mm") or {}; sw = spk.get("min"); smed = spk.get("median")
    ku = max(o["u_p99"]["R_knee"], o["u_p99"]["L_knee"]); ks = 100 * max(o["sat_frac"]["R_knee"], o["sat_frac"]["L_knee"])
    hop = o.get("hop_n", 0)
    return st, f"{p5:.1f}" if p5 is not None else "", f"{smed:.1f} / {sw:.1f}" if sw is not None else "", str(hop), f"{ku:.2f}", f"{ks:.1f}"


def cell_stand(o):
    if "error" in o:
        return "오류"
    return "버팀" if not o["fallen"] else f"**넘어짐 {o['fall_t']:.1f} s**"


def oat_tables():
    rows = json.load(open(os.path.join(RES, "oat.json")))["rows"]
    idx = {(r["meta"]["backpack"], r["meta"]["test"], r["meta"]["axis"], r["meta"]["val"]): r["res"] for r in rows}
    axes = []
    for r in rows:
        if r["meta"]["axis"] not in axes:
            axes.append(r["meta"]["axis"])
    out = []
    hdr = ("| 값 | 정지 자세 | 걷기 18걸음 | 압력중심 여유 하위 5 % (mm) | 흔드는 발 최고 높이 중앙값 / 최솟값 (mm) | 짧게 뜬 횟수 (0.03–0.3 s) | 무릎 사용률 99 % | 무릎 포화 (%) |\n"
           "|---|---|---|---|---|---|---|---|")
    for bp in (False, True):
        out.append(f"\n### 배낭 {'있음 (250 g, 등 뒤 x −117 · z 280 mm)' if bp else '없음 (지금 LB-012 + Jetson 2S)'}\n")
        for an in axes:
            vals = sorted({r["meta"]["val"] for r in rows if r["meta"]["axis"] == an and r["meta"]["backpack"] == bp},
                          key=lambda v: -1e9 if v is None else v)
            out.append(f"\n**{an}**\n\n" + hdr)
            for v in vals:
                s = idx.get((bp, "정지 자세", an, v)); w = idx.get((bp, "걷기 0.077 m/s", an, v))
                c = cell_walk(w) if w else ("",) * 6
                out.append(f"| {'기준' if v is None else v} | {cell_stand(s) if s else ''} | " + " | ".join(c) + " |")
    return "\n".join(out)


def combo_tables():
    p = os.path.join(RES, "combo.json")
    if not os.path.exists(p):
        return ""
    rows = json.load(open(p))["rows"]
    out = ["\n## 나쁜 쪽 조합\n", "| 배낭 | 무게중심 | 조합 | 정지 자세 | 걷기 18걸음 | 압력중심 여유 하위 5 % (mm) | 흔드는 발 최고 높이 중앙값 / 최솟값 (mm) | 짧게 뜬 횟수 (0.03–0.3 s) | 무릎 사용률 99 % | 무릎 포화 (%) |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    keys = []
    for r in rows:
        k = (r["meta"]["backpack"], r["meta"]["com_dx"], r["meta"]["combo"])
        if k not in keys:
            keys.append(k)
    idx = {(r["meta"]["backpack"], r["meta"]["com_dx"], r["meta"]["combo"], r["meta"]["test"]): r["res"] for r in rows}
    for k in sorted(keys, key=lambda k: (k[0], -k[1], k[2] != "전부")):
        s = idx.get(k + ("정지 자세",)); w = idx.get(k + ("걷기 0.077 m/s",))
        c = cell_walk(w) if w else ("",) * 6
        out.append(f"| {'있음' if k[0] else '없음'} | {k[1] * 1000:+.0f} mm | {k[2]} | {cell_stand(s) if s else ''} | " + " | ".join(c) + " |")
    return "\n".join(out)


def extra_tables():
    p = os.path.join(RES, "extra.json")
    if not os.path.exists(p):
        return ""
    rows = json.load(open(p))["rows"]
    out = ["\n## 추가 (무게중심 경계 · 제어기가 무게중심을 알 때 · 같은 설정 근처 흩어짐)\n",
           "| 배낭 | 항목 | 값 | 정지 자세 | 걷기 18걸음 | 압력중심 여유 하위 5 % (mm) | 흔드는 발 최고 높이 중앙값 / 최솟값 (mm) | 짧게 뜬 횟수 (0.03–0.3 s) | 무릎 사용률 99 % | 무릎 포화 (%) |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    idx = {}
    for r in rows:
        m_ = r["meta"]; idx.setdefault((m_["backpack"], m_["axis"], str(m_["val"])), {})[m_["test"]] = r["res"]
    for (bp, ax, v), d in sorted(idx.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2])):
        s_ = d.get("정지 자세"); w = d.get("걷기 0.077 m/s")
        c = cell_walk(w) if w else ("",) * 6
        out.append(f"| {'있음' if bp else '없음'} | {ax} | {v} | {cell_stand(s_) if s_ else ''} | " + " | ".join(c) + " |")
    return "\n".join(out)


if __name__ == "__main__":
    txt = "## 하나씩 바꾸기\n" + oat_tables() + "\n" + extra_tables() + "\n" + combo_tables() + "\n"
    open(os.path.join(RES, "tables.md"), "w").write(txt)
    print(txt)
