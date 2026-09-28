"""
PCG 레벨 레이아웃 생성기 (단일 파일 Gradio 웹 앱)

필수 규칙만 강제하고 나머지 형태는 자유롭게:
  - 길은 spawn에서 시작해 goal에서 끝나는 하나의 이어진 선
  - 모든 영역은 길 위, 전투 개수는 요청대로, 값은 범위 안
  - 길의 방향·모양·점 개수·시작/끝 위치는 자유

필요 패키지:  pip install gradio openai matplotlib
실행:        python app.py
접속:        ssh -J ... -L 7860:localhost:7860 ...  →  브라우저 http://localhost:7860

환경변수(선택): VLLM_URL (기본 http://localhost:8000/v1), MODEL (기본: 떠 있는 모델 자동 감지)
"""
import copy
import json
import math
import os
import random
import tempfile
import uuid
from datetime import datetime

import gradio as gr
from matplotlib.figure import Figure
from matplotlib.patches import Circle
from openai import OpenAI

# ================================================================ 설정
VLLM_URL = os.environ.get("VLLM_URL", "http://localhost:8000/v1")
LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "qa_log.jsonl")
client = OpenAI(base_url=VLLM_URL, api_key="EMPTY")


def detect_model():
    if os.environ.get("MODEL"):
        return os.environ["MODEL"]
    try:
        return client.models.list().data[0].id
    except Exception as e:
        raise SystemExit(f"vLLM 서버({VLLM_URL})에 연결할 수 없습니다. 서버가 떠 있는지 확인하세요.\n{e}")


MODEL = detect_model()

# 범위 상수
MAP_MIN, MAP_MAX = 0.03, 0.97
R_MIN, R_MAX = 0.03, 0.15
W_MIN, W_MAX = 200, 1500
MIN_SPAWN_GOAL = 0.45   # spawn-goal 최소 직선거리 (맵 한 변 대비)
COMBAT_GAP_MAX = 0.12      # 전투 원 가장자리가 길에서 떨어질 수 있는 최대 거리 (필수 규칙)
OFF_PATH_RATIO = (0.2, 0.7)
MIN_PATH_GAP = 0.05        # 길의 서로 다른 구간 사이 최소 간격 (교차 금지 포함)
MIN_TURN_ANGLE = 25        # 길이 이보다 급하게 꺾여 되돌아가면(헤어핀) 겹친 것으로 봄 (도)  # 맵마다 길 옆으로 빠지는 전투 비율을 이 범위에서 랜덤으로

# ================================================================ 스키마
POINT = {
    "type": "object",
    "properties": {"x": {"type": "number"}, "y": {"type": "number"}},
    "required": ["x", "y"],
    "additionalProperties": False,
}

SCHEMA = {
    "type": "object",
    "properties": {
        "theme": {"type": "string", "enum": ["forest", "grassland", "desert", "snow"]},
        "base_environment": {
            "type": "object",
            "properties": {
                "tree_density": {"type": "number"},
                "rock_density": {"type": "number"},
                "grass_density": {"type": "number"},
            },
            "required": ["tree_density", "rock_density", "grass_density"],
            "additionalProperties": False,
        },
        "main_path": {
            "type": "object",
            "properties": {
                "path_type": {"type": "string", "enum": ["main"]},
                "path_width": {"type": "number"},
                "normalized_points": {"type": "array", "items": POINT, "minItems": 2, "maxItems": 10},
            },
            "required": ["path_type", "path_width", "normalized_points"],
            "additionalProperties": False,
        },
        "combat_count": {"type": "integer"},
        "areas": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "area_type": {"type": "string", "enum": ["spawn", "combat", "goal"]},
                    "normalized_center": POINT,
                    "normalized_radius": {"type": "number"},
                    "detail_density": {"type": "number"},
                },
                "required": ["area_type", "normalized_center", "normalized_radius", "detail_density"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["theme", "base_environment", "main_path", "combat_count", "areas"],
    "additionalProperties": False,
}

# ================================================================ 프롬프트
# 형식 참고용 예시 2개. 방향과 모양이 서로 다르게 만들어 한쪽으로 끌리지 않게 한다.
EXAMPLES = [
    {
        "theme": "forest",
        "base_environment": {"tree_density": 0.75, "rock_density": 0.2, "grass_density": 0.5},
        "main_path": {
            "path_type": "main", "path_width": 450,
            "normalized_points": [
                {"x": 0.08, "y": 0.62}, {"x": 0.27, "y": 0.55}, {"x": 0.52, "y": 0.41},
                {"x": 0.78, "y": 0.58}, {"x": 0.92, "y": 0.47},
            ],
        },
        "combat_count": 1,
        "areas": [
            {"area_type": "spawn", "normalized_center": {"x": 0.08, "y": 0.62}, "normalized_radius": 0.05, "detail_density": 0.2},
            {"area_type": "combat", "normalized_center": {"x": 0.40, "y": 0.48}, "normalized_radius": 0.09, "detail_density": 0.6},
            {"area_type": "goal", "normalized_center": {"x": 0.92, "y": 0.47}, "normalized_radius": 0.06, "detail_density": 0.4},
        ],
    },
    {
        "theme": "desert",
        "base_environment": {"tree_density": 0.05, "rock_density": 0.65, "grass_density": 0.1},
        "main_path": {
            "path_type": "main", "path_width": 800,
            "normalized_points": [
                {"x": 0.50, "y": 0.92}, {"x": 0.58, "y": 0.70}, {"x": 0.40, "y": 0.50},
                {"x": 0.24, "y": 0.34}, {"x": 0.32, "y": 0.16}, {"x": 0.14, "y": 0.08},
            ],
        },
        "combat_count": 2,
        "areas": [
            {"area_type": "spawn", "normalized_center": {"x": 0.50, "y": 0.92}, "normalized_radius": 0.04, "detail_density": 0.2},
            {"area_type": "combat", "normalized_center": {"x": 0.40, "y": 0.50}, "normalized_radius": 0.05, "detail_density": 0.4},
            {"area_type": "combat", "normalized_center": {"x": 0.28, "y": 0.25}, "normalized_radius": 0.12, "detail_density": 0.8},
            {"area_type": "goal", "normalized_center": {"x": 0.14, "y": 0.08}, "normalized_radius": 0.06, "detail_density": 0.5},
        ],
    },
]

SYSTEM = f"""너는 사용자의 자연어 설명을 언리얼 PCG용 레벨 레이아웃 JSON으로 변환한다.

좌표계: 맵 전체를 0~1로 정규화한 2D 평면. x는 왼쪽(0)→오른쪽(1), y는 위(0)→아래(1).

[반드시 지킬 규칙]
- main_path는 spawn에서 시작해 goal에서 끝나는 하나의 이어진 길이다.
  normalized_points의 첫 점이 spawn 중심, 마지막 점이 goal 중심이다.
- spawn과 goal은 직선거리로 {MIN_SPAWN_GOAL} 이상 떨어뜨린다.
- 길은 자기 자신과 교차하지 않는다. 서로 다른 구간끼리 {MIN_PATH_GAP} 이상 떨어지고, 제자리로 확 꺾여 되돌아가지 않는다.
- 모든 좌표는 {MAP_MIN}~{MAP_MAX} 사이.
- combat_count에 사용자가 요청한 전투 횟수를 먼저 적는다. "3회", "세 번", "전투 3개"는 모두 3.
  언급이 없으면 1, "전투 없이"면 0.
- areas에는 spawn 1개, goal 1개, combat은 정확히 combat_count개.
__COMBAT_RULE__
- 전투끼리는 겹치지 않게 떨어뜨린다.
- density와 detail_density는 0.0~1.0, normalized_radius는 {R_MIN}~{R_MAX}, path_width는 {W_MIN}~{W_MAX}(cm).

[자유롭게 정할 것]
- 길의 방향, 모양, 점 개수(2~10개), 시작과 끝의 위치는 맵 어디든 자유다. 왼쪽→오른쪽일 필요 없다.
- 요청이 같아도 매번 새로운 레이아웃을 만든다. 뻔한 배치나 예시의 모양을 반복하지 않는다.
- 전투의 위치 간격, 크기 차이, 밀도 값도 요청에 어긋나지 않는 선에서 자유롭게 정한다.

[말 해석 참고]
- 곧은 길: 점이 적고 방향 변화가 작다. 구불구불한 길: 점이 많고 방향이 자주 바뀐다.
- 넓은 길 path_width 900 이상, 좁은 길 400 이하.
- 크기: 작은 영역 radius 0.03~0.06, 보통 0.06~0.09, 큰 0.10~0.15.
- 전투 강도: 약함 detail_density 0.3~0.4, 보통 0.5~0.6, 격함 0.7~0.9.
- 테마 기본 성향: forest 나무 많음, grassland 나무 적고 풀 많음, desert 풀 거의 없고 바위 많음, snow 풀 거의 없음.

[형식 예시] 형식만 참고하고 값과 모양은 따라하지 않는다.
{json.dumps(EXAMPLES[0], ensure_ascii=False)}
{json.dumps(EXAMPLES[1], ensure_ascii=False)}

사용자가 기존 맵의 수정을 요청하면, 현재 JSON에서 요청된 부분만 바꾸고 나머지는 유지한 전체 JSON을 출력한다.
JSON만 출력한다."""

def build_system(gap):
    if gap <= 0:
        rule = "- 모든 combat 중심은 길 위(길과의 거리가 normalized_radius 이내)에 둔다."
    else:
        rule = (f"- combat은 길을 따라 원하는 지점에 둔다. 길 위에 걸쳐도 되고 길 옆에 둬도 되지만, "
                f"combat 원의 가장자리가 길에서 {gap} 넘게 떨어지면 안 된다.")
    return SYSTEM.replace("__COMBAT_RULE__", rule)


# (질문, 입력 예시, 관련 필드, 답이 비었을 때 안내)
QUESTIONS = [
    ("1. 어떤 환경의 맵인가요? 테마(숲/초원/사막/설원)와 나무·바위·풀의 양",
     "예) 나무가 빽빽한 숲, 풀은 적당히, 바위는 거의 없음",
     "theme, base_environment", "테마와 밀도는 자유롭게 정해라."),
    ("2. 메인 길은 어떤 모양인가요? 곧은지/구불구불한지, 넓은지/좁은지",
     "예) 좁고 많이 구불구불한 길",
     "main_path", "길 모양은 자유롭게 정해라."),
    ("3. 시작부터 도착까지 어떤 흐름인가요? 전투 횟수, 위치, 규모",
     "예) 초반은 조용하고, 중반에 작은 전투, 도착 직전에 큰 전투",
     "combat_count, areas", "전투는 1개로 하고 배치는 자유롭게 정해라."),
]

# ================================================================ 기하 유틸
def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


def _closest_on_seg(p, a, b):
    dx, dy = b["x"] - a["x"], b["y"] - a["y"]
    L2 = dx * dx + dy * dy
    t = 0.0 if L2 == 0 else _clamp(((p["x"] - a["x"]) * dx + (p["y"] - a["y"]) * dy) / L2, 0.0, 1.0)
    return {"x": a["x"] + t * dx, "y": a["y"] + t * dy}, t


def _dist(a, b):
    return math.hypot(a["x"] - b["x"], a["y"] - b["y"])


def _seg_lengths(pts):
    return [_dist(pts[i], pts[i + 1]) for i in range(len(pts) - 1)]


def dist_to_path(p, pts):
    return min(_dist(p, _closest_on_seg(p, pts[i], pts[i + 1])[0]) for i in range(len(pts) - 1))


def closest_on_path(p, pts):
    return min((_closest_on_seg(p, pts[i], pts[i + 1])[0] for i in range(len(pts) - 1)), key=lambda q: _dist(p, q))


def progress_of(p, pts):
    """점 p를 길에 투영했을 때 전체 길이 대비 진행도(0~1)"""
    segs = _seg_lengths(pts)
    total = sum(segs) or 1.0
    best, best_d, acc = 0.0, 1e9, 0.0
    for i, L in enumerate(segs):
        q, t = _closest_on_seg(p, pts[i], pts[i + 1])
        d = _dist(p, q)
        if d < best_d:
            best_d, best = d, (acc + t * L) / total
        acc += L
    return best


def point_at_progress(pts, t):
    segs = _seg_lengths(pts)
    target, acc = _clamp(t, 0.0, 1.0) * sum(segs), 0.0
    for i, L in enumerate(segs):
        if acc + L >= target and L > 0:
            u = (target - acc) / L
            return {"x": round(pts[i]["x"] + u * (pts[i + 1]["x"] - pts[i]["x"]), 3),
                    "y": round(pts[i]["y"] + u * (pts[i + 1]["y"] - pts[i]["y"]), 3)}
        acc += L
    return dict(pts[-1])

def tangent_at_progress(pts, t):
    segs = _seg_lengths(pts)
    target, acc = _clamp(t, 0.0, 1.0) * sum(segs), 0.0
    last = None
    for i, L in enumerate(segs):
        if L > 0:
            last = ((pts[i + 1]["x"] - pts[i]["x"]) / L, (pts[i + 1]["y"] - pts[i]["y"]) / L)
            if acc + L >= target:
                return last
        acc += L
    return last or (1.0, 0.0)


def path_coords(p, pts):
    """점 p → (길 진행도 t, 길 기준 좌우 오프셋)"""
    t = progress_of(p, pts)
    q = point_at_progress(pts, t)
    tx, ty = tangent_at_progress(pts, t)
    return t, (p["x"] - q["x"]) * (-ty) + (p["y"] - q["y"]) * tx


def place_on_path(pts, t, off):
    """(진행도, 오프셋) → 좌표"""
    q = point_at_progress(pts, t)
    tx, ty = tangent_at_progress(pts, t)
    return {"x": round(_clamp(q["x"] - ty * off, MAP_MIN, MAP_MAX), 3),
            "y": round(_clamp(q["y"] + tx * off, MAP_MIN, MAP_MAX), 3)}


# ================================================================ 길 자기 겹침
def _orient(a, b, c):
    return (b["x"] - a["x"]) * (c["y"] - a["y"]) - (b["y"] - a["y"]) * (c["x"] - a["x"])


def _segs_cross(a, b, c, d):
    return (_orient(a, b, c) * _orient(a, b, d) < 0) and (_orient(c, d, a) * _orient(c, d, b) < 0)


def _seg_seg_dist(a, b, c, d):
    if _segs_cross(a, b, c, d):
        return 0.0
    return min(_dist(a, _closest_on_seg(a, c, d)[0]), _dist(b, _closest_on_seg(b, c, d)[0]),
               _dist(c, _closest_on_seg(c, a, b)[0]), _dist(d, _closest_on_seg(d, a, b)[0]))


def _turn_angle(a, b, c):
    """b에서의 사잇각(도). 180이면 직진, 0에 가까우면 제자리로 되돌아감"""
    v1, v2 = (a["x"] - b["x"], a["y"] - b["y"]), (c["x"] - b["x"], c["y"] - b["y"])
    n1, n2 = math.hypot(*v1), math.hypot(*v2)
    if n1 == 0 or n2 == 0:
        return 180.0
    cos = _clamp((v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2), -1.0, 1.0)
    return math.degrees(math.acos(cos))


def path_overlaps(pts):
    """겹치는 곳 목록: ('cross'|'near', i, j) 또는 ('hairpin', i, i+1)"""
    out = []
    n = len(pts) - 1
    for i in range(n):
        for j in range(i + 2, n):
            dd = _seg_seg_dist(pts[i], pts[i + 1], pts[j], pts[j + 1])
            if dd < MIN_PATH_GAP:
                out.append(("cross" if dd == 0 else "near", i, j))
    for i in range(1, len(pts) - 1):
        if _turn_angle(pts[i - 1], pts[i], pts[i + 1]) < MIN_TURN_ANGLE:
            out.append(("hairpin", i - 1, i))
    return out


def untangle_path(pts):
    """양 끝점(spawn, goal)은 고정하고 길의 자기 겹침을 없앤다. 보정 내역 반환."""
    notes = []
    if not path_overlaps(pts):
        return notes

    # 1단계: 교차하는 두 구간 사이를 뒤집어 매듭 풀기 (2-opt)
    for _ in range(30):
        cross = next((o for o in path_overlaps(pts) if o[0] == "cross"), None)
        if cross is None:
            break
        _, i, j = cross
        pts[i + 1:j + 1] = pts[i + 1:j + 1][::-1]
        notes.append("교차하는 길 구간의 순서를 바꿔 매듭을 풂")
    if not path_overlaps(pts):
        return notes[:1]

    # 2단계: 너무 가깝거나 급하게 꺾이는 곳의 중간 점을 하나씩 제거
    for _ in range(len(pts)):
        ov = path_overlaps(pts)
        if not ov or len(pts) <= 3:
            break
        _, i, j = ov[0]
        cand = [k for k in (i + 1, j, j + 1, i) if 0 < k < len(pts) - 1]
        best = min(cand, key=lambda k: len(path_overlaps(pts[:k] + pts[k + 1:])))
        del pts[best]
        notes.append("겹치는 곳의 길 점을 제거")
    if not path_overlaps(pts):
        return list(dict.fromkeys(notes))

    # 3단계(최후): 시작→끝 방향으로 한쪽으로만 나아가는 길로 정리 (구조적으로 겹칠 수 없음)
    start, end = pts[0], pts[-1]
    L = _dist(start, end) or 1.0
    ux, uy = (end["x"] - start["x"]) / L, (end["y"] - start["y"]) / L
    inner = sorted(pts[1:-1], key=lambda p: (p["x"] - start["x"]) * ux + (p["y"] - start["y"]) * uy)
    mono, last = [start], 0.0
    for p in inner:
        proj = (p["x"] - start["x"]) * ux + (p["y"] - start["y"]) * uy
        if proj - last >= MIN_PATH_GAP and proj <= L - MIN_PATH_GAP:
            mono.append(p)
            last = proj
    mono.append(end)
    while path_overlaps(mono) and len(mono) > 2:
        del mono[1 + max(range(len(mono) - 2), key=lambda k: 180 - _turn_angle(mono[k], mono[k + 1], mono[k + 2]))]
    if len(mono) < 3:  # 가운데 점 하나로 완만하게
        mono.insert(1, {"x": round((start["x"] + end["x"]) / 2 - uy * 0.08, 3),
                        "y": round((start["y"] + end["y"]) / 2 + ux * 0.08, 3)})
    pts[:] = mono
    notes.append("길이 계속 겹쳐서 시작→끝 방향으로 나아가는 형태로 정리")
    return list(dict.fromkeys(notes))


# ================================================================ 검증 (필수 규칙만)
def validate(d, gap=COMBAT_GAP_MAX):
    errs = []
    for k, v in d["base_environment"].items():
        if not 0 <= v <= 1:
            errs.append(f"{k}={v} 범위 밖 (0~1)")

    path = d["main_path"]
    pts = path["normalized_points"]
    if not W_MIN <= path["path_width"] <= W_MAX:
        errs.append(f"path_width={path['path_width']} 범위 밖 ({W_MIN}~{W_MAX})")
    if len(pts) < 2:
        return errs + ["경로 점이 2개 미만"]
    for p in pts:
        if not (MAP_MIN <= p["x"] <= MAP_MAX and MAP_MIN <= p["y"] <= MAP_MAX):
            errs.append(f"경로 점 ({p['x']:.2f}, {p['y']:.2f}) 맵 밖")
    ov = path_overlaps(pts)
    if ov:
        kinds = {"cross": "교차", "near": "너무 가까움", "hairpin": "급하게 되돌아감"}
        errs.append("길이 스스로 겹침: " + ", ".join(sorted({kinds[o[0]] for o in ov})) + f" ({len(ov)}곳)")
    if sum(_seg_lengths(pts)) < 0.4:
        errs.append("길이 너무 짧음 (전체 길이 0.4 미만)")

    spawns = [a for a in d["areas"] if a["area_type"] == "spawn"]
    goals = [a for a in d["areas"] if a["area_type"] == "goal"]
    combats = [a for a in d["areas"] if a["area_type"] == "combat"]
    if len(spawns) != 1:
        errs.append(f"spawn 개수 {len(spawns)} (1개여야 함)")
    if len(goals) != 1:
        errs.append(f"goal 개수 {len(goals)} (1개여야 함)")

    for a in d["areas"]:
        if not R_MIN <= a["normalized_radius"] <= R_MAX:
            errs.append(f"{a['area_type']} radius={a['normalized_radius']} 범위 밖")
    if spawns and _dist(spawns[0]["normalized_center"], pts[0]) > 0.01:
        errs.append("spawn이 길의 시작점에 있지 않음")
    if goals and _dist(goals[0]["normalized_center"], pts[-1]) > 0.01:
        errs.append("goal이 길의 끝점에 있지 않음")
    if _dist(pts[0], pts[-1]) < MIN_SPAWN_GOAL:
        errs.append(f"spawn과 goal이 너무 가까움 (직선거리 {_dist(pts[0], pts[-1]):.2f} < {MIN_SPAWN_GOAL})")
    for a in combats:
        dd = dist_to_path(a["normalized_center"], pts)
        lim = a["normalized_radius"] + gap
        if dd > lim + 1e-3:
            errs.append(f"combat 영역이 길에서 너무 멂 (중심 거리 {dd:.3f}, 허용 {lim:.3f})")
    for i in range(len(combats)):
        for j in range(i + 1, len(combats)):
            a, b = combats[i], combats[j]
            if _dist(a["normalized_center"], b["normalized_center"]) < a["normalized_radius"] + b["normalized_radius"]:
                errs.append("전투 영역끼리 겹침")
    return errs

# ================================================================ 자동 보정 (필수 규칙 복구)
def _push_end_away(pts, rng):
    """길의 끝점이 시작점에서 MIN_SPAWN_GOAL 이상 떨어지도록 끝점을 옮긴다. 옮겼으면 True."""
    start, end = pts[0], pts[-1]
    if _dist(start, end) >= MIN_SPAWN_GOAL:
        return False
    # 원래 끝점 방향을 우선으로, 안 되면 다른 방향 중 원래 끝점에 가장 가까운 후보를 고른다
    base = math.atan2(end["y"] - start["y"], end["x"] - start["x"]) if _dist(start, end) > 1e-6 \
        else math.atan2(0.5 - start["y"], 0.5 - start["x"])
    best = None
    for k in range(36):
        ang = base + math.radians(10 * ((k + 1) // 2) * (1 if k % 2 else -1))
        for dist in (MIN_SPAWN_GOAL + 0.05, MIN_SPAWN_GOAL + 0.01):
            c = {"x": round(start["x"] + dist * math.cos(ang), 3), "y": round(start["y"] + dist * math.sin(ang), 3)}
            if MAP_MIN <= c["x"] <= MAP_MAX and MAP_MIN <= c["y"] <= MAP_MAX and _dist(start, c) >= MIN_SPAWN_GOAL:
                best = c
                break
        if best:
            break
    if best is None:  # 시작점이 이상한 위치면 맵 중심 반대편 모서리 쪽으로
        best = {"x": MAP_MAX if start["x"] < 0.5 else MAP_MIN, "y": MAP_MAX if start["y"] < 0.5 else MAP_MIN}
    pts[-1] = {"x": round(best["x"], 3), "y": round(best["y"], 3)}
    return True


def auto_fix(d, rng, gap=COMBAT_GAP_MAX):
    notes = []
    env = d["base_environment"]
    for k in env:
        v = round(_clamp(env[k], 0.0, 1.0), 3)
        if v != env[k]:
            notes.append(f"{k} {env[k]} → {v}")
            env[k] = v

    path = d["main_path"]
    w = _clamp(path["path_width"], W_MIN, W_MAX)
    if w != path["path_width"]:
        notes.append(f"path_width {path['path_width']} → {w}")
        path["path_width"] = w

    # 경로: 맵 안으로, 연속 중복 점 제거, 최소 3점 (순서는 건드리지 않음)
    pts = path["normalized_points"]
    for p in pts:
        p["x"], p["y"] = _clamp(p["x"], MAP_MIN, MAP_MAX), _clamp(p["y"], MAP_MIN, MAP_MAX)
    cleaned = [pts[0]]
    for p in pts[1:]:
        if _dist(p, cleaned[-1]) < MIN_PATH_GAP:
            notes.append(f"겹치는 경로 점 ({p['x']:.2f}, {p['y']:.2f}) 제거")
            continue
        cleaned.append(p)
    pts[:] = cleaned
    if len(pts) < 2:  # 점이 하나밖에 없으면 임의 방향으로 길 생성
        ang = rng.uniform(0, 2 * math.pi)
        pts.append({"x": _clamp(pts[0]["x"] + 0.6 * math.cos(ang), MAP_MIN, MAP_MAX),
                    "y": _clamp(pts[0]["y"] + 0.6 * math.sin(ang), MAP_MIN, MAP_MAX)})
        notes.append("경로 점이 부족해 끝점 생성")
    while len(pts) < 3:
        i = max(range(len(pts) - 1), key=lambda k: _dist(pts[k], pts[k + 1]))
        mid = {"x": round((pts[i]["x"] + pts[i + 1]["x"]) / 2, 3), "y": round((pts[i]["y"] + pts[i + 1]["y"]) / 2, 3)}
        pts.insert(i + 1, mid)
        notes.append("경로 중간점 추가")

    if _push_end_away(pts, rng):
        notes.append(f"spawn과 goal이 너무 가까워 길의 끝점을 {MIN_SPAWN_GOAL} 이상 떨어진 곳으로 옮김")
        # 끝점을 옮기면서 생긴 마지막 구간이 너무 길면 중간점 추가
        if _dist(pts[-2], pts[-1]) > 0.35 and len(pts) < 10:
            pts.insert(len(pts) - 1, {"x": round((pts[-2]["x"] + pts[-1]["x"]) / 2, 3),
                                      "y": round((pts[-2]["y"] + pts[-1]["y"]) / 2, 3)})

    # 길이 스스로 겹치지 않게 (양 끝점은 고정)
    notes += untangle_path(pts)

    for a in d["areas"]:
        a["normalized_radius"] = round(_clamp(a["normalized_radius"], R_MIN, R_MAX), 3)
        a["detail_density"] = round(_clamp(a["detail_density"], 0.0, 1.0), 3)

    spawns = [a for a in d["areas"] if a["area_type"] == "spawn"]
    goals = [a for a in d["areas"] if a["area_type"] == "goal"]
    combats = [a for a in d["areas"] if a["area_type"] == "combat"]

    # spawn / goal: 정확히 1개, 길의 양 끝에 붙임
    spawn = spawns[0] if spawns else {"area_type": "spawn", "normalized_radius": 0.05, "detail_density": 0.2,
                                      "normalized_center": dict(pts[0])}
    goal = goals[0] if goals else {"area_type": "goal", "normalized_radius": 0.06, "detail_density": 0.4,
                                   "normalized_center": dict(pts[-1])}
    if len(spawns) != 1 or len(goals) != 1:
        notes.append(f"spawn {len(spawns)}개, goal {len(goals)}개 → 각 1개로 정리")
    # spawn/goal은 항상 길의 양 끝점에 정확히 붙인다 (눈에 띄게 옮긴 경우만 알림)
    if _dist(spawn["normalized_center"], pts[0]) > 0.01:
        notes.append("spawn을 길의 시작점으로 이동")
    if _dist(goal["normalized_center"], pts[-1]) > 0.01:
        notes.append("goal을 길의 끝점으로 이동")
    spawn["normalized_center"] = {"x": pts[0]["x"], "y": pts[0]["y"]}
    goal["normalized_center"] = {"x": pts[-1]["x"], "y": pts[-1]["y"]}

    # 길에서 너무 먼 전투는 허용 거리 안쪽으로 당김 (방향은 유지)
    def pull_near(a):
        c, lim = a["normalized_center"], a["normalized_radius"] + gap
        dd = dist_to_path(c, pts)
        if dd <= lim + 1e-4:
            return False
        q = closest_on_path(c, pts)
        k = (lim * 0.97) / dd
        a["normalized_center"] = {"x": round(q["x"] + (c["x"] - q["x"]) * k, 3),
                                  "y": round(q["y"] + (c["y"] - q["y"]) * k, 3)}
        return True

    if sum(pull_near(a) for a in combats):
        notes.append("길에서 너무 먼 전투 영역을 허용 거리 안으로 당김")

    # 전투 개수 맞추기: 기존 전투는 살리고, 부족하면 빈 구간에 랜덤 추가, 많으면 제거
    want = d.pop("combat_count", None)
    if want is not None:
        want = max(0, int(want))
        combats.sort(key=lambda a: progress_of(a["normalized_center"], pts))
        if len(combats) > want:
            notes.append(f"전투 {len(combats)}개 → {want}개로 줄임")
            combats = combats[:want]
        elif len(combats) < want:
            notes.append(f"전투 {len(combats)}개 → {want}개로 늘림 (빈 구간에 랜덤 배치)")
            used = [progress_of(a["normalized_center"], pts) for a in combats]
            r = sum(a["normalized_radius"] for a in combats) / len(combats) if combats else round(rng.uniform(0.05, 0.08), 3)
            dd = sum(a["detail_density"] for a in combats) / len(combats) if combats else round(rng.uniform(0.4, 0.6), 2)
            while len(combats) < want:
                gap = 0.6 / (want + 1)
                for _ in range(50):
                    t = rng.uniform(0.18, 0.85)
                    if all(abs(t - u) > gap for u in used):
                        break
                used.append(t)
                nr = round(_clamp(r * rng.uniform(0.85, 1.15), R_MIN, R_MAX), 3)
                off = rng.uniform(-(nr + gap), nr + gap) * 0.9 if gap > 0 else 0.0
                combats.append({"area_type": "combat", "normalized_center": place_on_path(pts, t, off),
                                "normalized_radius": nr,
                                "detail_density": round(_clamp(dd + rng.uniform(-0.1, 0.1), 0, 1), 2)})
            combats.sort(key=lambda a: progress_of(a["normalized_center"], pts))

    # 전투끼리 겹치면(길이 되돌아와서 멀리 떨어진 전투끼리 만나는 경우 포함)
    # 먼저 길을 따라 뒤쪽 전투를 밀어내 보고, 안 되면 크기를 줄임
    if len(combats) > 1:
        total = sum(_seg_lengths(pts)) or 1.0
        combats.sort(key=lambda a: progress_of(a["normalized_center"], pts))
        changed = False
        for it in range(80):
            pair = next(((a, b) for i, a in enumerate(combats) for b in combats[i + 1:]
                         if _dist(a["normalized_center"], b["normalized_center"])
                         < a["normalized_radius"] + b["normalized_radius"]), None)
            if pair is None:
                break
            changed = True
            a, b = pair
            t = progress_of(b["normalized_center"], pts) + 0.03
            if it < 40 and t <= 0.92:
                b["normalized_center"] = place_on_path(pts, t, path_coords(b["normalized_center"], pts)[1])
            else:
                for c in (a, b):
                    c["normalized_radius"] = round(max(R_MIN, c["normalized_radius"] * 0.85), 3)
                if a["normalized_radius"] == b["normalized_radius"] == R_MIN:
                    b["normalized_center"] = point_at_progress(pts, _clamp(t, 0.1, 0.95))
        # 그래도 겹치면: 길을 따라 고르게 다시 펼치고, 간격에 맞게 크기 제한
        def _any_overlap():
            return any(_dist(a["normalized_center"], b["normalized_center"]) < a["normalized_radius"] + b["normalized_radius"]
                       for i, a in enumerate(combats) for b in combats[i + 1:])
        if _any_overlap():
            n = len(combats)
            cap = max(R_MIN, 0.45 * (0.7 * total / n))
            for k, c in enumerate(combats):
                off = path_coords(c["normalized_center"], pts)[1] * 0.5
                c["normalized_center"] = place_on_path(pts, 0.15 + 0.73 * (k + 0.5) / n, off)
                c["normalized_radius"] = round(_clamp(min(c["normalized_radius"], cap), R_MIN, R_MAX), 3)
            while _any_overlap() and any(c["normalized_radius"] > R_MIN for c in combats):
                for c in combats:
                    c["normalized_radius"] = round(max(R_MIN, c["normalized_radius"] * 0.9), 3)
        for c in combats:  # 크기를 줄였으면 허용 거리도 줄었으니 다시 확인
            pull_near(c)
        if changed:
            notes.append("겹치는 전투 영역을 떼어놓거나 크기를 줄임")
        combats.sort(key=lambda a: progress_of(a["normalized_center"], pts))

    d["areas"] = [spawn] + combats + [goal]
    return notes

# ================================================================ 전투 위치 랜덤 (길 위 / 길 옆)
def scatter_combats(d, rng, gap=COMBAT_GAP_MAX):
    """각 전투를 길 위에 걸칠지 길 옆으로 뺄지 랜덤으로 정한다. 길을 따라가는 위치(진행도)는 유지."""
    pts = d["main_path"]["normalized_points"]
    combats = [a for a in d["areas"] if a["area_type"] == "combat"]
    if not combats:
        return []
    ratio = rng.uniform(*OFF_PATH_RATIO)
    n_off = 0
    for a in combats:
        t, _ = path_coords(a["normalized_center"], pts)
        r = a["normalized_radius"]
        side = rng.choice((-1, 1))
        if rng.random() < ratio:
            # 길 옆: 원 가장자리와 길 사이 간격을 0.01 ~ gap 사이에서 랜덤
            off = side * (r + rng.uniform(0.01, gap * 0.95))
            n_off += 1
        else:
            # 길 위: 원이 길을 덮되 중심은 조금 흔들림
            off = side * rng.uniform(0, 0.5 * r)
        a["normalized_center"] = place_on_path(pts, t, off)
    return [f"전투 {len(combats)}개 중 {n_off}개를 길 옆으로 배치 (이번 맵 길 옆 비율 {ratio:.0%})"]


# ================================================================ 코드 변주 (규칙은 지키고 모양만 흔들기)
def vary(d, rng, strength, gap=COMBAT_GAP_MAX):
    d = copy.deepcopy(d)
    pts = d["main_path"]["normalized_points"]
    combats = [a for a in d["areas"] if a["area_type"] == "combat"]
    prog = [path_coords(a["normalized_center"], pts) for a in combats]

    s = 0.08 * strength
    for p in pts:
        p["x"] = round(_clamp(p["x"] + rng.uniform(-s, s), MAP_MIN, MAP_MAX), 3)
        p["y"] = round(_clamp(p["y"] + rng.uniform(-s, s), MAP_MIN, MAP_MAX), 3)

    for a, (t, off) in zip(combats, prog):
        a["normalized_radius"] = round(_clamp(a["normalized_radius"] * rng.uniform(1 - 0.2 * strength, 1 + 0.2 * strength), R_MIN, R_MAX), 3)
        lim = a["normalized_radius"] + gap
        off = _clamp(off + rng.uniform(-0.05, 0.05) * strength, -lim * 0.95, lim * 0.95)
        a["normalized_center"] = place_on_path(pts, _clamp(t + rng.uniform(-0.08, 0.08) * strength, 0.15, 0.88), off)
        a["detail_density"] = round(_clamp(a["detail_density"] + rng.uniform(-0.1, 0.1) * strength, 0, 1), 2)

    for a in d["areas"]:
        if a["area_type"] == "spawn":
            a["normalized_center"] = dict(pts[0])
        elif a["area_type"] == "goal":
            a["normalized_center"] = dict(pts[-1])

    env = d["base_environment"]
    for k in env:
        env[k] = round(_clamp(env[k] + rng.uniform(-0.06, 0.06) * strength, 0, 1), 2)
    d["main_path"]["path_width"] = round(_clamp(d["main_path"]["path_width"] * rng.uniform(1 - 0.15 * strength, 1 + 0.15 * strength), W_MIN, W_MAX))
    return d

# ================================================================ LLM 호출
def ask(messages, temperature, seed):
    resp = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=temperature,
        seed=seed,
        max_tokens=1800,
        response_format={"type": "json_schema", "json_schema": {"name": "level_layout", "schema": SCHEMA}},
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    return json.loads(resp.choices[0].message.content)


def build_request(answers):
    lines = ["아래 질문과 답변을 바탕으로 레벨 레이아웃 JSON을 만들어라.\n"]
    for (q, _, fields, hint), a in zip(QUESTIONS, answers):
        lines.append(f"[질문] {q}")
        lines.append(f"[관련 필드] {fields}")
        lines.append(f"[답변] {a if a else '(답변 없음) ' + hint}\n")
    return "\n".join(lines)


def log(event, state, **extra):
    record = {"time": datetime.now().isoformat(timespec="seconds"), "session": state["session"],
              "model": MODEL, "event": event, "answers": state["answers"], "seed": state["seed"], **extra}
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

# ================================================================ 미리보기
AREA_COLORS = {"spawn": "#2e86de", "combat": "#e74c3c", "goal": "#27ae60"}
THEME_BG = {"forest": "#e8f3e0", "grassland": "#f1f7d9", "desert": "#f7ecd0", "snow": "#eef3f8"}


def draw(d):
    fig = Figure(figsize=(6, 6))
    ax = fig.add_subplot(111)
    ax.set_facecolor(THEME_BG.get(d["theme"], "#f0f0f0"))
    ax.set_xlim(0, 1)
    ax.set_ylim(1, 0)
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)

    pts = d["main_path"]["normalized_points"]
    xs, ys = [p["x"] for p in pts], [p["y"] for p in pts]
    ax.plot(xs, ys, color="#8d6e63", lw=max(2, d["main_path"]["path_width"] / 60), solid_capstyle="round", alpha=0.8)
    ax.plot(xs, ys, "o", color="#4e342e", ms=4)
    for i in range(len(pts) - 1):  # 진행 방향 화살표
        mx, my = (xs[i] + xs[i + 1]) / 2, (ys[i] + ys[i + 1]) / 2
        ax.annotate("", xy=(mx + (xs[i + 1] - xs[i]) * 0.05, my + (ys[i + 1] - ys[i]) * 0.05), xytext=(mx, my),
                    arrowprops=dict(arrowstyle="->", color="#3e2723", lw=1.5))

    for a in d["areas"]:
        c, r = a["normalized_center"], a["normalized_radius"]
        color = AREA_COLORS.get(a["area_type"], "gray")
        if a["area_type"] == "combat" and dist_to_path(c, pts) > 0.005:
            q = closest_on_path(c, pts)
            ax.plot([c["x"], q["x"]], [c["y"], q["y"]], ls="--", color=color, lw=1)
        ax.add_patch(Circle((c["x"], c["y"]), r, color=color, alpha=0.35))
        ax.add_patch(Circle((c["x"], c["y"]), r, fill=False, ec=color, lw=2))
        ax.plot(c["x"], c["y"], "+", color=color, ms=10, mew=2)
        ax.text(c["x"], c["y"] - r - 0.02, a["area_type"], ha="center", va="bottom", fontsize=9, weight="bold")

    env = d["base_environment"]
    ax.set_title(f"{d['theme']}  |  tree {env['tree_density']:.2f}  rock {env['rock_density']:.2f}  "
                 f"grass {env['grass_density']:.2f}  |  width {d['main_path']['path_width']:.0f}", fontsize=9)
    fig.tight_layout()
    return fig

# ================================================================ 화면 갱신
def fix_note(fixes):
    return "🔧 **자동 보정**\n" + "\n".join(f"- {f}" for f in fixes) if fixes else ""


def render(state, note=""):
    d = state["data"]
    errs = validate(d, state["gap"])
    check = "✅ **필수 규칙 통과**" if not errs else "⚠️ **검증 경고**\n" + "\n".join(f"- {e}" for e in errs)
    check = f"`seed {state['seed']}`\n\n" + (note + "\n\n" if note else "") + check

    path = os.path.join(tempfile.mkdtemp(), "layout.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(d, f, ensure_ascii=False, indent=2)
    hist = "\n".join(f"{i + 1}. {e}" for i, e in enumerate(state["edits"])) or "_아직 수정 없음_"
    return json.dumps(d, ensure_ascii=False, indent=2), check, draw(d), "**수정 기록**\n" + hist, path, errs


def _new_seed(seed_text):
    s = str(seed_text).strip()
    return int(s) if s.lstrip("-").isdigit() else random.randint(0, 2**31 - 1)


def _commit(state, data, fixes, event, **extra):
    state["data"] = data
    js, check, fig, hist, path, errs = render(state, fix_note(fixes))
    log(event, state, output=data, fixes=fixes, errors=errs, **extra)
    return js, check, fig, hist, path, state


def on_generate(a1, a2, a3, temperature, seed_text, gap=COMBAT_GAP_MAX):
    answers = [a1.strip(), a2.strip(), a3.strip()]
    seed = _new_seed(seed_text)
    state = {"session": uuid.uuid4().hex[:8], "answers": answers, "seed": seed, "gap": gap,
             "messages": [], "data": None, "edits": []}
    request = build_request(answers)
    state["messages"] = [{"role": "system", "content": build_system(gap)}, {"role": "user", "content": request}]
    try:
        data = ask(state["messages"], temperature, seed)
    except Exception as e:
        return "", f"❌ 생성 실패: {e}", None, "", None, None

    raw_errs = validate({k: v for k, v in data.items() if k != "combat_count"}, gap)
    rng = random.Random(seed)
    fixes = auto_fix(data, rng, gap)
    fixes += scatter_combats(data, rng, gap)
    fixes += auto_fix(data, rng, gap)
    state["messages"].append({"role": "assistant", "content": json.dumps(data, ensure_ascii=False)})
    return _commit(state, data, fixes, "generate", request=request, temperature=temperature, gap=gap, raw_errors=raw_errs)


def _set_gap(state, gap):
    state["gap"] = gap
    state["messages"][0]["content"] = build_system(gap)


def on_vary(strength, state, gap=COMBAT_GAP_MAX):
    if state is None:
        return "", "먼저 **생성**을 눌러주세요.", None, "", None, None
    _set_gap(state, gap)
    state["seed"] = random.randint(0, 2**31 - 1)
    rng = random.Random(state["seed"])
    data = vary(state["data"], rng, strength, gap)
    fixes = auto_fix(data, rng, gap)
    fixes += scatter_combats(data, rng, gap)
    fixes += auto_fix(data, rng, gap)
    state["edits"].append(f"(모양 변주, 강도 {strength})")
    state["messages"] += [
        {"role": "user", "content": "[코드로 모양을 변주했다. 이후 수정은 아래 JSON을 기준으로 한다.]"},
        {"role": "assistant", "content": json.dumps(data, ensure_ascii=False)},
    ]
    return _commit(state, data, fixes, "vary", strength=strength)


def on_edit(msg, temperature, state, gap=COMBAT_GAP_MAX):
    msg = msg.strip()
    if state is None:
        return "", "먼저 질문에 답하고 **생성**을 눌러주세요.", None, "", None, None, msg
    _set_gap(state, gap)
    if not msg:
        return (*render(state)[:5], state, "")
    messages = state["messages"] + [{"role": "user", "content": msg}]
    try:
        data = ask(messages, temperature, state["seed"])
    except Exception as e:
        return (*render(state, f"❌ 수정 실패: {e}")[:5], state, msg)

    raw_errs = validate({k: v for k, v in data.items() if k != "combat_count"}, gap)
    fixes = auto_fix(data, random.Random(state["seed"]), gap)
    state["messages"] = messages + [{"role": "assistant", "content": json.dumps(data, ensure_ascii=False)}]
    state["edits"].append(msg)
    return (*_commit(state, data, fixes, "edit", request=msg, temperature=temperature, raw_errors=raw_errs), "")


def on_reset():
    return "", "", "", "", "", "", "", "", None, None, None

# ================================================================ UI
with gr.Blocks(title="PCG 레벨 생성기") as demo:
    gr.Markdown(f"# PCG 레벨 레이아웃 생성기\n모델: `{MODEL}` · 모르는 질문은 비워두면 모델이 알아서 정합니다.")
    state = gr.State(None)

    with gr.Row():
        with gr.Column(scale=1):
            answer_boxes = [gr.Textbox(label=q, placeholder=ex, lines=2) for q, ex, _, _ in QUESTIONS]
            with gr.Row():
                temp = gr.Slider(0.0, 1.2, value=0.8, step=0.1, label="창의성 (temperature)")
                seed_box = gr.Textbox(label="seed (비우면 랜덤)", placeholder="예) 1234")
            gen_btn = gr.Button("생성", variant="primary")

            gr.Markdown("---\n### 모양만 다르게")
            with gr.Row():
                strength = gr.Slider(0.2, 2.0, value=1.0, step=0.1, label="변주 강도")
                vary_btn = gr.Button("다른 버전 (규칙 유지, 모양만 변주)")

            gr.Markdown("---\n### 수정")
            edit_box = gr.Textbox(label="수정 요청", placeholder="예) 나무 좀 줄이고 전투 하나 더 추가", lines=2)
            with gr.Row():
                edit_btn = gr.Button("수정 적용")
                reset_btn = gr.Button("처음부터")
            history = gr.Markdown()

        with gr.Column(scale=1):
            plot = gr.Plot(label="배치 미리보기")
            check = gr.Markdown()
            json_out = gr.Code(language="json", label="결과 JSON")
            file_out = gr.File(label="layout.json 다운로드")

    outputs = [json_out, check, plot, history, file_out, state]
    gen_btn.click(on_generate, inputs=answer_boxes + [temp, seed_box], outputs=outputs)
    vary_btn.click(on_vary, inputs=[strength, state], outputs=outputs)
    edit_btn.click(on_edit, inputs=[edit_box, temp, state], outputs=outputs + [edit_box])
    edit_box.submit(on_edit, inputs=[edit_box, temp, state], outputs=outputs + [edit_box])
    reset_btn.click(on_reset, outputs=answer_boxes + [seed_box, json_out, check, history, edit_box, plot, file_out, state])

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860)