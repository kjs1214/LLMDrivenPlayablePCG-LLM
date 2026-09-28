# LLM-Driven PCG Layout Generator

자연어 답변을 바탕으로 언리얼 엔진 PCG(Procedural Content Generation)에 넣을 레벨 레이아웃 JSON을 생성하는 웹 앱입니다.

질문 3개에 답하면 LLM이 테마, 환경 밀도, 메인 길, 전투 구역 배치를 JSON으로 만들고, 코드가 필수 규칙을 검증하고 보정한 뒤 미리보기 그림과 함께 보여줍니다.

## 구성

```
브라우저 → app.py (Gradio, CPU) → HTTP → vLLM 서버 (GPU) → JSON 반환
```

- **vLLM**: LLM 서빙 (OpenAI 호환 API, JSON schema 강제 출력)
- **app.py**: 웹 UI, 프롬프트 구성, 검증 및 자동 보정, 미리보기, 로그 기록

`app.py`는 GPU를 사용하지 않으며, vLLM과는 별도의 파이썬 환경에서 실행해도 됩니다.

## 주요 기능

- 고정 질문 3개(환경 / 길 모양 / 진행 흐름)에 답하면 레이아웃 생성
- 대화로 수정 ("나무 좀 줄이고 전투 하나 더 추가")
- **다른 버전**: LLM을 다시 부르지 않고 요청 내용은 유지한 채 모양만 랜덤 변주
- **seed**: 같은 seed로 같은 결과 재현
- 배치 미리보기 (길 진행 방향, 영역 중심, 길 밖 전투 연결선 표시)
- `layout.json` 다운로드
- 모든 생성·수정 기록을 `qa_log.jsonl`에 저장 (보정 전 오류, 보정 내역 포함)

## 필수 규칙

LLM 출력이 어기더라도 코드가 자동으로 보정합니다. 이 규칙 외의 형태(길의 방향, 모양, 점 개수, 시작과 끝 위치, 전투 간격 등)는 자유롭게 생성됩니다.

| 규칙 | 기준 |
|---|---|
| 길 연결 | 길의 첫 점 = spawn, 마지막 점 = goal |
| spawn–goal 거리 | 직선거리 0.45 이상 |
| 길 자기 겹침 금지 | 교차 없음, 서로 다른 구간 간격 0.05 이상, 25도 미만 급회전 없음 |
| 전투 개수 | 요청한 개수와 정확히 일치 |
| 전투 위치 | 원 가장자리가 길에서 0.12 이내 (길 위 또는 길 옆) |
| 전투 겹침 | 전투 영역끼리 겹치지 않음 |
| 값 범위 | density 0~1, radius 0.03~0.15, path_width 200~1500 |

전투를 길 위에 둘지 길 옆으로 뺄지는 seed에 따라 랜덤으로 정해집니다. 기준값은 `app.py` 상단의 상수로 조절할 수 있습니다.

```python
MIN_SPAWN_GOAL = 0.45        # spawn-goal 최소 직선거리
COMBAT_GAP_MAX = 0.12        # 전투 원 가장자리와 길 사이 최대 거리
OFF_PATH_RATIO = (0.2, 0.7)  # 맵마다 길 옆으로 빠지는 전투 비율 범위
MIN_PATH_GAP = 0.05          # 길의 서로 다른 구간 사이 최소 간격
MIN_TURN_ANGLE = 25          # 이보다 좁게 꺾이면 헤어핀으로 판단 (도)
```

## 출력 형식

좌표는 맵 전체를 0~1로 정규화한 2D 평면입니다. x는 왼쪽(0)에서 오른쪽(1), y는 위(0)에서 아래(1)로 증가합니다.

```json
{
  "theme": "forest",
  "base_environment": {
    "tree_density": 0.75,
    "rock_density": 0.2,
    "grass_density": 0.5
  },
  "main_path": {
    "path_type": "main",
    "path_width": 450,
    "normalized_points": [
      { "x": 0.08, "y": 0.62 },
      { "x": 0.27, "y": 0.55 },
      { "x": 0.52, "y": 0.41 },
      { "x": 0.78, "y": 0.58 },
      { "x": 0.92, "y": 0.47 }
    ]
  },
  "areas": [
    { "area_type": "spawn",  "normalized_center": { "x": 0.08, "y": 0.62 }, "normalized_radius": 0.05, "detail_density": 0.2 },
    { "area_type": "combat", "normalized_center": { "x": 0.40, "y": 0.48 }, "normalized_radius": 0.09, "detail_density": 0.6 },
    { "area_type": "goal",   "normalized_center": { "x": 0.92, "y": 0.47 }, "normalized_radius": 0.06, "detail_density": 0.4 }
  ]
}
```

| 필드 | 설명 |
|---|---|
| `theme` | `forest`, `grassland`, `desert`, `snow` |
| `base_environment` | 나무·바위·풀 밀도 (0~1) |
| `main_path.path_width` | 길 너비 (언리얼 단위, cm) |
| `main_path.normalized_points` | 길을 이루는 점들 (spawn → goal 순서) |
| `areas[].area_type` | `spawn`, `combat`, `goal` |
| `areas[].normalized_radius` | 영역 반경 (맵 한 변 대비) |
| `areas[].detail_density` | 영역 내부 디테일 밀도 (0~1) |

## 설치

두 개의 파이썬 환경을 권장합니다. 하나는 vLLM 서빙용, 하나는 앱용입니다.

```bash
# 앱용 환경
conda create -n pcg python=3.11 -y
conda activate pcg
pip install -r requirements.txt
```

```bash
# 서빙용 환경 (별도)
pip install vllm
```

## 실행

### 1. 모델 서버

```bash
CUDA_VISIBLE_DEVICES=0 vllm serve Qwen/Qwen3-8B --max-model-len 8192 --port 8000
```

`Application startup complete`가 출력될 때까지 기다립니다. 모델 저장 위치를 바꾸려면 `HF_HOME` 환경변수를 지정하세요.

12GB급 GPU 환경을 가정한다면 4비트 양자화 모델을 사용할 수 있습니다.

```bash
vllm serve Qwen/Qwen3-8B-AWQ --max-model-len 4096 --gpu-memory-utilization 0.85 --port 8000
```

`--gpu-memory-utilization`은 GB가 아니라 GPU 전체 메모리 대비 비율입니다.

### 2. 웹 앱

```bash
python app.py
```

실행 시 vLLM에 떠 있는 모델을 자동으로 감지합니다. 다른 주소나 모델을 쓰려면 환경변수로 지정합니다.

```bash
VLLM_URL=http://localhost:8000/v1 MODEL=Qwen/Qwen3-8B python app.py
```

### 3. 접속

같은 컴퓨터에서 실행했다면 브라우저에서 `http://localhost:7860`에 접속합니다.

원격 서버에서 실행했다면 SSH 포트 포워딩으로 로컬에서 접속할 수 있습니다.

```bash
ssh -L 7860:localhost:7860 user@server
# 점프 호스트를 거치는 경우
ssh -J user@jump-host:port -p port -L 7860:localhost:7860 user@server
```

## 언리얼 연동 시 참고

길 점과 영역 중심은 **같은 변환 함수**로 월드 좌표로 바꿔야 합니다. 두 곳의 변환 방식이 다르거나, 스플라인을 로컬 좌표로 추가했는데 액터가 회전되어 있으면 길과 영역이 서로 어긋납니다.

```cpp
FVector NormToWorld(const FVector2D& N, const FBox& MapBounds)
{
    // 미리보기 오른쪽(+x) → 월드 +Y, 미리보기 아래쪽(+y) → 월드 -X
    const double WorldY = FMath::Lerp(MapBounds.Min.Y, MapBounds.Max.Y, N.X);
    const double WorldX = FMath::Lerp(MapBounds.Max.X, MapBounds.Min.X, N.Y);
    return FVector(WorldX, WorldY, MapBounds.Max.Z);
}

double NormRadiusToWorld(double R, const FBox& MapBounds)
{
    const FVector Size = MapBounds.GetSize();
    return R * FMath::Min(Size.X, Size.Y);
}

// 스플라인 점은 월드 좌표로 추가
Spline->AddSplinePoint(NormToWorld(P, MapBounds), ESplineCoordinateSpace::World, false);
```

축 방향은 에디터 뷰 설정에 따라 다를 수 있으니, 방향이 뒤집혀 보이면 해당 축의 `Lerp` 순서를 바꾸세요.

길 옆에 배치된 전투 구역은 전투 중심에서 길 위 가장 가까운 점까지 샛길을 만들어 주거나, 주변 오브젝트를 비워주면 자연스럽게 보입니다.

## 로그

`qa_log.jsonl`에 한 줄씩 기록됩니다.

| 필드 | 설명 |
|---|---|
| `event` | `generate`, `edit`, `vary` |
| `answers` | 질문 3개에 대한 답변 |
| `seed`, `temperature` | 재현용 설정 |
| `raw_errors` | 보정 전 LLM 출력의 규칙 위반 |
| `fixes` | 자동 보정 내역 |
| `output` | 최종 JSON |

`raw_errors`를 모아 보면 모델이 어떤 실수를 자주 하는지 알 수 있어, 모델 비교나 파인튜닝 데이터 구축에 활용할 수 있습니다. 입력 내용이 담기므로 공개 저장소에는 올리지 않도록 `.gitignore`에 포함되어 있습니다.
