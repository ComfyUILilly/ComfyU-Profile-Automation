# ComfyU Profile Automation

기존 ComfyUI 자동 생성 프로젝트를 **IL / Anime 프로필 분리형**으로 재구성한 프로젝트입니다.

기존 `lilly1987/ComfyU-auto-script-call` 저장소는 참고용으로만 사용하며 수정하지 않습니다.

## 실행 원칙

1. IL / Anime 프로필을 설정 가중치에 따라 먼저 선택합니다.
2. 선택된 프로필 안에서 Checkpoint, Character, LoRA를 선택합니다.
3. 해당 프로필의 YAML, Wildcard, Workflow, 사용 기록만 사용합니다.
4. 설정된 배치 생성 수가 끝난 후 다음 프로필을 다시 랜덤 선택합니다.
5. 프로필 간 리소스 혼합은 차단합니다.

## 지원 기능

- Checkpoint / Character / LoRA의 Weight, Random, DB, Cycle, Favorites, Skip 선택 및 반복 생성 주기
- LoRA 그룹별 `WeightLora.yml`의 per / weight / total / limit 설정
- Anime의 `UNETLoader`, IL의 `CheckpointLoaderSimple` 우선 처리
- 프로필별 `setupWorkflow.yml`, `setupWildcard.yml`, 자산별 YAML 설정
- positive / negative prompt 조합 및 중첩 Wildcard 확장
- PNG `prompt` 메타데이터에서 API Workflow 복원(fromImg)
- 프로필별 TinyDB 사용 횟수 기록 및 XLSX 내보내기
- 프로필 데이터 및 모델 폴더 변경 감시(변경 내용은 다음 배치 로드부터 반영)
- ComfyUI API 큐 전송 및 큐 대기

## 설치 및 실행

Python 3.10+ 및 실행 중인 ComfyUI가 필요합니다.

```bash
python -m pip install -r requirements.txt
```

`config.example.yml`을 `config.yml`로 복사한 뒤 ComfyUI 경로와 프로필별 모델 경로를 수정합니다.

```bash
python scripts/init_data.py
python main.py --config config.yml
```

Windows의 ComfyUI portable embedded Python을 사용하는 경우 프로젝트의 `run.bat`을 실행합니다. 실행기는 프로젝트 루트와 패키지 폴더를 확인하고 `PYTHONPATH`를 설정합니다.

`ModuleNotFoundError: No module named 'comfy_profile_automation'`가 발생하면 다음 파일이 반드시 존재하는지 확인합니다.

```text
ComfyU-Profile-Automation/
├── main.py
└── comfy_profile_automation/
    ├── __init__.py
    └── config.py
```

GitHub에서 `main.py`만 따로 받으면 안 됩니다. 저장소 전체를 clone 또는 ZIP으로 받아야 합니다. 명령줄에서 확인하려면 프로젝트 루트에서 다음을 실행합니다.

```bash
python -c "import comfy_profile_automation; print(comfy_profile_automation.__file__)"
```

실제 생성 전에 경로와 선택 흐름을 확인하려면 `--dry-run`을 사용할 수 있습니다. ComfyUI로 전송하지 않으며, 이 모드는 서버 모델 목록 동기화나 API 연결까지 검사하지는 않습니다. 전체 Workflow JSON을 로그로 보려면 `generation.print_workflow: true`로 설정합니다.

## 데이터 구조

```text
ComfyU-auto-script_data/
├── IL/
│   ├── checkpoint/
│   ├── lora/
│   │   ├── char/
│   │   └── etc/
│   ├── WeightCheckpoint.yml
│   ├── WeightChar.yml
│   ├── WeightLora.yml
│   ├── setupWorkflow.yml
│   ├── setupWildcard.yml
│   └── workflow_api.json
└── Anime/
    ├── checkpoint/
    ├── lora/
    │   ├── char/
    │   └── etc/
    ├── WeightCheckpoint.yml
    ├── WeightChar.yml
    ├── WeightLora.yml
    ├── setupWorkflow.yml
    ├── setupWildcard.yml
    └── workflow_api.json
```

각 프로필의 실제 API 형식 `workflow_api.json`과 YAML 설정은 기존 프로젝트의 파일을 해당 프로필 폴더로 옮겨서 사용합니다. `fromImg`를 사용하려면 프로필의 `from_img_path`에 PNG 결과 이미지가 들어 있는 폴더를 지정해야 합니다.

## 반복 생성 주기

`generation.images_per_checkpoint`, `images_per_character`, `images_per_lora`로 각각의 선택 결과를 몇 장 동안 유지할지 설정합니다. 기존 설정의 `CheckpointLoop`, `CharLoop`, `queueLoop` 키도 호환되며, 이 경우 기존 카운터 방식에 맞춰 지정값보다 1회 더 반복합니다.

## 선택 모드 설정

`checkpoint_kind`, `character_kind`, `lora_kind`는 문자열 하나 또는 가중치 딕셔너리로 지정할 수 있습니다. 예: `{Weight: 4, Random: 1, DB: 1, Cycle: 1}`.

`Favorites` / `Skip` 모드는 해당 선택 모드에서만 각 자산 YAML의 `favorites` / `skip` 값으로 후보를 고릅니다. `skip: true`를 모든 선택 모드에서 제외하는 전역 필터로 사용하지 않습니다. 따라서 자동 추출 직후 `skip: true`인 자산도 기존 선택 모드(예: Weight 또는 Random)에서는 선택 대상에 남고, 사용자가 검토한 뒤 값을 변경하면 Skip 모드의 후보군이 달라집니다. `Skip_Weight`는 skip 플래그가 있는 후보 안에서 가중치를 적용합니다. `WeightLora.yml`은 기존 그룹형 구조(`per`, `weight`, `total`, `*Max`, `dic`)를 사용합니다.

## 참고

- 사용 기록은 프로필별 `database/usage.json`으로 분리되고, XLSX로 내보낼 수 있습니다.
- 모델 식별자는 파일명만이 아니라 프로필·종류·상대 경로를 포함합니다.
- 이 프로젝트는 사용자의 ComfyUI 설치 및 실제 Workflow 노드 ID에 따라 경로/노드 설정을 조정해야 합니다.
## 호환성 및 오류 처리

- `workflow_api.json`는 JSON 문자열 내부를 손상시키지 않는 범위에서 객체/배열 끝의 trailing comma를 허용합니다. 그 외 JSON 문법 오류는 파일 경로, 줄, 열을 표시합니다.
- YAML 구문 오류 또는 YAML 최상위 구조가 잘못된 경우 해당 파일 경로와 위치를 포함해 오류를 표시합니다.
- 기존 `dataPath/count.db`가 있으면 각 프로필의 `database/usage.json`으로 사용 기록(Checkpoint / Character / LoRA / 조합 / 태그)을 최초 실행 시 이전합니다. 원본 `count.db`는 그대로 유지합니다.
- `queue_prompt: false`이면 ComfyUI API로 모델 목록 동기화나 큐 전송을 하지 않으며, 실제 전송이 없으므로 사용 횟수도 기록하지 않습니다.
- `SaveImage1`와 `SaveImage2`의 파일명 prefix 끝에 각각 `-1`, `-2`가 붙고, `SaveVideo`에는 공통 prefix를 사용합니다.
- 기존 설정의 `safetensorsStart`가 유효한 `프로필/파일` 경로를 가리키면 첫 배치에서 해당 Checkpoint를 먼저 사용합니다.
