# ComfyU Profile Automation

ComfyUI 이미지 자동 생성 스크립트의 프로필 분리형 재설계 프로젝트입니다.

> 기존 `lilly1987/ComfyU-auto-script-call` 저장소는 참고용으로만 사용하며 이 프로젝트에서 수정하지 않습니다.

## 핵심 설계

- **프로필을 먼저 선택**합니다. `IL` / `Anime`는 설정 가중치로 선택하며, 한 프로필의 생성 묶음이 끝나면 다음 프로필을 다시 선택합니다.
- 선택된 프로필의 Checkpoint, Character, LoRA, YAML, Wildcard, Workflow만 사용합니다. IL과 Anime 리소스를 서로 섞지 않습니다.
- 프로필별 파일은 상대 경로를 식별자로 사용합니다. 같은 파일명이 서로 다른 하위 폴더에 있어도 덮어쓰지 않습니다.
- ComfyUI API 전송은 별도 모듈로 분리합니다.

## 설치 및 실행

Python 3.10+ 및 실행 중인 ComfyUI가 필요합니다.

```bash
python -m pip install -r requirements.txt
python main.py --config config.yml
```

Windows에서는 `run.bat`를 실행할 수 있습니다. 최초 실행 전 `config.example.yml`을 `config.yml`로 복사하세요.

## 데이터 폴더

기본 경로는 프로젝트 외부의 `../ComfyU-auto-script_data`입니다.

```text
ComfyU-auto-script_data/
├── IL/
│   ├── checkpoint/
│   ├── lora/
│   ├── WeightCheckpoint.yml
│   ├── WeightChar.yml
│   ├── WeightLora.yml
│   ├── setupWorkflow.yml
│   ├── setupWildcard.yml
│   └── workflow_api.json
└── Anime/
    ├── checkpoint/
    ├── lora/
    ├── WeightCheckpoint.yml
    ├── WeightChar.yml
    ├── WeightLora.yml
    ├── setupWorkflow.yml
    ├── setupWildcard.yml
    └── workflow_api.json
```

`Anime` 폴더명은 요청한 그대로 유지합니다.

## 현재 구현 범위

초기 기반 코드에는 프로필 단위 선택기, 프로필 격리형 리소스 카탈로그, 가중치 선택, Workflow 로딩/치환, ComfyUI 큐 전송, 기본 테스트가 포함됩니다. 기존 프로젝트의 모든 선택 모드(DB/Favorites/fromImg/Observer/TinyDB 등)를 완전히 이식한 상태는 아닙니다. 이식은 기존 설정 및 각 로직을 대조해 단계적으로 진행합니다.

```bash
python -m unittest discover -s tests -v
```
