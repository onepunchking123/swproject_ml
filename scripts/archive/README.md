# archive

현재 파이프라인에서 쓰지 않는 스크립트. 재현·재개용으로 남긴다.

| 스크립트 | 용도 | 상태 |
|---|---|---|
| fetch_omnifall.py · fetch_public.py · prepare_omnifall.py | 공개 데이터셋 → Drive 확보 (`configs/datasets.yaml` 참조) | 완료 — Drive 에 63GB 확보됨 |
| aihub_stream.py · stream_parallel.sh | AI Hub → Drive 스트리밍 | 중단 — 연결 끊김 반복, 라벨 142MB 만 확보 |
| run_pipeline.sh | 3개 데이터셋 일괄 해제→추출 | `expand_one.sh` 로 대체 (세션 회수 대비 분할) |
| 01_explore_dataset.ipynb | AI Hub 구조 탐색 노트북 | Colab CLI 로 대체 |
