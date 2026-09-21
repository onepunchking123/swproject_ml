# 개발 환경 설정

## Colab CLI

브라우저 없이 터미널에서 Colab 런타임을 사용한다.

```bash
uv tool install google-colab-cli --with "jupyter-kernel-client==0.15.0"
colab sessions   # 최초 1회 OAuth — URL을 브라우저에서 열고 코드를 붙여넣는다
```

### 의존성 버전 고정이 필요한 이유

`google-colab-cli` 0.6.0 은 `jupyter-kernel-client` 를 버전 상한 없이 의존한다.
그런데 해당 패키지가 1.0.0 에서 클래스명을 바꿨다:

```
KernelClient  →  JupyterKernelClient
```

그대로 설치하면 1.0.2 가 딸려와 모든 `colab exec` 가 실패한다:

```
AttributeError: module 'jupyter_kernel_client' has no attribute 'KernelClient'
```

세션 생성(`colab new`)과 상태 조회(`colab status`)는 멀쩡히 동작하므로,
실제로 코드를 실행하려 할 때까지 문제가 드러나지 않는다.

`0.15.0` (1.0.0 직전 버전) 으로 고정하면 해결된다. 업스트림에서 고쳐지면 제거할 것.

확인:

```bash
~/.local/share/uv/tools/google-colab-cli/bin/python \
  -c "import jupyter_kernel_client as k; print(k.__version__, hasattr(k,'KernelClient'))"
# 0.15.0 True
```

## 확인된 VM 사양 (CPU 세션)

| 항목 | 값 |
|---|---|
| 디스크 여유 | 94 GB (전체 116 GB) |
| CPU | 2 코어 |
| RAM | 14 GB |
| Python | 3.13.15 |
| 작업 디렉토리 | `/content` |

압축 해제에 원본의 2~3배 용량이 필요하므로, **한 세션당 30~40GB 파트**가 상한이다.
300GB 전체는 8~10회로 나눠 처리한다.

## Colab 계정 — Drive 와 같은 계정이어야 한다

**Colab 은 런타임 계정과 같은 계정의 Drive 만 마운트한다.** 2021년 11월부터의 제약이다
([colabtools#2497](https://github.com/googlecolab/colabtools/issues/2497)). 마운트 화면에서
다른 계정을 고르면 "Close this tab/window" 빈 화면만 뜨고 실패한다.

따라서 **세션(컴퓨팅)·Drive·Colab Pro 결제가 전부 한 계정에 묶인다.** 본 프로젝트는
400GB Drive 가 있는 계정을 Colab CLI 의 주 계정으로 쓴다 (2026-09-20 전환).

### 계정 전환 절차

토큰은 `~/.config/colab-cli/token.json` 하나다. 삭제하지 말고 이름을 바꿔 백업한다.

```bash
mv ~/.config/colab-cli/token.json ~/.config/colab-cli/token.school.json
colab sessions          # 인증 URL → 원하는 계정으로 로그인 → 코드 붙여넣기
colab whoami            # Email 이 바뀌었는지 확인
```

되돌리려면 파일 이름을 다시 바꾸면 된다. 학교 계정 토큰은 `token.school.json` 으로 남아 있다.

전환 후에는 이전 계정의 세션이 CLI 에 보이지 않으므로, **전환 전에 `colab stop` 으로
모두 종료**해야 유닛이 새지 않는다.

### Drive 마운트가 `mount failed` 로 실패하면 — 한 번 더 시도한다

`colab drivemount` 는 브라우저 승인 → "Credentials propagated" → `drive.mount()` 순으로
진행되는데, 첫 시도에서 다음처럼 실패할 수 있다 (2026-09-20 실측):

```
ValueError: mount failed
```

`/root/.config/Google/DriveFS/Logs/drive_fs.txt` 를 보면 원인이 나온다:

```
metadata_server_credential.cc: Failed to request .../guest-attributes/auth/user-id
HTTP code: 404 → Account context authorization failed → CANNOT_START_CORE
```

DriveFS 가 시작될 때 VM 메타데이터 서버에 자격증명이 **아직 기록되지 않은** 경쟁 조건이다.
타임아웃(`QueryManager timed out`)도, 도메인 정책 차단도 아니다. 몇 초 뒤 같은 명령을
다시 실행하면 자격증명이 이미 있어 URL 단계 없이 바로 `Mounted at /content/drive` 가 된다.

확인 방법 (VM 에서): `auth/user-id` 가 200 이면 재시도만 하면 된다.

```python
urllib.request.urlopen(urllib.request.Request(
    "http://172.28.0.1:8009/computeMetadata/v1/instance/guest-attributes/auth/user-id",
    headers={"Metadata-Flavor": "Google"})).status   # 200
```

## 학습 세션에서 OmniFall 데이터 준비

Drive 에는 zip 원본(30GB)만 둔다. 학습 세션마다 로컬 디스크로 풀어서 쓴다 —
Drive FUSE 위에서 직접 해제하거나 수천 개 클립을 읽으면 매우 느리다.

```bash
colab new -s train --gpu T4
colab drivemount -s train            # 사용자가 직접 (세션마다)
colab upload -s train scripts/prepare_omnifall.py /content/prepare_omnifall.py
echo 'import subprocess; subprocess.run("python /content/prepare_omnifall.py --zips /content/drive/MyDrive/falldata/omnifall --out /content/omnifall --staging /content/zips", shell=True)' \
  | colab exec -s train --timeout 3600
```

결과: `/content/omnifall/<데이터셋>/clips/...` 와 `/content/omnifall/manifest.csv`
(컬럼 `dataset,clip,path,label,start,end,subject,cam,source_zip`).

- 긴 형식 중복 클립은 추출하지 않으므로 디스크는 zip 합계의 약 절반이다.
- 각 zip 의 라벨 CSV 와 행 수·라벨 분포를 대조하며, 불일치가 있으면 종료 코드 1 과 ⚠ 표시.
- `--only Cauca_fall.zip` 처럼 일부만 풀어 파이프라인을 먼저 검증할 수 있다.

### Zenodo → Colab 전송 속도 (실측 2026-09-20)

`scripts/fetch_omnifall.py` 단일 연결: 시작 1MB/s → 수십 초 뒤 2~3MB/s (TCP 램프업).
30GB 기준 약 3시간. `colab exec --timeout 21600` 으로 클라이언트 타임아웃을 넉넉히 잡고
`nohup` 으로 분리해 로그 파일에 기록한다. 커널이 이 셀을 실행하는 동안 같은 세션의
`exec` 는 대기열에 걸리므로, 병행 작업은 `colab console`(별도 셸)로 한다.

## rclone → Drive 업로드가 100% 에서 되풀이 재시작되면

증상: `rclone copyto` 가 100% 에 도달한 뒤 통계 총량이 2배·3배로 늘며 처음부터 다시 올린다.
INFO 로그에는 오류가 없다. `-vv` 로 보면 원인이 나온다:

```
pacer: low level retry 1/20 (error googleapi: Error 403: Quota exceeded for quota metric 'Queries'
and limit 'Requests per minute' of service 'drive.googleapis.com' for consumer 'project_number:202264815644'
```

`202264815644` 는 **rclone 의 공용 client_id** 다. 전 세계 rclone 사용자가 나눠 쓰는 분당 요청
쿼터에 걸리면 업로드 마무리 호출이 403 을 받고, rclone 은 저수준 재시도로 업로드를 다시 시작한다.
1.8GB 파일이 네 번 재시작 뒤 5회차에 성공했다 (2026-09-21). 파일이 작을수록(수백 MB) 통과 확률이 높다.

**해법: 자체 OAuth client_id 를 만든다** — https://rclone.org/drive/#making-your-own-client-id.
rclone 이 "shared client_id 가 2026년 중 폐기된다" 고 알리는 것과 같은 문제다. 임시 완화책은
`--tpslimit 2 --drive-chunk-size 64M` 로 요청 수를 줄이는 것.

## 세션 운영

```bash
colab new -s omnifall              # CPU (공개 데이터셋 전송·전처리용)
colab drivemount -s omnifall       # Drive 마운트 — 사용자가 직접, 세션마다 다시
colab new -s train --gpu T4        # GPU (학습용)
colab exec -s omnifall -f x.py     # 로컬 스크립트를 VM에서 실행
colab stop -s omnifall             # 반드시 종료
```

- **커널 상태는 `exec` 호출 간 유지된다.** 데이터를 한 번 로드해 두고 여러 번 분석할 수 있다.
- **세션을 방치하면 컴퓨팅 유닛이 24시간까지 계속 소진된다.** 작업이 끝나면 `colab stop`.
- GPU 할당은 계정 등급 제한이 있다. 400 에러면 T4로 낮추거나 CPU로 돌린다.
- `drivemount`, `auth`, `repl`, `console` 은 TTY가 필요해 사람이 직접 실행해야 한다.

## AI Hub API Key

키를 소스에 넣거나 채팅에 붙여넣지 않는다. 로컬 파일에 두고 읽는다:

```bash
echo 'AIHUB_KEY=발급받은-키' > ~/.aihub_key   # .gitignore 로 차단됨
chmod 600 ~/.aihub_key
```
