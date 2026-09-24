# GCP 서울 VM 으로 AI Hub → Drive 중계

_작성: 2026-09-20. 맥북 회선 실측 후 확정된 경로._

## 왜 필요한가

AI Hub 는 한국 IP 에서만 내려받을 수 있고, Colab VM 은 미국이다. 맥북(한국 IP)이 중계할 수
있지만 실측 결과 이 회선은 **다운로드 총량 3MB/s, 업로드 0.5MB/s** 다.

| 작업 | 맥북 회선 | 서울 VM (기가비트) |
|---|---|---|
| 라벨 142MB | 5분 | 수 초 |
| VS.zip 55GB | ~30시간 | AI Hub 연결당 상한에 따라 수십 분~수 시간 |
| TS 436GB (filekey 5개) | ~10일 | 5 병렬, 수 시간~반나절 |

서울 VM 은 한국 IP 이고, 회선이 빠르고, 맥북이 잠들어도 돌아간다. 비용은 `e2-small` 기준
시간당 약 $0.02 — 하루 켜둬도 $0.5 수준이라 신규 크레딧 $300 안에서 무시할 만하다.

스크립트는 그대로 쓴다: `scripts/aihub_stream.py` 는 디스크를 쓰지 않으므로 VM 디스크는
기본 10GB 로 충분하다.

## 절차

### 0. GCP 준비 (사용자, 1회)

1. [console.cloud.google.com](https://console.cloud.google.com) 에서 프로젝트 생성 (예: `fall-detection`)
2. 결제 계정 연결 — 신규 가입 시 $300 크레딧
3. Compute Engine API 활성화
4. 로컬에 `gcloud` 설치: `brew install --cask google-cloud-sdk` → `gcloud init` (브라우저 로그인)

### 1. VM 생성

```bash
gcloud compute instances create aihub-relay \
  --zone=asia-northeast3-a --machine-type=e2-small \
  --image-family=debian-12 --image-project=debian-cloud \
  --boot-disk-size=10GB
```

### 2. 도구 설치 (VM 안)

```bash
gcloud compute ssh aihub-relay --zone=asia-northeast3-a
# ── VM 안 ──
sudo apt-get update -qq && sudo apt-get install -y -qq python3 curl tmux
curl https://rclone.org/install.sh | sudo bash
```

### 3. 자격증명 전달 (사용자가 직접)

두 파일이 필요하다. 둘 다 비밀이므로 채팅에 붙이지 말고 `scp` 로 보낸다.

```bash
gcloud compute scp ~/.aihub_key aihub-relay:~/.aihub_key --zone=asia-northeast3-a
gcloud compute scp ~/.config/rclone/rclone.conf aihub-relay:~/rclone.conf --zone=asia-northeast3-a
gcloud compute scp scripts/aihub_stream.py aihub-relay:~/aihub_stream.py --zone=asia-northeast3-a
# VM 안에서
mkdir -p ~/.config/rclone && mv ~/rclone.conf ~/.config/rclone/ && chmod 600 ~/.aihub_key ~/.config/rclone/rclone.conf
rclone about gdrive:     # 5 TiB 가 보이면 정상
```

rclone 토큰은 맥북에서 발급받은 것을 그대로 쓴다 (refresh token 이 들어 있어 VM 에서도 갱신된다).
작업이 끝나면 VM 을 삭제하므로 토큰도 함께 사라진다.

### 4. 실행 — 병렬은 filekey 단위로

AI Hub 응답은 요청마다 즉석 생성되는 tar 라 한 filekey 안에서는 병렬·이어받기가 불가능하다.
대신 filekey 가 다르면 독립 스트림이므로 **프로세스를 filekey 마다 하나씩** 띄운다.

```bash
tmux new -d -s relay
# VS.zip (55GB) 먼저 — 단독 스트림 속도로 AI Hub 서버측 상한을 확인한다
tmux send-keys -t relay 'python3 ~/aihub_stream.py --filekey 531137 2>&1 | tee ~/vs.log' Enter
```

VS 속도가 확인되면 TS 5개를 동시에:

```bash
for fk in 531131 531132 531133 531134 531135; do
  nohup python3 ~/aihub_stream.py --filekey $fk > ~/ts_$fk.log 2>&1 &
done
tail -f ~/ts_*.log
```

| filekey | 파일 | 크기 |
|---|---|---|
| 531137 | VS.zip | 55 GB |
| 531131~531134 | TS.z01~z04 | 각 100 GB |
| 531135 | TS.zip | 36 GB |
| 531136 / 531138 | TL.zip / VL.zip | 126 / 16 MB (맥북에서 이미 전송) |

결과는 `gdrive:falldata/aihub/<filekey>/<파일>.partNNN` 와 `parts.json` 으로 남는다.

### 5. 정리

```bash
gcloud compute instances delete aihub-relay --zone=asia-northeast3-a
```

VM 을 지우면 과금이 멈추고 자격증명 사본도 사라진다.

## Colab 에서 원본 복원

`parts.json` 의 순서대로 이어붙이면 원본이 된다. Drive → `/content` 로 복사 후:

```bash
cat /content/aihub/531137/VS.zip.part* > /content/VS.zip      # part 번호는 3자리 0 채움이라 사전순 = 숫자순
unzip -tq /content/VS.zip                                       # 무결성 확인
```

TS 는 분할 zip(`TS.z01~z04 + TS.zip`, 436GB)이라 Colab 디스크(242GB)에 통째로 둘 수 없다.
`cat TS.z01 TS.z02 TS.z03 TS.z04 TS.zip | bsdtar -xf - --include '...'` 로 스트리밍 해제하면서
필요한 것만 남기거나(라벨 JSON, 클립 일부), ffmpeg 로 640px 트랜스코딩해 축소본을 Drive 에 둔다.
이 부분은 VS 로 Stage 1 을 먼저 돌려본 뒤 TS 가 정말 필요한지 판단하고 설계한다.

## 주의

- AI Hub 데이터는 재배포 금지다. Drive 폴더를 공유 링크로 열지 않는다.
- rclone 의 공용 client_id 는 2026년 중 폐기 예정이라는 알림이 있다. 장기적으로는
  [자체 client_id](https://rclone.org/drive/#making-your-own-client-id) 를 만든다.
