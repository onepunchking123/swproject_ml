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

## 세션 운영

```bash
colab new -s falldata              # CPU (탐색·전처리용)
colab new -s train --gpu T4        # GPU (학습용)
colab exec -s falldata -f x.py     # 로컬 스크립트를 VM에서 실행
colab stop -s falldata             # 반드시 종료
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
