# 인공지능과 데이터분석환경 | PyTorch Tensor 실습

Python 가상환경 구축부터 Tensor 생성, 인덱싱, 구조 변경, 벡터 연산과 수치 안정성까지 정리한 실습 저장소입니다. 최종 제출한 PDF 보고서와 노트북, 실행 출력이 포함된 학습용 노트북을 함께 보관합니다.

## 바로 보기

- [최종 제출 PDF·노트북](submissions/README.md)
- [실습 전체 코드와 저장된 실행 결과](docs/code-and-results.md)
- [Part 1~5 상세 해설](01-pytorch-tensor-basics/README.md)
- [학습용 노트북](01-pytorch-tensor-basics/notebooks/pytorch-tensor-basics.ipynb)
- [설치 패키지 목록](requirements.txt)

## 실습 구성

| 구분 | 실습 내용 | 확인한 내용 |
|---|---|---|
| Part 1 환경 구축 | Python 3.11, uv, 가상환경, 패키지 설치 | 인터프리터와 프로젝트 의존성 분리, CPU/CUDA 환경 확인 |
| Part 2 Tensor 생성 | scalar, vector, matrix, ones, full, randn | 차원·shape·dtype·device 구분 |
| Part 3 데이터 선택 | 인덱싱, 슬라이싱, Ellipsis, 불리언 마스크 | 축별 선택 결과와 범위·마스크 오류 원인 |
| Part 4 구조 변경 | reshape/view, squeeze/unsqueeze, transpose/permute, expand/repeat | 원소 수 보존, 축 순서, 메모리 공유와 복사 차이 |
| Part 5 연산 | 내적, 벡터화, broadcasting, 행렬 곱, Softmax | 연산별 shape 규칙과 수치 안정성 |

## 주요 결과와 학습

### Tensor의 모양을 기준으로 연산 이해하기

`torch.tensor([[1, 2, 3], [4, 5, 6]])`의 shape은 `(2, 3)`입니다. `t[0]`은 `(3,)`의 첫 행, `t[:, 1:]`은 `(2, 2)`의 부분 행렬을 반환합니다. `t[-5]`와 길이가 맞지 않는 행 마스크는 오류를 일으킵니다. 인덱스 범위와 마스크가 적용되는 축을 확인하는 방식으로 원인을 구분했습니다.

### 구조 변경과 메모리 차이

`reshape`와 `view`로 동일한 값 배치를 만들고, `unsqueeze`와 `squeeze`로 길이가 1인 축을 추가·제거했습니다. `expand`와 `repeat`는 같은 값을 보여도 저장공간 동작이 다릅니다. 기록된 `data_ptr()` 비교에서 `expand`는 원본과 같은 주소, `repeat`는 다른 주소를 사용했습니다. 값의 모양뿐 아니라 저장 방식도 함께 확인했습니다.

### 연산 결과 비교

`[1, 2, 3]`과 `[4, 5, 6]`의 내적은 Python 반복문, NumPy, PyTorch에서 모두 `32`입니다. `(3, 1)`과 `(3,)`의 덧셈은 broadcasting으로 `(3, 3)`이 됩니다. `(100, 128) @ (128, 256)`은 `(100, 256)`을 반환하며, 안쪽 차원이 다른 행렬 곱은 실패합니다. 전치를 적용해 차원을 맞추는 과정까지 노트북에 포함했습니다.

### 벡터화 성능 비교

저장된 백만 원소 제곱 실험에서 리스트 컴프리헨션은 평균 `49.8 ms`, NumPy는 `688 μs`로 기록되었습니다. 이 기록에서는 약 72배 차이가 납니다. 측정값은 하드웨어·버전·부하에 따라 달라지므로 일반적인 성능 보장으로 해석하지 않습니다. Python 수준 반복 비용과 배열 연산의 차이를 관찰한 실험입니다.

### Softmax 수치 안정성

`[1000, 2000, 3000]`을 그대로 지수 함수에 넣으면 overflow와 `nan`이 발생합니다. 최댓값을 뺀 뒤 계산하면 `[0., 0., 1.]`, 합계 `1.0`을 얻습니다. 수학적으로 같은 비율을 유지하면서 계산 가능한 범위로 값을 옮기는 이유를 확인했습니다.

## 실행 방법

저장소 루트에서 실행합니다.

```powershell
py -3.11 -m pip install uv
uv venv .venv --python=3.11
.venv\Scripts\activate
uv pip install -r requirements.txt
uv run jupyter lab
```

JupyterLab에서 `01-pytorch-tensor-basics/notebooks/pytorch-tensor-basics.ipynb`를 엽니다. 노트북에는 오류 원인을 확인하기 위한 예외 셀이 포함되어 있습니다. 오류 셀을 확인한 뒤 다음 셀로 진행하면 정상 예제와 비교할 수 있습니다. CUDA 사용 가능 여부에 따라 장치를 선택하며, 기존 기록은 CPU 환경입니다.

## 파일 구성

| 경로 | 내용 |
|---|---|
| `submissions/` | 최종 제출 PDF와 ipynb 원본 |
| `01-pytorch-tensor-basics/` | 상세 실습 해설·학습용 노트북·의존성 |
| `docs/code-and-results.md` | 노트북 코드·텍스트 출력·예외 메시지 전체 정리 |
| `requirements.txt` | 과제에서 사용한 기본 의존성 목록 |

최종 제출본 두 파일은 기존 보관 파일과 바이트 단위로 비교해 동일함을 확인했습니다. 학습용 노트북의 저장된 출력은 보존했으며, 이번 정리에서는 노트북을 다시 실행하지 않았습니다.
