# PyTorch Tensor 실습: 코드와 실행 결과

노트북에 저장된 코드와 출력을 순서대로 정리했습니다. 아래 결과는 기존 실습 기록이며, 이번 저장소 정리 과정에서 새로 측정한 값은 아닙니다. 난수·성능 측정값은 실행 환경에 따라 달라집니다.

# PyTorch Tensor 기초 실습

이 노트북은 인공지능과 데이터분석환경 실습의 Part 2~Part 5를 주제와 번호 순서대로 정리한 공개용 사본입니다. 개인 이름과 학번은 제외했습니다.

셀에는 Tensor 생성과 속성, 인덱싱·슬라이싱, 구조 변경, 벡터·행렬 연산 및 Softmax 실습 코드와 실행 출력이 포함되어 있습니다. 일부 셀은 과제에서 요구한 오류를 재현합니다.

Part 1의 환경 설치 과정과 전체 요구 패키지 목록은 이 폴더의 README와 `requirements.txt`에서 확인할 수 있습니다. 실행 환경은 Windows, Python 3.11.9, PyTorch 2.15.0+cpu이며 CUDA는 사용할 수 없어 CPU로 연산했습니다.

## Part 2. Tensor 생성과 속성

### 2.1 Tensor 생성, 차원, shape, dtype

스칼라는 원소가 하나이고 차원이 0인 텐서입니다. 벡터는 한 축, 행렬은 두 축, 3차원 텐서는 세 축을 가집니다. 3차원 예시에서 바깥 대괄호는 첫 번째 축의 묶음, 그 안의 대괄호는 두 번째 축, 가장 안쪽 대괄호는 세 번째 축의 원소 묶음에 대응합니다. 정수로 만든 값은 기본적으로 `torch.int64`, 소수점이 있는 값은 기본적으로 `torch.float32`가 됩니다. 정수형은 인덱스나 개수처럼 정수 값을 정확히 표현할 때 적합하고, 실수형은 소수 값과 미분을 포함하는 모델 계산에 사용됩니다.

```python
import sys
import torch
import numpy as np

print("Python:", sys.version.split()[0])
print("PyTorch:", torch.__version__)
print("CUDA 사용 가능:", torch.cuda.is_available())
print("CUDA 버전:", torch.version.cuda)
print("실습 장치:", "cuda" if torch.cuda.is_available() else "cpu")
```

```text
Python: 3.11.9
PyTorch: 2.15.0+cpu
CUDA 사용 가능: False
CUDA 버전: None
실습 장치: cpu

```

```python
scalar = torch.tensor(7.0)
vector = torch.tensor([1, 2, 3])
matrix = torch.tensor([[1, 2, 3], [4, 5, 6]])
tensor_3d = torch.tensor([
    [[1, 2], [3, 4]],
    [[5, 6], [7, 8]]
])

for name, tensor in [
    ("스칼라", scalar),
    ("벡터", vector),
    ("행렬", matrix),
    ("3차원 텐서", tensor_3d),
]:
    print(f"{name}:")
    print(tensor)
    print("차원 수:", tensor.dim(), "| shape:", tuple(tensor.shape), "| dtype:", tensor.dtype)
    print()
```

```text
스칼라:
tensor(7.)
차원 수: 0 | shape: () | dtype: torch.float32

벡터:
tensor([1, 2, 3])
차원 수: 1 | shape: (3,) | dtype: torch.int64

행렬:
tensor([[1, 2, 3],
        [4, 5, 6]])
차원 수: 2 | shape: (2, 3) | dtype: torch.int64

3차원 텐서:
tensor([[[1, 2],
         [3, 4]],

        [[5, 6],
         [7, 8]]])
차원 수: 3 | shape: (2, 2, 2) | dtype: torch.int64


```

### 2.2 `ones`, `full`, `randn`

`torch.ones(shape)`는 지정한 shape을 1로 채우고, `torch.full(shape, value)`는 모든 원소를 지정 값으로 채웁니다. `torch.randn(shape)`은 표준정규분포(평균 0, 표준편차 1)에서 실수 난수를 생성합니다. shape 인자를 바꾸면 행·열 수가 바뀌고, `full`의 두 번째 인자를 바꾸면 채움 값이 바뀝니다. 난수는 실행할 때마다 달라질 수 있습니다.

```python
ones_tensor = torch.ones((3, 2))
full_tensor = torch.full((2, 3), -2)
random_tensor = torch.randn((2, 3))

for name, tensor in [
    ("ones", ones_tensor),
    ("full", full_tensor),
    ("randn", random_tensor),
]:
    print(f"{name}:")
    print(tensor)
    print("shape:", tuple(tensor.shape), "| dtype:", tensor.dtype)
    print()
```

```text
ones:
tensor([[1., 1.],
        [1., 1.],
        [1., 1.]])
shape: (3, 2) | dtype: torch.float32

full:
tensor([[-2, -2, -2],
        [-2, -2, -2]])
shape: (2, 3) | dtype: torch.int64

randn:
tensor([[ 1.3945, -0.3429, -0.2287],
        [-0.1075,  0.0571,  0.3255]])
shape: (2, 3) | dtype: torch.float32


```

### 2.3 장치와 CUDA 환경 확인

`tensor.device`는 텐서가 있는 장치를 보여줍니다. CUDA를 사용할 수 있으면 장치 이름을 조회하고 텐서를 CUDA로 옮깁니다. 사용할 수 없으면 과제 지침에 맞춰 CPU를 선택합니다. `torch.version.cuda`가 `None`인 CPU 전용 빌드에서는 CUDA 런타임이 포함되지 않았다는 뜻입니다.

```python
x = torch.tensor([1, 2, 3])

print("처음 텐서의 장치:", x.device)
print("CUDA 사용 가능:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU 이름:", torch.cuda.get_device_name(0))
    moved = x.to("cuda")
else:
    print("GPU 이름: 없음 (CPU 환경)")
    moved = x.to("cpu")

print("이동 후 텐서의 장치:", moved.device)
```

```text
처음 텐서의 장치: cpu
CUDA 사용 가능: False
GPU 이름: 없음 (CPU 환경)
이동 후 텐서의 장치: cpu

```

### 2.4 CPU와 CUDA 텐서의 장치 불일치

PyTorch는 서로 다른 장치에 있는 텐서끼리 바로 연산하지 못합니다. CUDA가 있는 경우 아래 셀은 CPU 텐서와 CUDA 텐서를 더해 `RuntimeError`를 재현한 뒤, 한쪽을 같은 장치로 옮겨 해결합니다. 현재처럼 CUDA가 없는 환경에서는 이 오류를 실제로 만들 수 없으므로, 해당 제한과 CPU 대체 실행 결과를 출력합니다.

```python
if torch.cuda.is_available():
    cpu_tensor = torch.tensor([1, 2, 3])
    cuda_tensor = torch.tensor([4, 5, 6], device="cuda")
    try:
        cpu_tensor + cuda_tensor
    except RuntimeError as error:
        print("CPU와 CUDA tensor를 직접 더한 오류:")
        print(type(error).__name__ + ":", error)
    print("두 tensor를 CUDA로 맞춘 결과:", cpu_tensor.to("cuda") + cuda_tensor)
else:
    print("CUDA 사용 불가: CPU/CUDA 혼합 연산 오류는 이 환경에서 재현할 수 없습니다.")
    print("대신 torch.cuda.is_available() == False와 CPU 장치 선택을 기록합니다.")
```

```text
CUDA 사용 불가: CPU/CUDA 혼합 연산 오류는 이 환경에서 재현할 수 없습니다.
대신 torch.cuda.is_available() == False와 CPU 장치 선택을 기록합니다.

```

## Part 3. 인덱싱과 슬라이싱

### 3.1 행 선택과 shape

`t`는 2행 3열입니다. `t[0]`은 첫 번째 행 전체, `t[1]`은 두 번째 행 전체를 가져옵니다.

```python
t = torch.tensor([
    [1, 2, 3],
    [4, 5, 6]
])

print("전체 텐서:\n", t)
print("shape:", t.shape)
print("t[0] 첫 번째 행:", t[0])
print("t[1] 두 번째 행:", t[1])
```

```text
전체 텐서:
 tensor([[1, 2, 3],
        [4, 5, 6]])
shape: torch.Size([2, 3])
t[0] 첫 번째 행: tensor([1, 2, 3])
t[1] 두 번째 행: tensor([4, 5, 6])

```

### 3.2 슬라이싱과 음수 인덱스

` t[:, 1:] `는 모든 행에서 1번 열부터 끝까지, `t[1:, :2]`는 두 번째 행부터 첫 두 열을 가져옵니다. `t[::2]`와 `t[::3]`는 행 축을 각각 2칸·3칸 간격으로 선택합니다. `t[-1]`은 마지막 행, `t[:, -2]`는 모든 행의 끝에서 두 번째 열입니다. `t[-5]`는 행 축 크기가 2인데 5번째 항목을 요구하므로 `IndexError`가 나야 하며, 이 오류는 음수 인덱스가 무제한으로 허용되는 것이 아님을 보여줍니다.

```python
slices = {
    "t[:, 1:]": t[:, 1:],
    "t[1:, :2]": t[1:, :2],
    "t[::2]": t[::2],
    "t[::3]": t[::3],
    "t[-1]": t[-1],
    "t[:, -2]": t[:, -2],
}

for expression, result in slices.items():
    print(f"{expression} 결과:")
    print(result)
    print()
```

```text
t[:, 1:] 결과:
tensor([[2, 3],
        [5, 6]])

t[1:, :2] 결과:
tensor([[4, 5]])

t[::2] 결과:
tensor([[1, 2, 3]])

t[::3] 결과:
tensor([[1, 2, 3]])

t[-1] 결과:
tensor([4, 5, 6])

t[:, -2] 결과:
tensor([2, 5])


```

```python
t[-5]
```

```text
IndexError: index -5 is out of bounds for dimension 0 with size 2
```

### 3.3 4차원 텐서와 Ellipsis

`...`(Ellipsis)는 생략된 차원 전체를 뜻합니다. 4차원 텐서에서 `x[1, ..., 1]`은 첫 번째 축에서 인덱스 1, 마지막 축에서 인덱스 1을 고정하고 가운데 두 축은 모두 남깁니다. `x[0, ..., 0]`도 같은 방식으로 선택합니다.

```python
x = torch.randn(2, 2, 2, 2)

print("4차원 텐서:\n", x)
print("x[1, ..., 1]:\n", x[1, ..., 1])
print("x[0, ..., 0]:\n", x[0, ..., 0])
```

```text
4차원 텐서:
 tensor([[[[ 0.3697, -0.8285],
          [-0.7148,  1.8636]],

         [[ 0.7422,  1.7179],
          [-0.7483,  1.2941]]],


        [[[-1.2970,  1.1398],
          [ 0.9514,  0.6407]],

         [[ 0.2536,  1.1964],
          [ 0.1496,  0.3034]]]])
x[1, ..., 1]:
 tensor([[1.1398, 0.6407],
        [1.1964, 0.3034]])
x[0, ..., 0]:
 tensor([[ 0.3697, -0.7148],
        [ 0.7422, -0.7483]])

```

### 3.4 불리언 마스크

`v > 3`, `v % 2 == 0`, `v % 2 == 1`은 조건을 만족하는 원소를 선택합니다. 마스크의 첫 차원 길이는 대상 텐서의 첫 차원 길이와 맞아야 합니다. 아래의 길이 3 마스크를 행이 2개인 `t`에 적용하면 첫 축 크기가 맞지 않아 `IndexError`가 발생합니다. 이어지는 길이 2 마스크는 정상적으로 행을 선택합니다.

```python
print("t > 3:", t[t > 3])
print("짝수:", t[t % 2 == 0])
print("홀수:", t[t % 2 == 1])

```

```text
t > 3: tensor([4, 5, 6])
짝수: tensor([2, 4, 6])
홀수: tensor([1, 3, 5])

```

```python
wrong_mask = torch.tensor([True, False, True])
t[wrong_mask]
```

```text
IndexError: The shape of the mask [3] at index 0 does not match the shape of the indexed tensor [2, 3] at index 0
```

```python
row_mask = torch.tensor([True, False])
print(t[row_mask])
```

```text
tensor([[1, 2, 3]])

```

## Part 4. Tensor 구조 변경

### 4.1 `reshape`, `view`, 평면화와 잘못된 크기

`torch.arange(12)`에는 원소 12개가 있으므로 `(3, 4)`로 바꿀 수 있습니다. `reshape`는 필요한 경우 새 메모리 배치를 만들 수 있고, `view`는 원본 메모리가 호환될 때 같은 저장 공간을 공유하는 모양 변경을 합니다. `reshape(-1)`의 `-1`은 나머지 원소 수로 해당 차원을 자동 계산하라는 뜻입니다. `reshape(-2)`는 `-1`처럼 허용된 자동 계산 표기가 아니므로 오류가 납니다.

```python
# arange(12)를 3x4로 변경
base12 = torch.arange(12)
print("reshape(3, 4):\n", base12.reshape(3, 4))
print("view(3, 4):\n", base12.view(3, 4))

# 차원 추가와 제거
print("unsqueeze(0):", base12.unsqueeze(0).shape)
print("unsqueeze(1):", base12.unsqueeze(1).shape)
print("unsqueeze(-2):", base12.unsqueeze(-2).shape)
print("다시 squeeze:", base12.unsqueeze(-2).squeeze().shape)

# 전치와 permute
m = base12.reshape(3, 4)
print("m.T shape:", m.T.shape)

tensor_3d = torch.randn(3, 4, 5)
print("원래 shape:", tensor_3d.shape)
print("permute(0, 2, 1) shape:", tensor_3d.permute(0, 2, 1).shape)
```

```text
reshape(3, 4):
 tensor([[ 0,  1,  2,  3],
        [ 4,  5,  6,  7],
        [ 8,  9, 10, 11]])
view(3, 4):
 tensor([[ 0,  1,  2,  3],
        [ 4,  5,  6,  7],
        [ 8,  9, 10, 11]])
unsqueeze(0): torch.Size([1, 12])
unsqueeze(1): torch.Size([12, 1])
unsqueeze(-2): torch.Size([1, 12])
다시 squeeze: torch.Size([12])
m.T shape: torch.Size([4, 3])
원래 shape: torch.Size([3, 4, 5])
permute(0, 2, 1) shape: torch.Size([3, 5, 4])

```

```python
base = torch.arange(1, 7)

print("원본:", base)
print("reshape 결과:\n", base.reshape(2, 3))
print("view 결과:\n", base.view(2, 3))
```

```text
원본: tensor([1, 2, 3, 4, 5, 6])
reshape 결과:
 tensor([[1, 2, 3],
        [4, 5, 6]])
view 결과:
 tensor([[1, 2, 3],
        [4, 5, 6]])

```

```python
matrix = base.reshape(2, 3)

print("원본:\n", matrix)
print("flatten 결과:", matrix.flatten())
```

```text
원본:
 tensor([[1, 2, 3],
        [4, 5, 6]])
flatten 결과: tensor([1, 2, 3, 4, 5, 6])

```

```python
print("자동 계산:", base.reshape(-1))
print("2×3 모양:", base.reshape(2, 3))
```

```text
자동 계산: tensor([1, 2, 3, 4, 5, 6])
2×3 모양: tensor([[1, 2, 3],
        [4, 5, 6]])

```

```python
base.reshape(-2)
```

```text
RuntimeError: invalid shape dimension -2 at index 0 of shape [-2]
```

### 4.2 `unsqueeze`와 `squeeze`

`unsqueeze(dim)`은 원소를 바꾸지 않고 길이 1인 차원을 추가합니다. 음수 차원은 뒤에서부터 센 위치입니다. `squeeze()` 또는 특정 축의 `squeeze(dim)`는 길이가 1인 축을 제거합니다. 1차원 텐서에서 `unsqueeze(2)`는 허용 범위를 벗어나므로 `IndexError`가 발생합니다.

```python
print("원본 shape:", matrix.shape)
print("unsqueeze(0):", matrix.unsqueeze(0).shape)
print("unsqueeze(1):", matrix.unsqueeze(1).shape)
print("unsqueeze(2):", matrix.unsqueeze(2).shape)
```

```text
원본 shape: torch.Size([2, 3])
unsqueeze(0): torch.Size([1, 2, 3])
unsqueeze(1): torch.Size([2, 1, 3])
unsqueeze(2): torch.Size([2, 3, 1])

```

```python
expanded = matrix.unsqueeze(1)

print("추가한 뒤:", expanded.shape)
print("squeeze(1) 적용 뒤:", expanded.squeeze(1).shape)
```

```text
추가한 뒤: torch.Size([2, 1, 3])
squeeze(1) 적용 뒤: torch.Size([2, 3])

```

```python
base12.unsqueeze(2)
```

```text
IndexError: Dimension out of range (expected to be in range of [-2, 1], but got 2)
```

### 4.3 전치와 `permute`

2차원 행렬의 `.T` 또는 `transpose(0, 1)`은 행과 열의 축을 맞바꿉니다. `permute(0, 2, 1)`은 결과의 0·1·2번 축이 원본의 0·2·1번 축을 차례로 사용하도록 축의 순서를 바꿉니다. 따라서 `(3, 4, 5)`는 `(3, 5, 4)`가 됩니다.

```python
print("원본:\n", matrix)
print("transpose(0, 1):\n", matrix.transpose(0, 1))
print("permute(1, 0):\n", matrix.permute(1, 0))
```

```text
원본:
 tensor([[1, 2, 3],
        [4, 5, 6]])
transpose(0, 1):
 tensor([[1, 4],
        [2, 5],
        [3, 6]])
permute(1, 0):
 tensor([[1, 4],
        [2, 5],
        [3, 6]])

```

### 4.4 `expand`와 `repeat`

두 함수는 겉으로 같은 반복 결과를 만들 수 있습니다. `expand`는 크기가 1인 축을 stride 조정으로 늘려 보이게 하므로 대체로 새 원소 저장 공간을 만들지 않습니다. `repeat`은 실제 값을 복제해 새 저장 공간을 사용합니다. 따라서 `expand` 결과에 값을 쓰면 같은 원본 위치를 공유하는 주의가 필요하고, 독립된 복사본이 필요하면 `repeat`이 적절합니다.

```python
row = torch.tensor([[1, 2, 3]])

print("expand 결과:\n", row.expand(2, 3))
print("repeat 결과:\n", row.repeat(2, 1))
```

```text
expand 결과:
 tensor([[1, 2, 3],
        [1, 2, 3]])
repeat 결과:
 tensor([[1, 2, 3],
        [1, 2, 3]])

```

```python
expanded = row.expand(2, 3)
repeated = row.repeat(2, 1)

print("expand가 원본 저장공간을 공유하나요?", row.data_ptr() == expanded.data_ptr())
print("repeat이 원본 저장공간을 공유하나요?", row.data_ptr() == repeated.data_ptr())
```

```text
expand가 원본 저장공간을 공유하나요? True
repeat이 원본 저장공간을 공유하나요? False

```

### 4.5 `resize_`, `unsqueeze`, `expand` 연속 적용

첫 예시는 8개 원소를 `(2, 4)`에 배치한 다음 축 1에 길이 1을 추가해 `(2, 1, 4)`로 만들고, 이를 `(2, 2, 4)`로 확장합니다. 두 번째 예시는 같은 순서를 `(3, 4) → (3, 1, 4) → (3, 5, 4)`로 확인합니다. `resize_`는 in-place로 크기와 저장 공간 배치를 바꾸는 함수이므로, 일반적인 shape 변환에서는 `reshape` 사용이 더 안전합니다.

```python
# (2, 4) → (2, 1, 4) → (2, 2, 4)
tensor_a = torch.arange(1, 9).resize_(2, 4)
tensor_a_expanded = tensor_a.unsqueeze(1).expand(2, 2, 4)

print("원래 shape:", tensor_a.shape)
print("unsqueeze(1) shape:", tensor_a.unsqueeze(1).shape)
print("expand 결과 shape:", tensor_a_expanded.shape)
print(tensor_a_expanded)

# 크기를 바꾼 예시: (3, 4) → (3, 1, 4) → (3, 5, 4)
tensor_b = torch.arange(1, 13).resize_(3, 4)
tensor_b_expanded = tensor_b.unsqueeze(1).expand(3, 5, 4)

print("두 번째 원래 shape:", tensor_b.shape)
print("unsqueeze(1) shape:", tensor_b.unsqueeze(1).shape)
print("두 번째 expand 결과 shape:", tensor_b_expanded.shape)
```

```text
원래 shape: torch.Size([2, 4])
unsqueeze(1) shape: torch.Size([2, 1, 4])
expand 결과 shape: torch.Size([2, 2, 4])
tensor([[[1, 2, 3, 4],
         [1, 2, 3, 4]],

        [[5, 6, 7, 8],
         [5, 6, 7, 8]]])
두 번째 원래 shape: torch.Size([3, 4])
unsqueeze(1) shape: torch.Size([3, 1, 4])
두 번째 expand 결과 shape: torch.Size([3, 5, 4])

```

## Part 5. 기본 연산

### 5.1 내적

내적은 같은 위치 원소끼리 곱한 뒤 모두 더합니다. 여기서는 `1×4 + 2×5 + 3×6 = 32`입니다. `zip` 반복문, 인덱스 반복문, NumPy의 `np.dot`과 `@`, PyTorch의 `torch.dot`과 `@`가 같은 값을 내는지 비교합니다.

```python
a_list = [1, 2, 3]
b_list = [4, 5, 6]

# zip을 이용한 for문
dot_zip = 0
for x, y in zip(a_list, b_list):
    dot_zip += x * y

# 인덱스를 이용한 for문
dot_index = 0
for i in range(len(a_list)):
    dot_index += a_list[i] * b_list[i]

# NumPy와 PyTorch
a_np = np.array(a_list)
b_np = np.array(b_list)
a_torch = torch.tensor(a_list)
b_torch = torch.tensor(b_list)

print("zip for문:", dot_zip)
print("인덱스 for문:", dot_index)
print("np.dot:", np.dot(a_np, b_np))
print("NumPy @:", a_np @ b_np)
print("torch.dot:", torch.dot(a_torch, b_torch))
print("PyTorch @:", a_torch @ b_torch)
```

```text
zip for문: 32
인덱스 for문: 32
np.dot: 32
NumPy @: 32
torch.dot: tensor(32)
PyTorch @: tensor(32)

```

### 5.2 벡터화와 브로드캐스팅

Python 반복문은 원소별 덧셈을 직접 처리하고, NumPy 배열 연산은 배열 전체에 연산을 적용합니다. `a+b`, `a*b`, `a**2`, `exp(a)`, `log(a+1)`을 확인합니다. shape `(3,1)`과 `(3,)`은 뒤쪽 차원부터 비교하면 `(3,1)`과 `(1,3)`으로 맞춰져 결과 shape `(3,3)`을 만듭니다.

```python
a_vec = np.array([1, 2, 3])
b_vec = np.array([4, 5, 6])

# for문으로 원소별 덧셈
sum_for = [a_vec[i] + b_vec[i] for i in range(len(a_vec))]
print("for문 덧셈:", sum_for)

# NumPy 벡터 연산
print("a + b:", a_vec + b_vec)
print("a * b:", a_vec * b_vec)
print("a ** 2:", a_vec ** 2)
print("exp(a):", np.exp(a_vec))
print("log(a + 1):", np.log(a_vec + 1))

# (3, 1)과 (3,) 배열의 브로드캐스팅
column = np.array([[1], [2], [3]])
row = np.array([10, 20, 30])

print("column shape:", column.shape)
print("row shape:", row.shape)
print("브로드캐스팅 결과:\n", column + row)
print("결과 shape:", (column + row).shape)
```

```text
for문 덧셈: [np.int64(5), np.int64(7), np.int64(9)]
a + b: [5 7 9]
a * b: [ 4 10 18]
a ** 2: [1 4 9]
exp(a): [ 2.71828183  7.3890561  20.08553692]
log(a + 1): [0.69314718 1.09861229 1.38629436]
column shape: (3, 1)
row shape: (3,)
브로드캐스팅 결과:
 [[11 21 31]
 [12 22 32]
 [13 23 33]]
결과 shape: (3, 3)

```

### 5.3 100만 원소 제곱의 실행 시간

`%timeit`은 여러 번 실행해 평균 실행 시간을 측정합니다. 리스트 컴프리헨션은 Python이 백만 번 반복하며 각 원소 연산을 처리하는 반면, NumPy는 연속된 배열을 저수준 벡터화 연산으로 계산합니다. 측정값은 환경에 따라 달라지지만 NumPy 방식이 대체로 빠른 이유는 반복문 처리 비용이 작기 때문입니다.

```python
data_list = list(range(1_000_000))
data_array = np.arange(1_000_000)

print("리스트 컴프리헨션:")
%timeit [x ** 2 for x in data_list]

print("NumPy:")
%timeit data_array ** 2
```

```text
리스트 컴프리헨션:
49.8 ms +- 6.17 ms per loop (mean +- std. dev. of 7 runs, 10 loops each)
NumPy:
688 us +- 15.4 us per loop (mean +- std. dev. of 7 runs, 1,000 loops each)

```

### 5.4 PyTorch 벡터 연산과 장치 선택

PyTorch도 `+`, `*`, 거듭제곱과 broadcasting을 지원합니다. `device`를 CUDA 사용 가능 여부에 따라 선택하고, `x` shape `(10000,128)`과 `w` shape `(128,)`의 행렬·벡터 곱 결과 shape `(10000,)`을 확인합니다. GPU가 없으면 같은 코드를 CPU에서 실행합니다.

```python
# PyTorch 벡터 연산
a_t = torch.tensor([1., 2., 3.])
b_t = torch.tensor([4., 5., 6.])

print("a + b:", a_t + b_t)
print("a * b:", a_t * b_t)
print("a ** 2:", a_t ** 2)

# PyTorch 브로드캐스팅
column_t = torch.tensor([[1.], [2.], [3.]])
row_t = torch.tensor([10., 20., 30.])
print("브로드캐스팅 결과:\n", column_t + row_t)
print("결과 shape:", (column_t + row_t).shape)

# CUDA가 있으면 GPU, 없으면 CPU 사용
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
x = torch.randn(10000, 128, device=device)
w = torch.randn(128, device=device)
result = x @ w

print("CUDA 사용 가능:", torch.cuda.is_available())
print("선택한 장치:", device)
print("x shape:", x.shape)
print("w shape:", w.shape)
print("x @ w 결과 shape:", result.shape)
```

```text
a + b: tensor([5., 7., 9.])
a * b: tensor([ 4., 10., 18.])
a ** 2: tensor([1., 4., 9.])
브로드캐스팅 결과:
 tensor([[11., 21., 31.],
        [12., 22., 32.],
        [13., 23., 33.]])
결과 shape: torch.Size([3, 3])
CUDA 사용 가능: False
선택한 장치: cpu
x shape: torch.Size([10000, 128])
w shape: torch.Size([128])
x @ w 결과 shape: torch.Size([10000])

```

### 5.5 행렬 곱셈의 shape 규칙과 교환법칙

`(100,128) @ (128,256)`은 안쪽 차원 128이 일치해 `(100,256)`을 만듭니다. `(100,128) @ (256,128)`은 128과 256이 맞지 않아 `RuntimeError`가 발생합니다. 오른쪽 행렬을 전치하면 `(128,256)`이 되어 계산할 수 있습니다. 또한 `X@W`와 `W@X`의 결과 shape이 각각 `(50,50)`, `(100,100)`으로 달라 교환 순서가 같지 않음을 확인합니다.

```python
A = torch.randn(100, 128)
B = torch.randn(128, 256)

print("(100, 128) @ (128, 256) 결과 shape:", (A @ B).shape)

# 안쪽 차원이 맞지 않아 오류가 발생하는 예
wrong_W = torch.randn(256, 128)
A @ wrong_W
```

```text
RuntimeError: mm: mat1 and mat2 shapes cannot be multiplied (100x128 and 256x128)
```

```python
# 전치하여 안쪽 차원을 맞춘 뒤 곱하기
fixed_result = A @ wrong_W.T
print("wrong_W.T shape:", wrong_W.T.shape)
print("A @ wrong_W.T 결과 shape:", fixed_result.shape)

# 곱하는 순서에 따라 결과 shape이 달라지는지 확인
X = torch.randn(50, 100)
W = torch.randn(100, 50)

print("X @ W 결과 shape:", (X @ W).shape)
print("W @ X 결과 shape:", (W @ X).shape)
```

```text
wrong_W.T shape: torch.Size([128, 256])
A @ wrong_W.T 결과 shape: torch.Size([100, 256])
X @ W 결과 shape: torch.Size([50, 50])
W @ X 결과 shape: torch.Size([100, 100])

```

### 5.6 Softmax와 수치 안정화

Softmax는 각 점수를 지수화한 뒤 지수 합으로 나누어 확률처럼 합이 1인 비율로 바꿉니다. 큰 logits에 바로 `exp`를 적용하면 표현 범위를 넘는 overflow가 발생해 `nan`이 됩니다. 모든 logits에서 최댓값을 빼도 softmax 비율은 변하지 않으면서 지수 계산의 크기를 줄일 수 있어 안정된 결과를 얻습니다.

```python
# 일반적인 logits의 Softmax
logits = np.array([2.3, 5.1, 0.7])
softmax = np.exp(logits) / np.exp(logits).sum()
print("일반 Softmax:", softmax)
print("합계:", softmax.sum())

# 큰 값에서 overflow 발생 확인
large_logits = np.array([1000, 2000, 3000])
unstable = np.exp(large_logits) / np.exp(large_logits).sum()
print("안정화 전:", unstable)

# 최댓값을 빼서 수치 안정화
shifted = large_logits - np.max(large_logits)
stable = np.exp(shifted) / np.exp(shifted).sum()
print("최댓값을 뺀 값:", shifted)
print("안정화 후:", stable)
print("합계:", stable.sum())
```

```text
일반 Softmax: [0.05666832 0.93189054 0.01144114]
합계: 0.9999999999999999
안정화 전: [nan nan nan]
최댓값을 뺀 값: [-2000 -1000     0]
안정화 후: [0. 0. 1.]
합계: 1.0

```

```text
<ipython-input-31-84519eb99f56>:9: RuntimeWarning: overflow encountered in exp
  unstable = np.exp(large_logits) / np.exp(large_logits).sum()
<ipython-input-31-84519eb99f56>:9: RuntimeWarning: invalid value encountered in divide
  unstable = np.exp(large_logits) / np.exp(large_logits).sum()

```

### 추가 확인: PyTorch 원소별 연산과 행렬 곱 예시

아래 셀은 위 항목과 별도로 PyTorch 원소별 연산, scalar broadcasting, 간단한 `(2,3) @ (3,2)` 행렬 곱을 보충 확인합니다.

```python
a = torch.tensor([1.0, 2.0, 3.0])
b = torch.tensor([4.0, 5.0, 6.0])

print("덧셈:", a + b)
print("뺄셈:", a - b)
print("곱셈:", a * b)
print("나눗셈:", a / b)
```

```text
덧셈: tensor([5., 7., 9.])
뺄셈: tensor([-3., -3., -3.])
곱셈: tensor([ 4., 10., 18.])
나눗셈: tensor([0.2500, 0.4000, 0.5000])

```

```python
print("더하기:", a + 10)
print("빼기:", a - 1)
print("곱하기:", a * 2)
print("나누기:", a / 2)
```

```text
더하기: tensor([11., 12., 13.])
빼기: tensor([0., 1., 2.])
곱하기: tensor([2., 4., 6.])
나누기: tensor([0.5000, 1.0000, 1.5000])

```

```python
other = torch.tensor([
    [1, 2],
    [3, 4],
    [5, 6]
])

print("첫 번째 행렬 shape:", matrix.shape)
print("두 번째 행렬 shape:", other.shape)
print("행렬 곱 결과:\n", torch.matmul(matrix, other))
```

```text
첫 번째 행렬 shape: torch.Size([2, 3])
두 번째 행렬 shape: torch.Size([3, 2])
행렬 곱 결과:
 tensor([[22, 28],
        [49, 64]])

```
