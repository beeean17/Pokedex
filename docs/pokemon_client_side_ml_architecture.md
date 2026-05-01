# Homework #6 포켓몬 분류 서비스 아키텍처 전략 보고서

## Client-side ML Serving 기반 포켓몬 분류 시스템

본 문서는 Transfer Learning으로 학습된 포켓몬 분류 모델을 백엔드 서버 없이 웹 브라우저에서 직접 실행하기 위한 Client-side Machine Learning 서빙 아키텍처 전략을 다룬다. 핵심 목표는 서버 비용을 전면 제거하고, 사용자의 로컬 컴퓨팅 자원을 활용하여 이미지 분류 추론을 수행하는 것이다.

본 과제에서는 모바일 환경까지 무리하게 지원하지 않고, 데스크톱 브라우저 환경을 우선 대상으로 한다. 모바일 기기는 메모리, 발열, 브라우저 호환성, WebGPU 지원 여부 등의 변수가 크므로 초기 버전에서는 접근을 제한한다.

---

## 1. 시스템 아키텍처 개요

기존의 서버 중심 ML 서비스는 사용자가 이미지를 서버로 업로드하고, 서버에서 모델 추론을 수행한 뒤 결과를 반환하는 구조를 가진다. 그러나 본 프로젝트에서는 모델 파일을 정적 자산으로 취급하여 클라이언트에 전달하고, 브라우저 내부에서 직접 추론을 수행하는 구조를 채택한다.

즉, 서버는 모델 추론을 수행하지 않으며, 단순히 정적 파일을 제공하는 역할만 한다.

| 구성 요소 | 역할 | 비고 |
|---|---|---|
| Frontend | React 기반 UI/UX 구현 | 이미지 업로드, 미리보기, 결과 시각화 |
| Inference Engine | ONNX Runtime Web | 브라우저 내 WASM/WebGPU 기반 추론 |
| Model Format | ONNX | PyTorch/TensorFlow 모델을 웹 추론용으로 변환 |
| Model Storage | Static Asset | `/public/models` 디렉터리에 모델 파일 저장 |
| Hosting | GitHub Pages | 정적 웹 페이지 무료 배포 |
| Cache Strategy | Browser Cache / Service Worker | 모델 재다운로드 방지 |
| Mobile Policy | 모바일 접근 제한 | 데스크톱 브라우저 전용 서비스 |

---

## 2. 핵심 아키텍처 방향

본 서비스의 핵심 방향은 다음과 같다.

```text
사용자 이미지 업로드
→ 브라우저에서 이미지 전처리
→ ONNX Runtime Web으로 모델 로드
→ 클라이언트 로컬 환경에서 추론
→ 결과를 화면에 표시
```

서버는 다음 역할을 하지 않는다.

```text
이미지 저장 X
이미지 업로드 처리 X
모델 추론 X
사용자 데이터 처리 X
백엔드 API 서버 운영 X
```

따라서 본 구조는 서버 비용이 거의 발생하지 않으며, 사용자의 이미지가 외부 서버로 전송되지 않는다는 장점이 있다.

---

## 3. 핵심 기술 스택

## 3.1 Model Training & Export

모델 학습은 Python 환경에서 수행한다.

| 항목 | 기술 |
|---|---|
| Training Framework | PyTorch 또는 TensorFlow |
| 학습 방식 | Transfer Learning |
| 권장 Backbone | MobileNetV3, EfficientNet-Lite |
| Export Format | ONNX |
| Optimization | Int8 Quantization |

포켓몬 이미지 분류 과제에서는 처음부터 대형 CNN 모델을 학습하는 것보다, ImageNet 등으로 사전 학습된 모델을 기반으로 Transfer Learning을 적용하는 것이 효율적이다.

권장 모델은 다음과 같다.

| 모델 | 특징 | 적합성 |
|---|---|---|
| MobileNetV3-Small | 가볍고 빠름 | 웹 추론에 적합 |
| MobileNetV3-Large | 정확도와 속도 균형 | 권장 |
| EfficientNet-Lite | 모바일/엣지 추론 최적화 | 정확도 우선 시 적합 |

---

## 3.2 Frontend Service

프론트엔드는 React 기반으로 구현하는 것을 권장한다.

| 항목 | 기술 |
|---|---|
| UI Framework | React |
| Build Tool | Vite |
| ML Runtime | onnxruntime-web |
| Styling | CSS / Tailwind CSS |
| Deployment | GitHub Pages |

React를 사용하는 이유는 다음과 같다.

```text
브라우저 ML 라이브러리와 연동이 쉽다.
이미지 업로드 및 미리보기 구현이 간단하다.
Vite 기반 정적 배포가 쉽다.
GitHub Pages와 호환성이 좋다.
```

Flutter Web도 가능하지만, ONNX Runtime Web과의 연동 및 브라우저 ML 생태계 측면에서는 React가 더 단순하다.

---

## 4. 모델 변환 및 최적화 전략

## 4.1 ONNX 변환

학습된 PyTorch 또는 TensorFlow 모델은 브라우저에서 직접 실행하기 어렵다. 따라서 모델을 ONNX 형식으로 변환해야 한다.

```text
PyTorch / TensorFlow 모델
→ ONNX 변환
→ ONNX Runtime Web에서 로드
→ 브라우저 추론 수행
```

ONNX는 다양한 프레임워크에서 학습된 모델을 공통된 추론 포맷으로 사용할 수 있게 해주는 표준 포맷이다.

---

## 4.2 Int8 양자화

브라우저에서 모델을 실행하려면 모델 크기와 메모리 사용량을 줄이는 것이 중요하다. 이를 위해 FP32 모델을 Int8로 양자화한다.

```text
FP32 모델
→ Int8 Quantization
→ 모델 크기 감소
→ 로딩 속도 개선
→ 추론 속도 개선 가능
```

양자화의 기대 효과는 다음과 같다.

| 항목 | 효과 |
|---|---|
| 모델 크기 | 약 50~80% 감소 가능 |
| 초기 로딩 시간 | 감소 |
| 브라우저 메모리 사용량 | 감소 |
| 추론 속도 | 환경에 따라 개선 가능 |

과제 제출용 프로젝트에서는 모델 파일 크기를 가능하면 100MB 이하, 권장 10~50MB 수준으로 유지하는 것이 좋다.

---

## 4.3 권장 모델 크기

| 구분 | 권장 크기 | 설명 |
|---|---:|---|
| 이상적 | 10~20MB | 빠른 로딩, 캐싱 용이 |
| 허용 가능 | 20~50MB | 과제용 웹 서비스로 충분히 가능 |
| 주의 필요 | 50~100MB | 초기 로딩이 길어질 수 있음 |
| 비권장 | 100MB 초과 | GitHub 제한 및 사용자 이탈 가능성 증가 |

---

## 5. 클라이언트 추론 구현 전략

## 5.1 추론 흐름

브라우저 추론은 다음 흐름으로 구현한다.

```text
1. 사용자가 포켓몬 이미지 업로드
2. 브라우저에서 이미지 미리보기 표시
3. 이미지를 모델 입력 크기로 리사이즈
4. RGB 값 정규화
5. Tensor 형태로 변환
6. ONNX Runtime Web에 입력
7. 추론 결과 획득
8. 가장 높은 확률의 클래스를 화면에 표시
```

---

## 5.2 이미지 전처리

사용자가 업로드하는 이미지는 크기와 비율이 다양하다. 따라서 모델 입력에 맞게 전처리해야 한다.

일반적인 입력 크기는 다음과 같다.

| 모델 | 입력 크기 |
|---|---|
| MobileNetV3 | 224x224 |
| EfficientNet-Lite | 224x224 또는 240x240 |

전처리 과정은 다음과 같다.

```text
원본 이미지
→ Canvas에 로드
→ 224x224로 리사이즈
→ RGB 값 추출
→ 0~1 범위로 정규화
→ Tensor 변환
```

이 과정을 브라우저에서 수행하면 이미지를 서버로 전송할 필요가 없다.

---

## 5.3 ONNX Runtime Web 실행 방식

ONNX Runtime Web은 브라우저 환경에서 모델 추론을 수행할 수 있게 해준다.

주요 실행 방식은 다음과 같다.

| 실행 방식 | 설명 |
|---|---|
| WASM | CPU 기반 브라우저 추론 |
| WASM SIMD | SIMD 명령어를 활용한 CPU 가속 |
| WebGPU | GPU 기반 브라우저 추론 |
| WebGL | 구형 GPU 가속 fallback |

초기 구현에서는 호환성이 높은 WASM을 기본으로 두고, 지원되는 환경에서는 WebGPU를 사용할 수 있도록 구성한다.

```javascript
const session = await ort.InferenceSession.create(
  "/models/pokemon-mobilenetv3-int8-v1.onnx",
  {
    executionProviders: ["webgpu", "wasm"]
  }
);
```

WebGPU가 동작하지 않는 브라우저에서는 WASM으로 fallback한다.

---

## 6. 모델 캐싱 및 재다운로드 방지 전략

Client-side inference 구조에서는 모델 파일이 정적 자산으로 제공된다. 따라서 모델 파일을 매번 다운로드하지 않도록 브라우저 캐싱 전략을 적용해야 한다.

---

## 6.1 파일명 기반 버전 관리

모델 파일은 버전이 포함된 이름으로 배포한다.

```text
/models/pokemon-mobilenetv3-int8-v1.onnx
/models/pokemon-mobilenetv3-int8-v2.onnx
```

같은 파일명을 유지하면 브라우저가 캐시된 모델을 재사용할 수 있다. 모델이 변경될 경우에는 파일명에 포함된 버전을 증가시켜 새 파일로 인식하게 한다.

```text
pokemon-model-v1.onnx  → 기존 캐시 사용
pokemon-model-v2.onnx  → 새 모델로 인식하여 다운로드
```

이 방식은 GitHub Pages처럼 HTTP Header를 세밀하게 제어하기 어려운 환경에서도 적용하기 쉽다.

---

## 6.2 Browser Cache 활용

모델 파일은 일반 정적 파일이므로 브라우저 캐시 대상이 된다.

이 전략의 효과는 다음과 같다.

| 상황 | 동작 |
|---|---|
| 최초 접속 | 모델 파일 다운로드 |
| 두 번째 접속 | 브라우저 캐시에서 모델 재사용 |
| 모델 버전 동일 | 재다운로드 최소화 |
| 모델 버전 변경 | 새 모델 파일 다운로드 |

즉, 사용자는 첫 접속 시에만 모델 다운로드 시간을 경험하고, 이후에는 캐시된 모델을 사용하여 더 빠르게 서비스를 이용할 수 있다.

---

## 6.3 Service Worker 기반 캐싱

PWA 또는 Service Worker를 적용하면 모델 파일을 명시적으로 캐싱할 수 있다.

```javascript
const MODEL_CACHE = "pokemon-model-cache-v1";
const MODEL_URL = "/models/pokemon-mobilenetv3-int8-v1.onnx";

self.addEventListener("install", event => {
  event.waitUntil(
    caches.open(MODEL_CACHE).then(cache => {
      return cache.add(MODEL_URL);
    })
  );
});
```

Service Worker 캐싱의 장점은 다음과 같다.

| 항목 | 효과 |
|---|---|
| 재방문 속도 | 향상 |
| 네트워크 사용량 | 감소 |
| 모델 재다운로드 | 최소화 |
| 시연 안정성 | 향상 |

과제 제출용으로는 기본 브라우저 캐시만으로도 충분하지만, 완성도를 높이려면 Service Worker 캐싱을 추가할 수 있다.

---

## 6.4 IndexedDB 캐싱

모델 파일을 `ArrayBuffer` 형태로 직접 받아 IndexedDB에 저장하는 방식도 가능하다.

```javascript
const response = await fetch("/models/pokemon-mobilenetv3-int8-v1.onnx");
const modelBuffer = await response.arrayBuffer();
```

이후 `modelBuffer`를 IndexedDB에 저장하면 모델을 직접 관리할 수 있다.

다만 구현 복잡도가 증가하므로 본 과제에서는 다음 우선순위를 권장한다.

```text
1순위: 파일명 기반 버전 관리 + 브라우저 캐시
2순위: Service Worker 캐싱
3순위: IndexedDB 직접 저장
```

---

## 7. 모바일 접근 제한 전략

본 프로젝트의 초기 버전에서는 모바일 환경을 지원하지 않는다. 모바일 브라우저는 다음 문제가 발생할 수 있다.

```text
메모리 부족
브라우저별 WebGPU 지원 차이
WASM 성능 저하
발열 및 배터리 소모
이미지 전처리 지연
화면 크기에 따른 UI 문제
```

따라서 모바일 환경에서는 서비스를 실행하지 않고, 데스크톱 브라우저 사용을 안내한다.

---

## 7.1 모바일 제한 이유

| 제한 요소 | 설명 |
|---|---|
| 메모리 | 모델 로드와 이미지 전처리에 많은 메모리 사용 가능 |
| 성능 | 모바일 CPU/WASM 성능이 데스크톱보다 낮음 |
| 발열 | 반복 추론 시 발열 및 배터리 소모 발생 |
| 호환성 | WebGPU, WebGL, WASM SIMD 지원 여부가 기기별로 다름 |
| UX | 작은 화면에서 이미지 업로드 및 결과 표시가 불편할 수 있음 |

모바일까지 지원하려면 별도의 모바일 전용 모델, UI, fallback 정책이 필요하다. 하지만 Homework #6의 범위에서는 데스크톱 웹 환경에 집중하는 것이 더 안정적이다.

---

## 7.2 모바일 감지 방식

프론트엔드에서 User-Agent 또는 화면 크기를 기준으로 모바일 접근을 제한할 수 있다.

```javascript
function isMobileDevice() {
  return /Mobi|Android|iPhone|iPad|iPod/i.test(navigator.userAgent);
}

if (isMobileDevice()) {
  // 모바일 차단 화면 표시
}
```

또는 화면 너비를 기준으로 제한할 수도 있다.

```javascript
function isSmallScreen() {
  return window.innerWidth < 768;
}
```

실제 구현에서는 두 조건을 함께 사용하는 것이 안전하다.

```javascript
const blocked = isMobileDevice() || isSmallScreen();
```

---

## 7.3 모바일 사용자 안내 문구

모바일 사용자가 접속한 경우 다음과 같은 안내 화면을 제공한다.

```text
이 서비스는 브라우저에서 직접 AI 모델을 실행하는 구조입니다.
모바일 기기에서는 메모리와 성능 문제로 인해 현재 지원하지 않습니다.
데스크톱 Chrome 또는 Edge 브라우저에서 접속해 주세요.
```

이 방식은 모바일에서 느린 추론이 발생하거나 오류가 발생하는 것보다 안정적인 사용자 경험을 제공한다.

---

## 8. GitHub Pages 및 모델 파일 배포 전략

## 8.1 GitHub Pages 배포

본 프로젝트는 백엔드 서버가 필요하지 않으므로 GitHub Pages를 통해 무료로 배포할 수 있다.

```text
React App Build
→ 정적 파일 생성
→ GitHub Pages에 배포
→ 브라우저에서 직접 실행
```

Vite 기반 React 프로젝트라면 다음 명령어로 빌드할 수 있다.

```bash
npm run build
```

배포에는 `gh-pages` 패키지를 사용할 수 있다.

```bash
npm install gh-pages --save-dev
```

---

## 8.2 GitHub LFS 사용 주의

GitHub는 일반 repository 파일 크기에 제한이 있다. 대용량 모델 파일은 GitHub LFS를 사용할 수 있지만, LFS에는 대역폭 제한이 존재한다.

따라서 모델 파일을 GitHub LFS에 의존하는 구조는 주의해야 한다.

권장 전략은 다음과 같다.

```text
1. 모델을 100MB 이하로 경량화한다.
2. 가능하면 /public/models 폴더에 정적 파일로 포함한다.
3. 파일명 기반 버전 관리를 적용한다.
4. 브라우저 캐시를 통해 재다운로드를 줄인다.
5. LFS는 최후 수단으로 사용한다.
```

---

## 8.3 권장 모델 배포 방식

과제용 프로젝트에서는 다음 방식이 가장 현실적이다.

```text
pokemon-mobilenetv3-int8-v1.onnx
→ 100MB 이하로 최적화
→ public/models/에 저장
→ GitHub Pages에서 정적 파일로 제공
→ 브라우저 캐시로 재사용
```

모델 크기가 100MB를 초과한다면 다음 대안을 고려한다.

| 대안 | 설명 |
|---|---|
| 모델 추가 양자화 | 모델 크기 축소 |
| 더 작은 Backbone 사용 | MobileNetV3-Small 등 사용 |
| GitHub Release Asset | 모델 파일을 Release에 첨부 |
| Hugging Face Hub | 모델 저장소로 활용 가능 |
| Cloudflare R2 | 정적 모델 파일 저장 가능 |

그러나 본 과제에서는 서버 비용 제거와 단순 배포가 목표이므로, 모델을 충분히 경량화하여 GitHub Pages 정적 자산으로 포함하는 전략이 가장 적합하다.

---

## 9. 프로젝트 디렉터리 구조

권장 디렉터리 구조는 다음과 같다.

```text
pokemon-classifier/
├── public/
│   ├── models/
│   │   ├── pokemon-mobilenetv3-int8-v1.onnx
│   │   └── labels.json
│   └── sample/
│       └── example-pokemon.png
├── src/
│   ├── components/
│   │   ├── ImageUploader.jsx
│   │   ├── PredictionResult.jsx
│   │   ├── LoadingModel.jsx
│   │   └── MobileBlocked.jsx
│   ├── ml/
│   │   ├── loadModel.js
│   │   ├── preprocessImage.js
│   │   └── predict.js
│   ├── utils/
│   │   └── deviceCheck.js
│   ├── App.jsx
│   └── main.jsx
├── package.json
├── vite.config.js
└── README.md
```

---

## 10. 주요 컴포넌트 설계

## 10.1 ImageUploader

사용자가 포켓몬 이미지를 업로드하는 컴포넌트이다.

역할은 다음과 같다.

```text
이미지 파일 선택
이미지 미리보기 표시
지원하지 않는 파일 형식 차단
선택된 이미지를 추론 함수로 전달
```

---

## 10.2 LoadingModel

모델 로딩 상태를 표시하는 컴포넌트이다.

Client-side inference에서는 모델 파일 다운로드와 초기화 시간이 발생하므로 로딩 UI가 필수이다.

표시할 정보는 다음과 같다.

```text
모델 다운로드 중
모델 초기화 중
캐시된 모델 로드 중
추론 준비 완료
```

---

## 10.3 PredictionResult

추론 결과를 사용자에게 보여주는 컴포넌트이다.

표시할 정보는 다음과 같다.

```text
예측된 포켓몬 이름
분류 확률
상위 N개 후보
처리 시간
```

예시 출력은 다음과 같다.

```text
예측 결과: Pikachu
신뢰도: 94.2%
추론 시간: 128ms
```

---

## 10.4 MobileBlocked

모바일 사용자가 접속했을 때 표시되는 차단 화면이다.

역할은 다음과 같다.

```text
모바일 접근 감지
서비스 제한 안내
데스크톱 브라우저 사용 권장
```

---

## 11. 전략적 기대 효과

## 11.1 서버 비용 제거

모델 추론을 서버에서 수행하지 않으므로 EC2, GCP, Azure, Firebase Functions와 같은 서버 비용이 발생하지 않는다.

```text
서버 추론 비용 없음
GPU 서버 비용 없음
API 호출 비용 없음
백엔드 운영 비용 없음
```

---

## 11.2 확장성 확보

추론 부하가 서버에 집중되지 않고 각 사용자의 브라우저에서 처리된다.

따라서 동시 접속자가 증가해도 서버 추론 병목이 발생하지 않는다.

```text
사용자 1명 → 사용자 브라우저에서 추론
사용자 100명 → 각자 브라우저에서 추론
사용자 1000명 → 서버 추론 부하 없음
```

---

## 11.3 데이터 프라이버시 향상

사용자의 이미지는 서버로 업로드되지 않는다. 브라우저 내부에서만 전처리와 추론이 수행된다.

```text
이미지 서버 전송 없음
이미지 저장 없음
개인 데이터 처리 최소화
로컬 추론 기반 보안성 확보
```

---

## 11.4 포트폴리오 가치

단순한 API 호출 기반 프로젝트가 아니라, 웹 브라우저에서 직접 ML 모델을 실행하는 구조이므로 기술적 차별성이 있다.

강조 가능한 기술 포인트는 다음과 같다.

```text
Transfer Learning
ONNX 모델 변환
Int8 Quantization
ONNX Runtime Web
Client-side Inference
Browser Cache
Service Worker Caching
GitHub Pages Deployment
Mobile Device Blocking
```

---

## 12. 주의 및 제약 사항

## 12.1 초기 로딩 시간

모델 파일은 최초 접속 시 다운로드되어야 하므로 초기 로딩 시간이 발생한다.

해결 전략은 다음과 같다.

```text
모델 경량화
Int8 양자화
로딩 UI 제공
브라우저 캐시 활용
Service Worker 캐싱 적용
```

---

## 12.2 브라우저 호환성

브라우저별로 WebGPU, WebGL, WASM SIMD 지원 여부가 다르다.

따라서 다음 우선순위로 실행한다.

```text
1순위: WebGPU
2순위: WASM SIMD
3순위: WASM
```

단, 초기 구현에서는 안정성을 위해 WASM 기반 실행을 우선으로 두는 것도 가능하다.

---

## 12.3 모바일 미지원

모바일 환경은 초기 버전에서 지원하지 않는다.

모바일 미지원은 단점이지만, 과제 범위에서는 안정성과 구현 완성도를 높이는 전략으로 볼 수 있다.

```text
모바일 지원 X
데스크톱 Chrome / Edge 권장
모바일 접속 시 안내 화면 표시
```

---

## 12.4 모델 파일 크기 제한

모델 파일이 너무 크면 GitHub Pages 배포와 사용자 로딩 속도 측면에서 문제가 발생한다.

따라서 다음 기준을 지킨다.

```text
권장: 10~50MB
허용: 100MB 이하
비권장: 100MB 초과
```

---

## 13. 최종 아키텍처 요약

최종 구조는 다음과 같다.

```text
[User Desktop Browser]
        |
        | 이미지 업로드
        v
[React Frontend]
        |
        | 이미지 전처리
        v
[ONNX Runtime Web]
        |
        | WASM / WebGPU 추론
        v
[Pokemon Classification Result]
```

모델 파일 제공 구조는 다음과 같다.

```text
[GitHub Pages]
        |
        | 정적 모델 파일 제공
        v
[Browser Cache / Service Worker]
        |
        | 최초 1회 다운로드 후 재사용
        v
[ONNX Runtime Web]
```

모바일 접근은 다음과 같이 처리한다.

```text
[Mobile Browser]
        |
        | User-Agent / 화면 크기 감지
        v
[MobileBlocked Component]
        |
        | 데스크톱 사용 안내
        v
[Inference 실행 안 함]
```

---

## 14. 결론

본 프로젝트는 서버 중심 ML 서비스가 아니라 Client-side ML Serving 구조를 채택한다. 학습된 포켓몬 분류 모델은 ONNX 형식으로 변환하고, Int8 양자화를 통해 경량화한 뒤, GitHub Pages에서 정적 자산으로 배포한다.

브라우저에서는 ONNX Runtime Web을 사용하여 사용자의 로컬 CPU 또는 GPU 자원으로 직접 추론을 수행한다. 모델 파일은 브라우저 캐시와 Service Worker를 활용하여 최초 1회 다운로드 이후 재사용되도록 설계한다.

모바일 환경은 메모리, 성능, 호환성 문제가 크기 때문에 초기 버전에서는 지원하지 않고, 데스크톱 브라우저 전용 서비스로 제한한다.

이 전략을 통해 다음 목표를 달성할 수 있다.

```text
서버 비용 제거
정적 배포만으로 서비스 운영
사용자 이미지 프라이버시 보장
모델 재다운로드 최소화
데스크톱 환경에서 안정적인 ML 추론 제공
포트폴리오 가치가 높은 웹 기반 AI 서비스 구현
```

따라서 Homework #6 포켓몬 분류 과제에서는 Client-side Inference 기반 아키텍처가 비용, 구현 난이도, 기술적 완성도 측면에서 가장 적합한 전략이다.
