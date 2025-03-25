async function run() {
    const params = new URLSearchParams(window.location.search);
    const target = params.get('target');
    if (!target) {
      document.getElementById("status").textContent = "유효하지 않은 URL입니다.";
      return;
    }
  
    document.getElementById("status").textContent = `분석 중: ${target}`;
  
    const session = await ort.InferenceSession.create('C:\\Users\\sclab\\Documents\\Downloads\\phishingURL\\notebook\\neuro_fuzzy_model.onnx');
  
    // 🎯 입력 전처리: URL을 벡터화 (예: 길이, 문자 빈도 등)
    const inputVector = urlToFeature(target);
    const inputTensor = new ort.Tensor('float32', inputVector, [1, inputVector.length]);
  
    const feeds = { input: inputTensor };  // input 이름은 ONNX export 시 지정한 이름
    const results = await session.run(feeds);
  
    const score = results.output.data[0];  // 첫 번째 출력 값 (위험도 점수)
  
    if (score > 0.8) {
      document.getElementById("status").textContent = "⚠️ 위험한 URL로 판단되어 차단되었습니다.";
    } else {
      document.getElementById("status").textContent = "✅ 안전합니다. 이동 중...";
      setTimeout(() => window.location.href = target, 1000);
    }
  }
  
  function urlToFeature(url) {
    // 예: 길이, 숫자 개수, 특수문자 비율 등 간단한 피처 추출
    const length = url.length;
    const hasHttps = url.startsWith("https") ? 1 : 0;
    const dotCount = (url.match(/\./g) || []).length;
    const digitCount = (url.match(/\d/g) || []).length;
    const specialCharCount = (url.match(/[^a-zA-Z0-9]/g) || []).length;
  
    return new Float32Array([length / 100.0, hasHttps, dotCount / 10.0, digitCount / 10.0, specialCharCount / 10.0]);
  }
  
  run();
  