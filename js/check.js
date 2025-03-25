async function run() {
  const params = new URLSearchParams(window.location.search);
  const target = params.get('target');
  if (!target) {
    document.getElementById("status").textContent = "유효하지 않은 URL입니다.";
    return;
  }

  document.getElementById("status").textContent = `분석 중: ${target}`;

  const session = await ort.InferenceSession.create('model/model.onnx');

  const x_fuzzy = extractFuzzyFeatures(target); // Float32Array [1, 15]
  const x_char = tokenizeChar(target);         // Int32Array [1, 100]
  const x_word = tokenizeWord(target);         // Int32Array [1, 30]

  const feeds = {
    x_fuzzy: new ort.Tensor('float32', x_fuzzy, [1, 15]),
    x_char: new ort.Tensor('int32', x_char, [1, 100]),
    x_word: new ort.Tensor('int32', x_word, [1, 30])
  };

  const results = await session.run(feeds);
  const score = results.neural_output.data[0];

  if (score > 0.5) {
    document.getElementById("status").textContent = "⚠️ 피싱 URL로 판단되어 차단되었습니다.";
  } else {
    document.getElementById("status").textContent = "✅ 안전한 URL입니다. 이동 중...";
    setTimeout(() => window.location.href = target, 1000);
  }
}

function extractFuzzyFeatures(url) {
  const url_length = url.length;
  const special_chars = (url.match(/[^\w\s]/g) || []).length;
  const digits = (url.match(/\d/g) || []).length;
  const hyphens = (url.match(/-/g) || []).length;
  const subdomains = (url.match(/\./g) || []).length;
  const entropy = computeEntropy(url);
  const has_ip = /\d{1,3}(\.\d{1,3}){3}/.test(url) ? 1 : 0;
  const has_at = url.includes('@') ? 1 : 0;
  const has_https = url.startsWith('https') ? 1 : 0;
  const has_www = url.includes('www.') ? 1 : 0;
  const has_com = url.includes('.com') ? 1 : 0;
  const domain_hyphen = /:\/\/(www\.)?([^\/]+)-/.test(url) ? 1 : 0;

  const typosquatting_keywords = [
    'goole', 'gogle', 'googl', 'gooogle', 'gmai', 'gamil', 'gmaill',
    'facebok', 'faceboook', 'facebokk', 'facbook', 'faccebook',
    'facebooo', 'facebokoo', 'lnstagram', 'instagraam', 'instaagram',
    'instagrma', 'instargram', 'lnstagraam', 'outlookk', 'outlok', 'yahoomail'
  ];
  const typo = typosquatting_keywords.some(kw => url.toLowerCase().includes(kw)) ? 1 : 0;

  const placeholder1 = 0;
  const placeholder2 = 0;

  return new Float32Array([
    url_length, special_chars, digits, hyphens, subdomains,
    entropy, has_ip, has_at, has_https, has_www, has_com,
    domain_hyphen, typo, placeholder1, placeholder2
  ]);
}

function computeEntropy(str) {
  const map = {};
  for (const char of str) map[char] = (map[char] || 0) + 1;
  const len = str.length;
  return -Object.values(map).reduce((acc, val) => {
    const p = val / len;
    return acc + p * Math.log2(p);
  }, 0);
}

function tokenizeChar(text, maxLen = 100, vocabSize = 128) {
  const seq = [];
  for (let i = 0; i < text.length && seq.length < maxLen; i++) {
    const code = text.charCodeAt(i);
    seq.push(code >= 32 && code < 32 + vocabSize ? code - 31 : 1);
  }
  while (seq.length < maxLen) seq.push(0);
  return Int32Array.from(seq);
}

function tokenizeWord(text, maxLen = 30, vocabSize = 10000) {
  const symbolRegex = /[\/\?&=\.:]/;
  const wordMap = tokenizeWord.wordMap || (tokenizeWord.wordMap = {});
  let index = Object.keys(wordMap).length + 1;
  const tokens = text.split(/\W+/).filter(t => !symbolRegex.test(t));
  const seq = [];
  for (const word of tokens) {
    if (!(word in wordMap)) {
      if (index >= vocabSize) seq.push(0);
      else wordMap[word] = index++;
    }
    seq.push(wordMap[word]);
  }
  while (seq.length < maxLen) seq.push(0);
  return Int32Array.from(seq.slice(0, maxLen));
}

run();