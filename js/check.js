// js/check.js

// 페이지 로드가 완료된 후 run 실행
window.addEventListener("DOMContentLoaded", run);

let domainClasses = [];
let subdomainClasses = [];

async function loadLabelEncoders() {
  const domainRes = await fetch("domain_classes_all.json");
  const subdomainRes = await fetch("subdomain_classes_all.json");
  domainClasses = await domainRes.json();
  subdomainClasses = await subdomainRes.json();
}

function getCategoryIndex(label, classesArray) {
  if (!Array.isArray(classesArray)) {
    console.error("classesArray는 배열이어야 합니다.");
    return 0;
  }
  let index = classesArray.indexOf(label);
  if (index === -1) {
    classesArray.push(label);
    index = classesArray.length - 1;
  }
  return index;
}

async function run() {
  const statusElem = document.getElementById("status");
  if (!statusElem) {
    console.error("❌ 'status' 요소를 찾을 수 없습니다.");
    return;
  }

  const params = new URLSearchParams(window.location.search);
  const target = params.get('target');
  if (!target) {
    statusElem.textContent = "유효하지 않은 URL입니다.";
    return;
  }

  await loadLabelEncoders();
  statusElem.textContent = `분석 중: ${target}`;

  const session = await ort.InferenceSession.create('neuro_fuzzy_model_all.onnx', {
    executionProviders: ['wasm']
  });

  const x_fuzzy = createFeatureVector(target); // Float32Array [1, 15]
  const x_char = tokenizeChar(target);         // Int32Array [1, 100]
  const x_word = tokenizeWord(target);         // Int32Array [1, 30]
  console.log("target: ", target)
  console.log("x_fuzzy: ", x_fuzzy)
  console.log("x_char: ", x_char)
  console.log("x_word: ", x_word)
  const feeds = {
    x_fuzzy: new ort.Tensor('float32', x_fuzzy, [1, 15]),
    x_char: new ort.Tensor('int64', BigInt64Array.from(Array.from(x_char, v => BigInt(v))), [1, 100]),
    x_word: new ort.Tensor('int64', BigInt64Array.from(Array.from(x_word, v => BigInt(v))), [1, 30])
  };

  const results = await session.run(feeds);
  const score = results.output.data[0];
  console.log("✅ 모델 예측 score:", score);

  if (score > 0.5) {
    statusElem.textContent = "⚠️ 피싱 URL로 판단되어 차단되었습니다.";
  } else {
    statusElem.textContent = "✅ 안전한 URL입니다. 이동 중...";
    let finalUrl = target;
    if (!/^https?:\/\//i.test(target)) {
      finalUrl = "https://" + target;
    }
    setTimeout(() => window.location.href = finalUrl, 1000);
  }
}

function extractTyposquatting(url) {
  const lower = url.toLowerCase();
  for (let i = 0; i < lower.length - 1; i++) {
    const pair = lower[i] + lower[i + 1];
    if ((pair === 'oo' || pair === '00') ||
        (pair === 'll' || pair === '11') ||
        (pair === 'il' || pair === 'li')) {
      return true;
    }
  }
  const typosquatting_keywords = [
    'goole', 'gogle', 'googl', 'gooogle', 'gmai', 'gamil', 'gmaill',
    'facebok', 'faceboook', 'facebokk', 'facbook', 'faccebook',
    'facebooo', 'facebokoo', 'lnstagram', 'instagraam', 'instaagram',
    'instagrma', 'instargram', 'lnstagraam', 'outlookk', 'outlok', 'yahoomail'
  ];
  return typosquatting_keywords.some(keyword => lower.includes(keyword));
}

function extractDomainAndSubdomain(url) {
  try {
    const psl = window.psl;
    const a = document.createElement("a");
    a.href = url.startsWith("http") ? url : "https://" + url;
    const hostname = a.hostname.replace(/[\u3002\uff0e\uff61]/g, '.');
    const parsed = psl.parse(hostname);

    if (!parsed.domain || !parsed.sld || !parsed.tld) {
      return { domain: hostname, subdomain: "None" };
    }

    const suffix = parsed.tld;
    const domain = parsed.sld;
    const fullDomain = domain;

    const labels = hostname.split(".");
    const domainIndex = labels.lastIndexOf(domain);
    const subdomainParts = labels.slice(0, domainIndex);
    return {
      domain: fullDomain,
      subdomain: subdomainParts.length > 0 ? subdomainParts.join(".") : "None"
    };
  } catch (e) {
    return { domain: "", subdomain: "None" };
  }
}

function createFeatureVector(url) {
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
  
  const { domain, subdomain } = extractDomainAndSubdomain(url);
  const domain_hyphen = domain.includes('-') ? 1 : 0;
  const typosquatting = extractTyposquatting(url) ? 1 : 0;
  
  console.log("domain: ", domain)
  console.log("subdomain: ", subdomain)
  
  const domainIndex = getCategoryIndex(domain, domainClasses);
  const subdomainIndex = getCategoryIndex(subdomain, subdomainClasses);
  
  return new Float32Array([
    url_length, special_chars, digits, hyphens, subdomains,
    entropy, has_ip, has_at, has_https, has_www, has_com,
    domain_hyphen, typosquatting, domainIndex, subdomainIndex
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

function wordPunctTokenizer(text) {
  return text.match(/[\w]+|[^\s\w]/g) || [];
}

function tokenizeWord(text, maxLen = 30, vocabSize = 10000) {
  const symbolRegex = /[\/\?&=\.:]/;
  const wordMap = tokenizeWord.wordMap || (tokenizeWord.wordMap = {});
  let index = Object.keys(wordMap).length + 1;
  const tokens = wordPunctTokenizer(text).filter(t => !symbolRegex.test(t));

  const seq = [];
  for (const word of tokens) {
    if (!(word in wordMap)) {
      if (index >= vocabSize) {
        wordMap[word] = 0;
      } else {
        wordMap[word] = index++;
      }
    }
    seq.push(wordMap[word]);
  }
  while (seq.length < maxLen) seq.push(0);
  return Int32Array.from(seq.slice(0, maxLen));
}
