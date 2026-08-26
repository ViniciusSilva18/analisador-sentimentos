"""Análise de sentimentos de avaliações via Gemini LLM."""
import os
from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
from dotenv import load_dotenv

load_dotenv()
app = Flask(__name__)
CORS(app)

HTML = '''<!DOCTYPE html>
<html lang="pt-PT">
<head><meta charset="UTF-8"/><meta name="viewport" content="width=device-width,initial-scale=1.0"/>
<title>SentimentVibe — Análise de Sentimentos</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet"/>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:Inter,sans-serif;background:#090d16;color:#f1f5f9;min-height:100vh;display:flex;flex-direction:column;align-items:center;justify-content:center;padding:24px}
.card{background:#0f172a;border:1px solid rgba(255,255,255,.08);border-radius:20px;padding:36px;width:100%;max-width:600px}
h1{font-size:24px;font-weight:800;background:linear-gradient(135deg,#06b6d4,#8b5cf6);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;margin-bottom:8px}
p.sub{font-size:14px;color:#64748b;margin-bottom:24px}
textarea{width:100%;background:rgba(255,255,255,.05);border:1px solid rgba(255,255,255,.08);border-radius:12px;padding:14px;color:#f1f5f9;font-size:14px;font-family:inherit;resize:vertical;min-height:120px;outline:none;transition:border-color .2s;margin-bottom:14px}
textarea:focus{border-color:#06b6d4}
button{width:100%;padding:14px;background:linear-gradient(135deg,#06b6d4,#8b5cf6);border:none;border-radius:12px;color:white;font-size:15px;font-weight:700;cursor:pointer;transition:transform .2s}
button:hover{transform:translateY(-2px)}
button:disabled{opacity:.5;cursor:not-allowed}
.result{margin-top:24px;padding:20px;background:rgba(255,255,255,.04);border:1px solid rgba(255,255,255,.08);border-radius:14px;display:none}
.sentiment-badge{display:inline-block;padding:6px 16px;border-radius:99px;font-weight:700;font-size:14px;margin-bottom:12px}
.positive{background:rgba(16,185,129,.15);color:#10b981;border:1px solid rgba(16,185,129,.3)}
.negative{background:rgba(244,63,94,.15);color:#f43f5e;border:1px solid rgba(244,63,94,.3)}
.neutral{background:rgba(100,116,139,.15);color:#94a3b8;border:1px solid rgba(100,116,139,.3)}
.mixed{background:rgba(245,158,11,.15);color:#f59e0b;border:1px solid rgba(245,158,11,.3)}
.score-bar{height:8px;background:rgba(255,255,255,.08);border-radius:99px;overflow:hidden;margin:8px 0}
.score-fill{height:100%;border-radius:99px;transition:width 1s ease}
.analysis-text{font-size:14px;color:#94a3b8;line-height:1.7;margin-top:12px}
.examples{display:flex;flex-direction:column;gap:8px;margin-top:16px}
.ex{font-size:13px;color:#64748b;cursor:pointer;padding:8px 12px;background:rgba(255,255,255,.03);border:1px solid rgba(255,255,255,.06);border-radius:8px;transition:all .2s}
.ex:hover{color:#06b6d4;border-color:#06b6d4}
</style></head>
<body>
<div class="card">
  <h1>💬 SentimentVibe</h1>
  <p class="sub">Análise de sentimentos de avaliações e textos via IA Gemini</p>
  <textarea id="txt" placeholder="Cole uma avaliação de produto, review ou qualquer texto para analisar..."></textarea>
  <button id="btn" onclick="analisar()">Analisar Sentimento →</button>
  <div class="examples">
    <div class="ex" onclick="usarExemplo(this)">Produto excelente! Chegou antes do prazo e embalagem impecável. Recomendo a todos!</div>
    <div class="ex" onclick="usarExemplo(this)">Muito dececionante. Qualidade péssima e suporte ao cliente não respondeu.</div>
    <div class="ex" onclick="usarExemplo(this)">O produto é OK, nada de especial. Entrega atrasou 2 dias.</div>
  </div>
  <div class="result" id="result"></div>
</div>
<script>
function usarExemplo(el){ document.getElementById('txt').value = el.textContent.trim(); }
async function analisar(){
  const text = document.getElementById('txt').value.trim();
  if(!text) return;
  const btn = document.getElementById('btn');
  btn.disabled = true; btn.textContent = 'A analisar...';
  try {
    const res = await fetch('/analyze', {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({text})});
    const data = await res.json();
    if(data.error){ alert(data.error); return; }
    const scoreColor = data.sentiment==='positive'?'#10b981':data.sentiment==='negative'?'#f43f5e':data.sentiment==='mixed'?'#f59e0b':'#94a3b8';
    document.getElementById('result').style.display='block';
    document.getElementById('result').innerHTML = `
      <span class="sentiment-badge ${data.sentiment}">${data.sentiment_pt} ${data.emoji}</span>
      <div>Confiança: ${data.confidence}%</div>
      <div class="score-bar"><div class="score-fill" style="width:${data.confidence}%;background:${scoreColor}"></div></div>
      <div class="analysis-text"><strong>Análise:</strong> ${data.analysis}</div>
      ${data.keywords?.length ? `<div class="analysis-text"><strong>Palavras-chave:</strong> ${data.keywords.join(', ')}</div>` : ''}
    `;
  } catch(e){ alert('Erro: ' + e.message); }
  finally{ btn.disabled=false; btn.textContent='Analisar Sentimento →'; }
}
</script></body></html>'''

def analyze_with_gemini(text: str) -> dict:
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        return {'sentiment': 'neutral', 'sentiment_pt': 'Neutro', 'emoji': '😐', 'confidence': 50, 'analysis': 'Configure GEMINI_API_KEY no .env para análise real.', 'keywords': []}
    import google.generativeai as genai
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-1.5-flash')
    prompt = f"""Analisa o sentimento do seguinte texto em português.
Responde APENAS em JSON com este formato:
{{"sentiment":"positive|negative|neutral|mixed","sentiment_pt":"Positivo|Negativo|Neutro|Misto","emoji":"emoji","confidence":0-100,"analysis":"análise em 1-2 frases","keywords":["palavra1","palavra2"]}}

Texto: {text[:1000]}"""
    import json, re
    try:
        response = model.generate_content(prompt)
        json_match = re.search(r'\{.*\}', response.text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
    except: pass
    return {'sentiment': 'neutral', 'sentiment_pt': 'Neutro', 'emoji': '😐', 'confidence': 50, 'analysis': 'Não foi possível analisar.', 'keywords': []}

@app.route('/')
def index(): return render_template_string(HTML)

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.json
    text = data.get('text', '').strip()
    if not text: return jsonify({'error': 'Texto é obrigatório.'}), 400
    if len(text) > 5000: return jsonify({'error': 'Texto demasiado longo (máx. 5000 caracteres).'}), 400
    return jsonify(analyze_with_gemini(text))

if __name__ == '__main__':
    print('💬 SentimentVibe em http://localhost:5000')
    app.run(debug=True, port=5000)
