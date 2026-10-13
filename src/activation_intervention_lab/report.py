"""Self-contained accessible report: original inputs, effects, controls, measurements."""
import hashlib
import json


def render_report(report):
    payload = json.dumps(report, allow_nan=False).replace("<", "\\u003c").replace("&", "\\u0026")
    revision = hashlib.sha256(payload.encode()).hexdigest()
    return TEMPLATE.replace("__DATA__", payload).replace("__REVISION__", revision)


TEMPLATE = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="artifact-revision" content="__REVISION__"><title>Residual activation intervention lab</title>
<style>
:root{font-family:system-ui,sans-serif;color:#172b3a;background:#f4f6f8;color-scheme:light;font-size:16px}
body{margin:0}main{max-width:1100px;margin:auto;padding:32px 24px 60px}h1{font-size:2rem;line-height:1.2;margin:.3em 0}
h2{font-size:1.35rem;margin:1.8em 0 .6em}p{max-width:88ch;line-height:1.6}small{font-size:.9rem}
.eyebrow{color:#456170;font-weight:600}.notice{border-left:4px solid #9c6a00;background:#fff7e5;padding:14px 18px}
.controls{display:flex;gap:18px;flex-wrap:wrap;margin:24px 0}label{display:grid;gap:6px;font-weight:600}
select{font:inherit;padding:8px;border:1px solid #8193a1;background:white;border-radius:4px;max-width:100%}
table{border-collapse:collapse;width:100%;background:white;margin:12px 0}th,td{padding:10px 12px;border:1px solid #d5dfe4;text-align:left}
th{background:#eaf0f4}td{font-variant-numeric:tabular-nums}.scroll{overflow-x:auto}caption{text-align:left;font-weight:600;padding:8px 0}
button{font:inherit;border:1px solid transparent;border-radius:3px;padding:10px 8px;background:transparent;width:100%;white-space:nowrap;cursor:pointer}
button:hover{border-color:#456170}button[aria-pressed=true]{outline:3px solid #172b3a;outline-offset:-3px;font-weight:700}
button:focus-visible,select:focus-visible,summary:focus-visible{outline:3px solid #006faa;outline-offset:3px}
.pos{background:#d1eee5}.neg{background:#f5dfe0}.zero{background:#f2f5f6}.heat th,.heat td{padding:3px;text-align:center}
.heat th{padding:10px}.heat{min-width:580px}.detail{background:white;padding:16px 20px;border:1px solid #d5dfe4}
details{margin-top:18px;background:white;padding:14px;border:1px solid #d5dfe4}summary{font-weight:600;cursor:pointer}
pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:.88rem;line-height:1.5}code{overflow-wrap:anywhere}
a{color:#006090}footer{margin-top:30px;line-height:1.5;font-size:.9rem;color:#456170}
@media(max-width:640px){main{padding:20px 14px 40px}h1{font-size:1.7rem}th,td{padding:8px}.controls{display:block}label{margin-bottom:16px}}
@media print{body{background:white}main{max-width:none;padding:0}button{padding:3px}.scroll{overflow:visible}.heat{min-width:0}details{break-inside:avoid}}
@media(prefers-reduced-motion:reduce){*{scroll-behavior:auto!important}}
</style></head><body><main>
<div class="eyebrow">Trained Pythia-14m · CPU · Offline experiment report</div>
<h1>Which residual patch changes the next-token score?</h1>
<p id="scope"></p>
<p>Each cell replaces one layer/position activation in a corrupted forward pass with the matching activation from the clean input. Every patch has its own example group. Scores are fixed next-token probes, with logit difference and probability shown separately.</p>
<div class="notice">A patch effect depends on the input pair, corruption method, and metric. It does not establish a complete circuit or general language capability. Embedding noise can place activations outside the training distribution.</div>
<div class="controls">
<label>Corruption comparison<select id="condition"><option value="0">Country replacement A: Italy / China</option><option value="1">Country replacement B: Germany / Korea</option><option value="2">Seeded embedding noise</option></select></label>
<label>Metric<select id="metric"><option value="0">Answer logit difference</option><option value="1">Answer probability</option></select></label>
<label>Example aggregation<select id="example"><option value="mean">Mean of two examples</option><option value="0">Example 1: France / Paris</option><option value="1">Example 2: Japan / Tokyo</option></select></label>
</div>
<h2>Exact inputs and answer tokens</h2><div id="inputs" class="scroll"></div><p id="noise"></p>
<h2>Patch effect at each layer and position</h2>
<p id="metric-note"></p><p>Choose a cell to compare the clean, corrupted, and patched scores. Positive values increase the selected metric; negative values decrease it. Position labels use example 1; the exact token IDs are included in the evidence.</p>
<div id="heatmap" class="scroll"></div>
<div class="detail" aria-live="polite"><strong id="selection"></strong><div id="comparison" class="scroll"></div></div>
<h2>Controls and numerical agreement</h2><div id="checks" class="scroll"></div>
<p id="sensitivity"></p>
<h2>All measured runs</h2><p>Forward counts include one corrupted baseline call per sweep. Timings retain every repetition with a rotated execution order. RSS is sampled every 5 ms, can miss short peaks, and includes the Python process, model and allocator. Reduced call count does not itself establish a speed gain.</p>
<div id="benchmark" class="scroll"></div>
<details><summary>Reproducibility, cache and model evidence</summary><pre id="evidence"></pre></details>
<footer>Method reference: <a href="https://arxiv.org/abs/2309.16042">Zhang &amp; Nanda, ICLR 2024</a>. Implementation reference: <a href="https://github.com/TransformerLensOrg/TransformerLens/tree/bd55eab6737584685b921fdfcd15f191579e64b1">TransformerLens pinned source</a>. This report is fully local and makes no network requests. Revision <code>__REVISION__</code>.</footer>
</main><script type="application/json" id="data">__DATA__</script>
<script>
'use strict';
const data=JSON.parse(document.getElementById('data').textContent);
const get=id=>document.getElementById(id);let layer=0,position=3;
const number=v=>Number.isFinite(v)?v.toPrecision(6):'undefined';
function table(headers,rows,caption){const t=document.createElement('table');if(caption){const c=document.createElement('caption');c.textContent=caption;t.append(c)}const head=document.createElement('thead'),tr=document.createElement('tr');headers.forEach(h=>{const th=document.createElement('th');th.scope='col';th.textContent=h;tr.append(th)});head.append(tr);t.append(head);const body=document.createElement('tbody');rows.forEach(row=>{const r=document.createElement('tr');row.forEach(value=>{const td=document.createElement('td');td.textContent=String(value);r.append(td)});body.append(r)});t.append(body);return t}
function place(id,element){get(id).replaceChildren(element)}
function render(){const c=data.conditions[Number(get('condition').value)],m=Number(get('metric').value),e=get('example').value;
get('scope').textContent=data.scope;
place('inputs',table(['Example','Clean input','Corrupted input','Answer','Alternative'],c.clean_inputs.map((s,i)=>[i+1,s,c.corrupted_inputs[i],JSON.stringify(c.answers[i]),JSON.stringify(c.alternatives[i])])));
get('noise').textContent=c.noise?`The text is unchanged; Gaussian noise is added at position ${c.noise.position}, with seed ${c.noise.seed} and scale ${c.noise.scale} × embedding standard deviation (${number(c.noise.embedding_std)}).`:'The changed country token is replaced with a semantically related country token.';
get('metric-note').textContent=m===0?'Logit difference = answer logit − alternative logit. It is an unbounded score, not a probability.':'Answer probability = softmax probability of the single answer token. It is measured in probability units, not logits.';
const effects=c.patch_effects[m],aggregate=a=>e==='mean'?a.reduce((x,y)=>x+y,0)/a.length:a[Number(e)];
const t=document.createElement('table');t.className='heat';const cap=document.createElement('caption');cap.textContent='Change from the corrupted score';t.append(cap);const h=document.createElement('tr');let th=document.createElement('th');th.textContent='Layer';h.append(th);c.token_labels[0].forEach((token,p)=>{const cell=document.createElement('th');cell.scope='col';cell.textContent=`${p}: ${JSON.stringify(token)}`;h.append(cell)});t.append(h);
effects.forEach((row,l)=>{const r=document.createElement('tr');const label=document.createElement('th');label.scope='row';label.textContent=l;r.append(label);row.forEach((value,p)=>{const effect=aggregate(value),td=document.createElement('td'),button=document.createElement('button');td.className=Math.abs(effect)<1e-7?'zero':effect>0?'pos':'neg';button.textContent=number(effect);button.setAttribute('aria-label',`Layer ${l}, position ${p}, effect ${number(effect)}`);button.setAttribute('aria-pressed',String(l===layer&&p===position));button.onclick=()=>{layer=l;position=p;render()};td.append(button);r.append(td)});t.append(r)});place('heatmap',t);
get('selection').textContent=`Selected intervention: residual input at layer ${layer}, position ${position}`;
place('comparison',table(['Example','Clean','Corrupted','Patched','Patch effect'],c.answers.map((_,i)=>[i+1,number(c.clean_metrics[m][i]),number(c.corrupted_metrics[m][i]),number(c.patched_metrics[m][layer][position][i]),number(c.patch_effects[m][layer][position][i])])));
place('checks',table(['Check','Observed result'],Object.entries(c.checks).map(([k,v])=>[k.replaceAll('_',' '),typeof v==='number'?number(v):v])));
get('sensitivity').textContent=`Within this condition, metric rankings differ: ${c.metric_ranking_differs}. Across the three corruption conditions, full patch rankings differ for logit difference: ${data.corruption_ranking_differs.logit_difference}; probability: ${data.corruption_ranking_differs.probability}. These are observed rankings for these probes only.`;
const rows=[];Object.entries(c.measurements).forEach(([mode,runs])=>runs.forEach(r=>rows.push([mode.replaceAll('_',' '),r.repetition+1,r.forward_calls,number(r.seconds),number(r.sampled_peak_rss_bytes/1048576),number(r.sampled_peak_delta_bytes/1048576)])));
place('benchmark',table(['Mode','Run','Forward calls','Seconds','Sampled peak RSS (MiB)','Peak − start (MiB)'],rows));
get('evidence').textContent=JSON.stringify({model:data.model,checkpoint_files:data.checkpoint_files,upstream:data.upstream,environment:data.environment,admission:data.admission,numerical_policy:data.numerical_policy,unpatched_batch_only_logits_max_abs:c.unpatched_batch_only_logits_max_abs,serial_vs_batched_rankings_equal:c.serial_vs_batched_rankings_equal,smallest_clean_corrupted_metric_gap:c.smallest_clean_corrupted_metric_gap,clean_token_ids:c.clean_token_ids,corrupted_token_ids:c.corrupted_token_ids,answer_token_ids:c.answer_token_ids,alternative_token_ids:c.alternative_token_ids,corrupted_top5_tokens:c.corrupted_top5_tokens,noise:c.noise,remaining_hooks:data.remaining_hooks,cache_released:data.cache_released,paper:data.paper},null,2);
}
['condition','metric','example'].forEach(id=>get(id).addEventListener('change',render));render();
</script></body></html>'''
