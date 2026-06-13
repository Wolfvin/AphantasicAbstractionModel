#!/usr/bin/env python3
"""Empirical benchmark v38 — incremental output"""
import os, sys, json, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'src'))
os.environ['TOKENIZERS_PARALLELISM'] = '0'
import logging; logging.disable(logging.WARNING)

# Clean patterns
pf = os.path.join(ROOT, 'data', 'learned_patterns.json')
if os.path.exists(pf): os.remove(pf)

OUTPUT = os.path.join(ROOT, 'benchmark', 'empirical_v38_results.json')

from derivation.text_comprehension import TextComprehension
tc = TextComprehension()

def check(answer, keywords):
    if answer is None: return False
    a = str(answer).lower().strip()
    return any(k.lower() in a for k in keywords)

def log(msg):
    print(msg, flush=True)

SOAL = [
    ('PB', 'Orang tua Siti bekerja keras siang dan malam.', 'Peribahasa untuk kerja keras?', ['banting tulang', 'kerja keras']),
    ('BK', 'Angin menjerit keras menggoyangkan pepohonan.', 'Kata menjerit termasuk majas....', ['personifikasi']),
    ('IM', 'Hujan turun tanpa henti. Sungai meluap.', 'Mengapa sungai meluap?', ['hujan', 'banjir']),
    ('CP', 'Rani belajar setiap malam. Doni hanya membaca sepintas.', 'Perbedaan cara belajar?', ['rajin', 'malas', 'tekun']),
    ('EK', 'Paman pergi ke pasar pada pukul 05.00 pagi.', 'Pukul berapa paman pergi?', ['05.00', '5']),
]

TEACH = [
    ('Para petani banting tulang dari subuh hingga magrib.', 'Peribahasa untuk kerja keras?', 'banting tulang', 'kerja keras fisik → banting tulang'),
    ('Hujan menari-nari di atap rumah.', 'Kata menari-nari termasuk majas....', 'personifikasi', 'non-human + human action → personifikasi'),
    ('Kebakaran gudang. Toko kehilangan persediaan.', 'Mengapa toko kehilangan?', 'kebakaran gudang', 'A→B→C, root cause = A'),
    ('Andi menabung teratur. Budi menghabiskan uang.', 'Perbedaan mengelola uang?', 'hemat vs boros', 'compare → abstract quality'),
    ('Dokter praktik mulai pukul 08.00 pagi.', 'Pukul berapa dokter praktik?', '08.00', 'pukul berapa → find time'),
]

GEN = [
    ('G-PB', 'Nelayan mengayuh perahu pagi buta hingga petang.', 'Peribahasa semangat kerja?', ['banting tulang', 'kerja keras']),
    ('G-BK', 'Bintang berkelipkan mata di langit malam.', 'Kata berkelipkan mata majas....', ['personifikasi']),
    ('G-IM', 'Gempa menghancurkan jembatan. Truk tidak lewat.', 'Mengapa truk tidak lewat?', ['gempa', 'jembatan']),
    ('G-CP', 'Eko kerja PR tepat waktu. Fajar menunda.', 'Perbedaan kebiasaan?', ['rajin', 'disiplin', 'malas']),
    ('G-EK', 'Perpustakaan buka pukul 09.00 pagi.', 'Pukul berapa buka?', ['09.00', '9']),
]

results = {'version': 'v38_empirical', 'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')}

# Phase 1
log('PHASE1:BASELINE')
p1 = []
for sid, text, q, kw in SOAL:
    t0 = time.time()
    r = tc.comprehend(text, q)
    dt = time.time() - t0
    ok = check(r.get('answer'), kw)
    p1.append({'id': sid, 'pass': ok, 'method': r.get('method',''), 'conf': round(r.get('confidence',0),3), 'time': round(dt,1)})
    log(f'  {"P" if ok else "F"} {sid}: {r.get("method","")[:50]} c={r.get("confidence",0):.2f} t={dt:.0f}s')

results['phase1'] = p1
results['p1_acc'] = sum(1 for r in p1 if r['pass']) / len(p1)

# Phase 2
log('PHASE2:TEACH')
for i, (t, q, a, e) in enumerate(TEACH):
    tc.teach(t, q, a, e)
    log(f'  taught {i+1}/5')

p_emb = sum(1 for pk, pd in tc.learned_patterns.items() if pd.get('question_embedding') and len(pd['question_embedding']) > 0)
results['patterns'] = len(tc.learned_patterns)
results['patterns_with_emb'] = p_emb

# Phase 3
log('PHASE3:RETEST')
p3 = []
for sid, text, q, kw in SOAL:
    t0 = time.time()
    r = tc.comprehend(text, q)
    dt = time.time() - t0
    ok = check(r.get('answer'), kw)
    was = next(x['pass'] for x in p1 if x['id'] == sid)
    chg = ' IMPROVED' if ok and not was else (' DEGRADED' if not ok and was else '')
    p3.append({'id': sid, 'pass': ok, 'method': r.get('method',''), 'conf': round(r.get('confidence',0),3), 'time': round(dt,1)})
    log(f'  {"P" if ok else "F"} {sid}: {r.get("method","")[:50]} c={r.get("confidence",0):.2f} t={dt:.0f}s{chg}')

results['phase3'] = p3
results['p3_acc'] = sum(1 for r in p3 if r['pass']) / len(p3)
results['net_improvement'] = sum(1 for i,r in enumerate(p3) if r['pass'] and not p1[i]['pass']) - sum(1 for i,r in enumerate(p3) if not r['pass'] and p1[i]['pass'])

# Phase 4
log('PHASE4:GEN')
p4 = []
for sid, text, q, kw in GEN:
    t0 = time.time()
    r = tc.comprehend(text, q)
    dt = time.time() - t0
    ok = check(r.get('answer'), kw)
    p4.append({'id': sid, 'pass': ok, 'method': r.get('method',''), 'conf': round(r.get('confidence',0),3), 'time': round(dt,1)})
    log(f'  {"P" if ok else "F"} {sid}: {r.get("method","")[:50]} c={r.get("confidence",0):.2f} t={dt:.0f}s')

results['phase4'] = p4
results['p4_acc'] = sum(1 for r in p4 if r['pass']) / len(p4)

# Method breakdown
all_r = p1 + p3 + p4
methods = {}
for r in all_r:
    m = r['method']
    if m not in methods: methods[m] = [0, 0]
    methods[m][0] += 1
    if r['pass']: methods[m][1] += 1
results['methods'] = {m: {'total': c[0], 'correct': c[1]} for m, c in methods.items()}

# Save
os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
with open(OUTPUT, 'w') as f:
    json.dump(results, f, indent=2, default=str)

log(f'DONE p1={results["p1_acc"]:.0%} p3={results["p3_acc"]:.0%} p4={results["p4_acc"]:.0%} net={results["net_improvement"]:+d}')
