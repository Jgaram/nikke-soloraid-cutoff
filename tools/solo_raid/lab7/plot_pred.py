"""예측 그림 (2026-11-01, 사용자 요청 「G2 보고에 22시즌 그래프도 붙여 줘」) — 칸 예측 파일을 정답 표와 함께 HTML 한 장으로.

  python3 -I tools/solo_raid/lab7/plot_pred.py --pred r7/pred_M3dt_D1.json --name "M3 dt" \
      --ref r7/F_D1.json --out $SCRATCH/g2.html [--seasons S21-S42] [--title ...]

- 위: 시즌별 한눈에 — 초반(T 24)·막판(T = H) 칸에서 정답 대비 오차와 80% 구간, 1·3·10%. 조정 시즌은 이름 옆에 ●.
- 아래: 시즌 하나를 골라 T마다 중앙·80% 구간·정답 (2026-10-31 채팅 그림과 같은 꼴).
- 맨 위 숫자(평균 |e|·80% 포함)는 그림용 간단 셈이다. 고르기·보고의 공식 숫자는 score.py.
한섭(KR) 칸만, 정답은 answer_key.json의 KR truth. 숫자만 읽고 쓴다 — 글·사진·닉네임 없음.
HTML은 레포에 커밋하지 않는다 (AGENTS.md 「결과물」 — 세션 임시 폴더에 두고 채팅으로 보낸다).
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
TUNE = ["S22", "S26", "S30", "S34", "S38", "S42"]   # common.TUNE과 같다 (common은 predict를 불러 무거워 따로 둠)
PCTS = [1.0, 3.0, 10.0]
EARLY = [18.0, 24.0, 36.0, 48.0]


def _load(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def parse_seasons(spec):
    out = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-")
            out += [f"S{n}" for n in range(int(a.lstrip("S")), int(b.lstrip("S")) + 1)]
        elif part:
            out.append(part)
    return out


def cells_of(path, seasons):
    d = {}
    for c in _load(path)["cells"]:
        if c.get("server", "KR") != "KR" or c["season"] not in seasons or c["pct"] not in PCTS:
            continue
        d[(c["season"], c["pct"], float(c["T"]))] = c
    return d


def build(args):
    main = _load(args.pred)
    seasons = parse_seasons(args.seasons) if args.seasons else sorted({c["season"] for c in main["cells"]}, key=lambda s: int(s[1:]))
    truth = {}
    for r in _load(args.answer)["rows"]:
        if r["server"] == "KR" and r["season"] in seasons and r["pct"] in PCTS and r["truth"] is not None:
            truth.setdefault((r["season"], r["pct"]), r["truth"])
    M = cells_of(args.pred, seasons)
    F = cells_of(args.ref, seasons) if args.ref else {}
    D, stat = {}, {}
    for s in seasons:
        Ts = [t for (ss, p, t) in M if ss == s]
        if not Ts:
            continue
        H = max(Ts)
        D[s] = {"H": H, "tune": s in TUNE, "p": {}}
        for p in PCTS:
            row = lambda c: [c["T"], round(c["pred"], 3), round(c["i80"][0], 3), round(c["i80"][1], 3), int(bool(c.get("conv", True)))]
            m = [row(M[k]) for k in sorted(k for k in M if k[0] == s and k[1] == p)]
            f = [row(F[k]) for k in sorted(k for k in F if k[0] == s and k[1] == p and k[2] <= H)]
            D[s]["p"][str(int(p))] = {"truth": truth.get((s, p)), "M": m, "F": f}
    # 그림용 간단 셈: 창마다 (시즌 평균 |e|의 평균, 80% 포함 비율), 1·3%와 10%
    for lab, src in (("M", M), ("F", F)):
        if not src:
            continue
        for grp, ps in (("13", [1.0, 3.0]), ("10", [10.0])):
            for win in ("early", "end"):
                es, cov, n = [], 0, 0
                for s in D:
                    H = D[s]["H"]
                    Tw = EARLY if win == "early" else [H - 12, H - 6, H]
                    for p in ps:
                        tr = truth.get((s, p))
                        if tr is None:
                            continue
                        e = [abs(src[(s, p, t)]["pred"] / tr - 1) for t in Tw if (s, p, t) in src]
                        for t in Tw:
                            c = src.get((s, p, t))
                            if c:
                                n += 1
                                cov += c["i80"][0] <= tr <= c["i80"][1]
                        if e:
                            es.append(sum(e) / len(e))
                if n:
                    stat[f"{lab}_{grp}_{win}"] = [round(100 * sum(es) / len(es), 2), round(cov / n, 2), n]
    return D, stat


HTML = r"""<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>예측 그림</title>
<style>
:root{color-scheme:light;--bg:#fcfcfb;--t1:#0b0b0b;--t2:#52514e;--t3:#8a8984;--grid:#e6e5e1;--s1:#2a78d6;--s1b:rgba(42,120,214,.22);--s2:#eb6834;--s2b:rgba(235,104,52,.13);--btn:#efeeea}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){color-scheme:dark;--bg:#1a1a19;--t1:#f0efec;--t2:#c3c2b7;--t3:#8f8e86;--grid:#33332f;--s1:#3987e5;--s1b:rgba(57,135,229,.30);--s2:#d95926;--s2b:rgba(217,89,38,.18);--btn:#2a2a28}}
:root[data-theme="dark"]{color-scheme:dark;--bg:#1a1a19;--t1:#f0efec;--t2:#c3c2b7;--t3:#8f8e86;--grid:#33332f;--s1:#3987e5;--s1b:rgba(57,135,229,.30);--s2:#d95926;--s2b:rgba(217,89,38,.18);--btn:#2a2a28}
body{margin:0;background:var(--bg);color:var(--t1);font:14px/1.5 system-ui,-apple-system,"Apple SD Gothic Neo","Noto Sans KR",sans-serif}
main{max-width:760px;margin:0 auto;padding:16px}
h1{font-size:18px;margin:0 0 4px}h3{font-size:15px;margin:22px 0 2px}p.sub{color:var(--t2);margin:0 0 10px;font-size:13px}
.row{display:flex;flex-wrap:wrap;gap:6px;margin:8px 0}
button{border:0;background:var(--btn);color:var(--t1);padding:6px 10px;border-radius:999px;font:inherit;font-size:13px;cursor:pointer}
button.on{background:var(--t1);color:var(--bg)}
.legend{display:flex;flex-wrap:wrap;gap:14px;font-size:12px;color:var(--t2);margin:6px 0 4px}
.legend i{display:inline-block;width:18px;height:10px;vertical-align:-1px;margin-right:5px;border-radius:2px}
.card{margin:14px 0 6px}.card h2{font-size:14px;margin:0;font-weight:600}.card .note{font-size:12px;color:var(--t2)}
svg{width:100%;height:auto;display:block;touch-action:pan-y}svg text{fill:var(--t3);font-size:11px}
.tip{position:fixed;pointer-events:none;background:var(--bg);color:var(--t1);border:1px solid var(--grid);border-radius:6px;padding:6px 8px;font-size:12px;box-shadow:0 2px 8px rgba(0,0,0,.15);display:none;z-index:9;white-space:nowrap}
details{margin-top:14px;font-size:13px;color:var(--t2)}table{border-collapse:collapse;font-size:12px;width:100%}td,th{padding:3px 6px;text-align:right;border-bottom:1px solid var(--grid)}
.box{background:var(--btn);border-radius:8px;padding:10px 12px;font-size:13px;color:var(--t2);margin:10px 0}.box b{color:var(--t1)}
</style></head><body><main>
<h1 id="ttl"></h1><p class="sub" id="sub"></p>
<div class="box" id="stat"></div>
<div class="box"><b>보는 법</b> — 점이 0% 선에 붙을수록 중앙값이 정확하고, 세로 막대(80% 구간)가 0% 선을 건너야 구간이 정답을 덮은 것이다. 열 시즌 중 여덟 시즌쯤 건너면 약속대로다. ● 표시는 조정에 쓴 시즌 — 나머지 시즌과 차이가 크면 조정 시즌에 맞춰진 것이다.</div>
<div class="legend" id="lg"></div>
<h3>시즌별 한눈에</h3><div class="row" id="ow"></div><div id="over"></div>
<h3>시즌 하나 자세히</h3><p class="sub">점선 = 실제 최종 컷. 회색 띠 = 점수에 쓰는 초반·막판 창. 그래프를 누르면 그 시각 숫자.</p>
<div class="row" id="seasons"></div><div class="row" id="views"></div><div id="charts"></div>
<details><summary>표로 보기 (고른 시즌)</summary><div id="tbl"></div></details>
<p class="sub" style="margin-top:14px" id="src"></p>
</main><div class="tip" id="tip"></div>
<script>
const D=__DATA__,ST=__STAT__,META=__META__;
const S=Object.keys(D),P=['1','3','10'];let cs=S.find(s=>D[s].tune)||S[0],view='err',ow='end';
const $=id=>document.getElementById(id),fm=x=>x==null?'':x.toFixed(Math.abs(x)>=100?1:2),sg=z=>(z>0?'+':'')+z.toFixed(2)+'%';
const hasF=S.some(s=>P.some(p=>D[s].p[p].F.length));
$('ttl').textContent=META.title;$('sub').textContent=META.sub;$('src').textContent=META.src;
$('lg').innerHTML=`<span><i style="background:var(--s1)"></i>${META.name} 중앙</span><span><i style="background:var(--s1b)"></i>${META.name} 80% 구간</span>`+(hasF?`<span><i style="background:var(--s2);height:3px;vertical-align:3px"></i>${META.ref} (참고)</span>`:'')+`<span><i style="border-top:2px dashed var(--t1);height:0;border-radius:0;vertical-align:3px"></i>정답</span>`;
const st=(k)=>ST[k]?`평균 |e| ${ST[k][0]}% · 80% 포함 ${Math.round(ST[k][1]*100)}% (${ST[k][2]}칸)`:'—';
$('stat').innerHTML=`<b>${META.name}</b> 1·3% — 초반 ${st('M_13_early')} / 막판 ${st('M_13_end')}<br>10% — 초반 ${st('M_10_early')} / 막판 ${st('M_10_end')}`+
 (ST.F_13_end?`<br><span style="opacity:.8">${META.ref} 1·3% — 초반 ${st('F_13_early')} / 막판 ${st('F_13_end')}</span>`:'')+`<br><span style="font-size:12px">그림용 간단 셈 (공식 점수·고르기는 score.py). 포함 목표 70% 이상.</span>`;
const Wd=()=>Math.max(300,Math.min(720,document.querySelector('main').clientWidth-32));
function ticks(lo,hi){const rs=(hi-lo)/4,mg=Math.pow(10,Math.floor(Math.log10(rs))),st=[1,2,2.5,5,10].map(x=>x*mg).find(x=>x>=rs),o=[];for(let v=Math.ceil(lo/st)*st;v<=hi+1e-9;v+=st)o.push(+v.toFixed(6));return o;}
function pick(a,T){return a.find(r=>r[0]==T);}
function overview(){$('ow').innerHTML=[['early','초반 (T 24h)'],['end','막판 (시즌 끝)']].map(([k,l])=>`<button class="${k==ow?'on':''}" data-o="${k}">${l}</button>`).join('');
 const W=Wd(),H=190,ml=44,mr=8,mt=10,mb=34;let html='';
 for(const p of P){const pts=[];for(const s of S){const d=D[s].p[p];if(d.truth==null)continue;const T=ow=='end'?D[s].H:24;
   const m=pick(d.M,T),f=pick(d.F,T),cv=v=>(v/d.truth-1)*100;pts.push({s,tune:D[s].tune,m:m&&[cv(m[1]),cv(m[2]),cv(m[3])],f:f&&[cv(f[1]),cv(f[2]),cv(f[3])],T,tr:d.truth,mr:m,fr:f});}
  if(!pts.length)continue;const all=pts.flatMap(q=>[...(q.m||[]),...(q.f||[])]);let lo=Math.min(0,...all),hi=Math.max(0,...all);const lim=(ow=='end'?6:15)*(p=='10'?1.7:1);lo=Math.max(lo,-lim);hi=Math.min(hi,lim);const pd=(hi-lo)*.08||.5;lo-=pd;hi+=pd;
  const step=(W-ml-mr)/pts.length,X=i=>ml+step*(i+.5),Y=v=>mt+(1-(Math.max(lo,Math.min(hi,v))-lo)/(hi-lo))*(H-mt-mb);
  let g='';for(const v of ticks(lo,hi))g+=`<line x1="${ml}" x2="${W-mr}" y1="${Y(v)}" y2="${Y(v)}" stroke="var(--grid)"/><text x="${ml-5}" y="${Y(v)+4}" text-anchor="end">${v>0?'+':''}${v}%</text>`;
  g+=`<line x1="${ml}" x2="${W-mr}" y1="${Y(0)}" y2="${Y(0)}" stroke="var(--t1)" stroke-width="1.2" stroke-dasharray="5 4"/>`;
  pts.forEach((q,i)=>{const x=X(i),o=hasF?Math.min(5,step*.18):0;
   if(q.f)g+=`<line x1="${x+o}" x2="${x+o}" y1="${Y(q.f[1])}" y2="${Y(q.f[2])}" stroke="var(--s2)" stroke-width="2" opacity=".55"/><circle cx="${x+o}" cy="${Y(q.f[0])}" r="2.5" fill="var(--s2)" opacity=".7"/>`;
   if(q.m)g+=`<line x1="${x-o}" x2="${x-o}" y1="${Y(q.m[1])}" y2="${Y(q.m[2])}" stroke="var(--s1)" stroke-width="3" stroke-linecap="round" opacity=".55"/><circle cx="${x-o}" cy="${Y(q.m[0])}" r="4" fill="var(--s1)" stroke="var(--bg)" stroke-width="1.5"/>`;
   g+=`<text x="${x}" y="${H-mb+14}" text-anchor="middle" transform="rotate(-50 ${x} ${H-mb+14})" style="font-size:10px${q.tune?';font-weight:700;fill:var(--t1)':''}">${q.s.slice(1)}${q.tune?'●':''}</text>`;
   g+=`<rect class="hit" data-i="${i}" x="${x-step/2}" y="0" width="${step}" height="${H}" fill="transparent"/>`;});
  html+=`<div class="card"><h2>${p}% 컷</h2><svg viewBox="0 0 ${W} ${H}" data-p="${p}">${g}</svg></div>`;
  overview['_'+p]=pts;}
 $('over').innerHTML=html;
 document.querySelectorAll('#over svg').forEach(sv=>{const pts=overview['_'+sv.dataset.p];
  const mv=e=>{const r=e.target.closest('.hit');if(!r)return;const q=pts[+r.dataset.i],pt=e.touches?e.touches[0]:e;
   tip.innerHTML=`<b>${q.s}${q.tune?' (조정)':''}</b> · T ${q.T}h · 정답 ${fm(q.tr)}`+(q.m?`<br>${META.name} ${fm(q.mr[1])} (${sg(q.m[0])}) · 80% ${fm(q.mr[2])}~${fm(q.mr[3])} ${q.m[1]<=0&&q.m[2]>=0?'— 포함':'— 밖'}`:'')+(q.f?`<br>${META.ref} ${fm(q.fr[1])} (${sg(q.f[0])})`:'');showTip(pt);};
  sv.addEventListener('mousemove',mv);sv.addEventListener('touchstart',mv,{passive:true});sv.addEventListener('mouseleave',hide);sv.addEventListener('touchend',()=>setTimeout(hide,1500));
  sv.addEventListener('click',e=>{const r=e.target.closest('.hit');if(r){cs=pts[+r.dataset.i].s;detail();}});});}
const tip=$('tip');function showTip(pt){tip.style.display='block';tip.style.left=Math.max(8,Math.min(pt.clientX+12,innerWidth-tip.offsetWidth-8))+'px';tip.style.top=Math.max(8,pt.clientY-tip.offsetHeight-12)+'px';}
function hide(){tip.style.display='none';document.querySelectorAll('.cross').forEach(c=>c.setAttribute('visibility','hidden'));}
function detail(){$('seasons').innerHTML=S.map(s=>`<button class="${s==cs?'on':''}" data-s="${s}">${s}${D[s].tune?'●':''}</button>`).join('');
 $('views').innerHTML=[['err','정답 대비 %'],['raw','컷 점수 그대로']].map(([k,l])=>`<button class="${k==view?'on':''}" data-v="${k}">${l}</button>`).join('');
 const W=Wd(),H=210,ml=48,mr=10,mt=10,mb=26,Hs=D[cs].H;let html='',tb='';
 for(const p of P){const d=D[cs].p[p];if(d.truth==null||!d.M.length){html+=`<div class="card"><h2>${p}% 컷</h2><div class="note">정답 없음</div></div>`;continue;}
  const tr=d.truth,cv=v=>view=='err'?(v/tr-1)*100:v,m=d.M.map(r=>[r[0],cv(r[1]),cv(r[2]),cv(r[3])]),f=d.F.map(r=>[r[0],cv(r[1]),cv(r[2]),cv(r[3])]),tv=cv(tr);
  const sub=[...m,...f].filter(r=>r[0]>=18);let lo=Math.min(tv,...sub.map(r=>r[2])),hi=Math.max(tv,...sub.map(r=>r[3]));const pd=(hi-lo)*.12||1;lo-=pd;hi+=pd;
  const X=t=>ml+(t/Hs)*(W-ml-mr),Y=v=>mt+(1-(Math.max(lo,Math.min(hi,v))-lo)/(hi-lo))*(H-mt-mb);
  const band=a=>a.length?'M'+a.map(r=>X(r[0])+','+Y(r[3])).join('L')+'L'+a.slice().reverse().map(r=>X(r[0])+','+Y(r[2])).join('L')+'Z':'';
  const line=a=>a.length?'M'+a.map(r=>X(r[0])+','+Y(r[1])).join('L'):'';
  let g=`<rect x="${X(18)}" y="${mt}" width="${X(48)-X(18)}" height="${H-mt-mb}" fill="var(--grid)" opacity=".35"/><rect x="${X(Hs-12)}" y="${mt}" width="${X(Hs)-X(Hs-12)}" height="${H-mt-mb}" fill="var(--grid)" opacity=".5"/>`;
  for(const v of ticks(lo,hi))g+=`<line x1="${ml}" x2="${W-mr}" y1="${Y(v)}" y2="${Y(v)}" stroke="var(--grid)"/><text x="${ml-6}" y="${Y(v)+4}" text-anchor="end">${view=='err'?(v>0?'+':'')+v+'%':v}</text>`;
  for(let t=0;t<=Hs;t+=24)g+=`<text x="${X(t)}" y="${H-8}" text-anchor="middle">${t}h</text>`;
  g+=`<text x="${(X(18)+X(48))/2}" y="${mt+12}" text-anchor="middle">초반 창</text><text x="${X(Hs-6)}" y="${mt+12}" text-anchor="middle">막판</text>`;
  g+=`<path d="${band(f)}" fill="var(--s2b)"/><path d="${band(m)}" fill="var(--s1b)"/><line x1="${ml}" x2="${W-mr}" y1="${Y(tv)}" y2="${Y(tv)}" stroke="var(--t1)" stroke-width="1.5" stroke-dasharray="5 4"/>`;
  g+=`<path d="${line(f)}" fill="none" stroke="var(--s2)" stroke-width="1.6" stroke-dasharray="4 3"/><path d="${line(m)}" fill="none" stroke="var(--s1)" stroke-width="2"/>`;
  g+=d.M.filter(r=>!r[4]).map(r=>`<circle cx="${X(r[0])}" cy="${Y(cv(r[1]))}" r="3" fill="none" stroke="var(--t2)"/>`).join('');
  g+=`<line class="cross" x1="0" x2="0" y1="${mt}" y2="${H-mb}" stroke="var(--t3)" visibility="hidden"/>`;
  const L=d.M[d.M.length-1];
  html+=`<div class="card"><h2>${p}% 컷 · 정답 ${fm(tr)}</h2><div class="note">끝(T=${Hs}h) ${META.name} ${fm(L[1])} (${sg((L[1]/tr-1)*100)}), 80% ${fm(L[2])}~${fm(L[3])} ${tr>=L[2]&&tr<=L[3]?'— 정답 포함':'— 정답 밖'}${d.M.some(r=>!r[4])?' · 빈 원 = 수렴 못 한 칸':''}</div><svg viewBox="0 0 ${W} ${H}" data-p="${p}" data-w="${W}" data-ml="${ml}" data-mr="${mr}">${g}<rect x="${ml}" y="0" width="${W-ml-mr}" height="${H}" fill="transparent"/></svg></div>`;
  tb+=`<h3 style="font-size:13px">${p}% (정답 ${fm(tr)})</h3><table><tr><th>T</th><th>${META.name}</th><th>80%</th>${hasF?`<th>${META.ref}</th><th>80%</th>`:''}</tr>`+d.M.map(r=>{const q=pick(d.F,r[0]);return `<tr><td>${r[0]}</td><td>${fm(r[1])}</td><td>${fm(r[2])}~${fm(r[3])}</td>${hasF?`<td>${q?fm(q[1]):''}</td><td>${q?fm(q[2])+'~'+fm(q[3]):''}</td>`:''}</tr>`}).join('')+'</table>';}
 $('charts').innerHTML=html;$('tbl').innerHTML=tb;
 document.querySelectorAll('#charts svg').forEach(sv=>{const d=D[cs].p[sv.dataset.p],W=+sv.dataset.w,ml=+sv.dataset.ml,mr=+sv.dataset.mr,cr=sv.querySelector('.cross');
  const mv=e=>{const pt=e.touches?e.touches[0]:e,b=sv.getBoundingClientRect(),t=((pt.clientX-b.left)/b.width*W-ml)/(W-ml-mr)*Hs;
   const r=d.M.reduce((a,c)=>Math.abs(c[0]-t)<Math.abs(a[0]-t)?c:a),q=pick(d.F,r[0]),X=ml+r[0]/Hs*(W-ml-mr),e2=v=>sg((v/d.truth-1)*100);
   cr.setAttribute('x1',X);cr.setAttribute('x2',X);cr.setAttribute('visibility','visible');
   tip.innerHTML=`<b>T = ${r[0]}h</b> · 정답 ${fm(d.truth)}<br>${META.name} ${fm(r[1])} (${e2(r[1])}) · 80% ${fm(r[2])}~${fm(r[3])}${r[4]?'':' · 수렴 못 함'}`+(q?`<br>${META.ref} ${fm(q[1])} (${e2(q[1])}) · 80% ${fm(q[2])}~${fm(q[3])}`:'');showTip(pt);};
  sv.addEventListener('mousemove',mv);sv.addEventListener('touchstart',mv,{passive:true});sv.addEventListener('touchmove',mv,{passive:true});sv.addEventListener('mouseleave',hide);sv.addEventListener('touchend',()=>setTimeout(hide,1500));});}
document.addEventListener('click',e=>{const b=e.target.closest('button');if(!b)return;if(b.dataset.s){cs=b.dataset.s;detail();}if(b.dataset.v){view=b.dataset.v;detail();}if(b.dataset.o){ow=b.dataset.o;overview();}});
overview();detail();
</script></body></html>
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--pred", required=True, help="칸 예측 파일 (score.py 꼴)")
    ap.add_argument("--name", default=None, help="그림에 쓸 방법 이름 (기본: 파일의 method)")
    ap.add_argument("--ref", default=None, help="참고 줄 예측 파일 (예: r7/F_D1.json)")
    ap.add_argument("--ref-name", default="F 기존")
    ap.add_argument("--answer", default=os.path.join(ROOT, "data", "solo-raid", "analysis", "answer_key.json"))
    ap.add_argument("--seasons", default=None, help="예: S21-S42 (기본: 예측 파일의 시즌 모두)")
    ap.add_argument("--title", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    D, stat = build(a)
    if not D:
        sys.exit("그릴 칸이 없다")
    pm = _load(a.pred)
    name = a.name or pm.get("method", "예측")
    meta = {
        "title": a.title or f"{name} 예측, {len(D)}시즌",
        "sub": f"한섭 최종 컷을 시즌 시작 뒤 시각(T)마다 낸 값. 자료 판 {pm.get('data_ver', '?')}. 개발 검증 (시험 시즌 없음).",
        "name": name, "ref": a.ref_name,
        "src": "출처: " + " · ".join(os.path.basename(p) for p in (a.pred, a.ref, a.answer) if p) + " — tools/solo_raid/lab7/plot_pred.py",
    }
    html = (HTML.replace("__DATA__", json.dumps(D, ensure_ascii=False))
            .replace("__STAT__", json.dumps(stat)).replace("__META__", json.dumps(meta, ensure_ascii=False)))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"{a.out}: {len(D)}시즌", {k: v[:2] for k, v in stat.items()})


if __name__ == "__main__":
    main()
