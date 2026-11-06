"""WP12 곡선 꼴 조사 — 부분 결과 합치기 · 요약 · 그림 (2026-11-04).

  python3 -I tools/solo_raid/lab7/wp12_plot.py \
      --parts r7/wp12/flood_D1.json r7/wp12/cut_D1.json \
      --json r7/wp12/curve_D1.json --md r7/wp12/curve_D1.md --html $SCRATCH/wp12.html

- 합치기: 부분 파일의 A~F(헤더·plots 말고 모든 키)를 한 파일로. 같은 키·같은 그림 id가 둘 이상이면 멈춘다. header = parts 목록 + 각 부분 header + date + data_ver.
- md: 항목마다 notes·tables(마크다운 표, 40행까지)·extra(「더 본 것」). 맨 위에 규칙·자료 판·만든 명령.
- html: plots 명세를 그대로 그린다 (선 line · 계단 step · 점 points · 띠 band · 시즌 가는 선 group "season"; x.log·y.log·x.reverse). 외부 라이브러리 없음.
  HTML은 커밋하지 않는다 (AGENTS.md 「결과물」) — 세션 임시 폴더 경로를 받는다.
숫자만 옮긴다 (글 번호·닉네임 등은 부분 파일에 원래 없다).
"""
import argparse
import json
import os
import sys

MAX_HTML = 5 * 1024 * 1024
MAX_ROWS = 40
RESERVED = ("header", "plots")


def load(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def merge(paths):
    heads, items, plots, owner = [], {}, [], {}
    for p in paths:
        d = load(p)
        name = os.path.basename(p)
        heads.append({"file": name, "header": d.get("header", {})})
        for k, v in d.items():
            if k in RESERVED:
                continue
            if k in items:
                sys.exit(f"같은 키 {k!r}가 {owner[k]}와 {name} 둘에 있다 — 멈춘다")
            items[k] = v
            owner[k] = name
        for pl in d.get("plots", []):
            if any(q.get("id") == pl.get("id") for q in plots):
                sys.exit(f"같은 그림 id {pl.get('id')!r}가 둘 이상 있다 ({name}) — 멈춘다")
            plots.append(pl)
    vers = sorted({h["header"].get("data_ver") for h in heads if h["header"].get("data_ver")})
    dates = sorted({h["header"].get("date") for h in heads if h["header"].get("date")})
    header = {"about": "WP12 곡선 꼴 조사 — 서술 (부분 결과 합침)", "data_ver": "+".join(vers) if vers else None,
              "date": dates[-1] if dates else None, "parts": heads}
    out = {"header": header}
    for k in sorted(items):
        out[k] = items[k]
    out["plots"] = plots
    return out


def num(x):
    if isinstance(x, bool) or x is None:
        return "" if x is None else str(x)
    if isinstance(x, int):
        return str(x)
    if isinstance(x, float):
        if x != x:
            return "nan"
        if abs(x) >= 100:
            return f"{x:.0f}"
        return f"{x:.3g}"
    if isinstance(x, (list, dict)):
        return json.dumps(x, ensure_ascii=False, separators=(",", ":"))
    return str(x)


def cell(x):
    return num(x).replace("|", "\\|").replace("\n", " ")


def table_md(rows, indent=""):
    rows = [r for r in rows if isinstance(r, dict)]
    if not rows:
        return []
    cols = []
    for r in rows:
        for k in r:
            if k not in cols:
                cols.append(k)
    out = [indent + "| " + " | ".join(cell(c) for c in cols) + " |", indent + "|" + "|".join("---" for _ in cols) + "|"]
    for r in rows[:MAX_ROWS]:
        out.append(indent + "| " + " | ".join(cell(r.get(c)) for c in cols) + " |")
    if len(rows) > MAX_ROWS:
        out.append("")
        out.append(f"{indent}… {len(rows) - MAX_ROWS}행 더 (json)")
    return out + [""]


def is_rows(v):
    return isinstance(v, list) and v and all(isinstance(r, dict) for r in v)


def value_md(name, v, level):
    h = "#" * min(level, 6)
    out = []
    if is_rows(v):
        out += [f"{h} {name}", ""] + table_md(v)
    elif isinstance(v, dict):
        out += [f"{h} {name}", ""]
        scalars = {k: x for k, x in v.items() if not isinstance(x, (dict, list))}
        for k, x in scalars.items():
            out.append(f"- {k}: {num(x)}")
        if scalars:
            out.append("")
        for k, x in v.items():
            if k not in scalars:
                out += value_md(k, x, level + 1)
    elif isinstance(v, list):
        out += [f"{h} {name}", ""]
        if all(not isinstance(x, (dict, list)) for x in v) and len(v) <= 12:
            out += [f"- {num(x)}" for x in v] + [""]
        else:
            out += ["- `" + num(v)[:300] + ("…" if len(num(v)) > 300 else "") + "`", ""]
    else:
        out += [f"- {name}: {num(v)}"]
    return out


def make_md(merged, argv):
    h = merged["header"]
    L = ["# WP12 곡선 꼴 조사 — 결과 요약", "",
         f"- 자료 판: {h.get('data_ver')} · 날짜: {h.get('date')}",
         "- 만든 명령: `" + " ".join(argv) + "`", ""]
    L += ["## 규칙 (부분 파일 header)", ""]
    for p in h["parts"]:
        ph = p["header"]
        L.append(f"**{p['file']}** — {ph.get('about', '')} (코드 {ph.get('code_head', '?')}, {num(ph.get('secs'))}초)")
        rules = ph.get("rules")
        if isinstance(rules, dict):
            for k, v in rules.items():
                L.append(f"  - {k}: {num(v)}")
        elif rules:
            L.append(f"  - {num(rules)}")
        L.append("")
    for k, v in merged.items():
        if k in RESERVED:
            continue
        L += [f"## {k}", ""]
        if not isinstance(v, dict):
            L += value_md(k, v, 3)
            continue
        notes = v.get("notes") or []
        if notes:
            L += ["### 관찰", ""] + [f"- {n}" for n in notes] + [""]
        for tn, rows in (v.get("tables") or {}).items():
            L += [f"### 표 {tn}", ""] + (table_md(rows) if is_rows(rows) else ["(비어 있음)", ""])
        ex = v.get("extra")
        if ex:
            L += ["### 더 본 것", ""]
            for ek, ev in ex.items():
                L += value_md(ek, ev, 4)
        for ok, ov in v.items():
            if ok not in ("notes", "tables", "extra"):
                L += value_md(ok, ov, 3)
    return "\n".join(L).rstrip() + "\n"


HTML = r"""<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>WP12 곡선 꼴</title>
<style>
:root{color-scheme:light;--bg:#fcfcfb;--t1:#0b0b0b;--t2:#52514e;--t3:#8a8984;--grid:#e6e5e1;--s1:#2a78d6;--s2:#eb6834;--s3:#1f9d63;--s4:#8a5cd6;--sg:#8a8984;--btn:#efeeea}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){color-scheme:dark;--bg:#1a1a19;--t1:#f0efec;--t2:#c3c2b7;--t3:#8f8e86;--grid:#33332f;--s1:#3987e5;--s2:#d95926;--s3:#3cc383;--s4:#a98af0;--sg:#8f8e86;--btn:#2a2a28}}
:root[data-theme="dark"]{color-scheme:dark;--bg:#1a1a19;--t1:#f0efec;--t2:#c3c2b7;--t3:#8f8e86;--grid:#33332f;--s1:#3987e5;--s2:#d95926;--s3:#3cc383;--s4:#a98af0;--sg:#8f8e86;--btn:#2a2a28}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--t1);font:14px/1.5 system-ui,-apple-system,"Apple SD Gothic Neo","Noto Sans KR",sans-serif}
main{max-width:760px;margin:0 auto;padding:16px}
h1{font-size:18px;margin:0 0 4px}p.sub{color:var(--t2);margin:0 0 10px;font-size:13px}
.row{display:flex;flex-wrap:wrap;gap:6px;margin:8px 0}
button{border:0;background:var(--btn);color:var(--t1);padding:6px 12px;border-radius:999px;font:inherit;font-size:13px;cursor:pointer}
button.on{background:var(--t1);color:var(--bg)}
.card{margin:22px 0 6px}.card h2{font-size:14px;margin:0;font-weight:600;overflow-wrap:anywhere}.card .note{font-size:12px;color:var(--t2);overflow-wrap:anywhere}
.legend{display:flex;flex-wrap:wrap;gap:4px 14px;font-size:12px;color:var(--t2);margin:6px 0 2px}
.legend span{display:inline-flex;align-items:center;gap:5px;max-width:100%}
.legend svg{width:22px;height:10px;flex:none}
svg.pl{width:100%;height:auto;display:block;touch-action:pan-y;user-select:none}svg text{fill:var(--t3);font-size:11px}
.tip{position:fixed;pointer-events:none;background:var(--bg);color:var(--t1);border:1px solid var(--grid);border-radius:6px;padding:6px 8px;font-size:12px;box-shadow:0 2px 8px rgba(0,0,0,.15);display:none;z-index:9;max-width:260px}
details{margin-top:6px;font-size:13px;color:var(--t2)}details ul{margin:6px 0;padding-left:18px}li{overflow-wrap:anywhere}
.box{background:var(--btn);border-radius:8px;padding:10px 12px;font-size:13px;color:var(--t2);margin:10px 0}
</style></head><body><main>
<h1>WP12 곡선 꼴</h1><p class="sub" id="sub"></p>
<div class="box">서술 그림이다 — 예측·채점이 아니다. 점·선을 누르면(마우스는 올리면) 가장 가까운 점의 숫자가 나온다. 옅은 회색 가는 선은 시즌별 선.</div>
<div class="row" id="parts"></div><div id="cards"></div>
<p class="sub" style="margin-top:18px" id="src"></p>
</main><div class="tip" id="tip"></div>
<script>
const PLOTS=__PLOTS__,NOTES=__NOTES__,META=__META__;
const $=id=>document.getElementById(id),esc=s=>String(s==null?'':s).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const COL={data:'var(--s1)',F:'var(--s2)',M3:'var(--s3)',act:'var(--s4)',season:'var(--sg)'};
const GN={data:'자료',F:'F',M3:'M3',act:'act',season:'시즌별'};
const DASH=['','6 3','2 3','8 3 2 3'];
const parts=[...new Set(PLOTS.map(p=>p.part))].sort();let cur='all';
$('sub').textContent=META.sub;$('src').textContent=META.src;
function fmt(v){if(v==null||!isFinite(v))return '';const a=Math.abs(v);if(a>=1e7)return +(v/1e6).toPrecision(4)+'M';if(a>=1e4)return +(v/1e3).toPrecision(4)+'k';if(a>=100)return String(Math.round(v));if(a===0)return '0';return String(+v.toPrecision(3));}
function nice(lo,hi){const rs=(hi-lo)/4||1,mg=Math.pow(10,Math.floor(Math.log10(rs))),st=[1,2,2.5,5,10].map(x=>x*mg).find(x=>x>=rs),o=[];for(let v=Math.ceil(lo/st-1e-9)*st;v<=hi+1e-9;v+=st)o.push(+v.toPrecision(10));return o;}
function logticks(lo,hi){const o=[],a=Math.floor(lo),b=Math.ceil(hi),wide=(hi-lo)>2.2;for(let e=a;e<=b;e++){for(const m of(wide?[1]:[1,2,5])){const v=Math.log10(m)+e;if(v>=lo-1e-9&&v<=hi+1e-9)o.push([v,m*Math.pow(10,e)]);}}return o.length>=2?o:nice(lo,hi).map(v=>[v,Math.pow(10,v)]);}
const Wd=()=>Math.max(288,Math.min(728,document.querySelector('main').clientWidth));
function legendSvg(s,i){const c=COL[s.group]||COL.data,d=DASH[i%DASH.length];
 if(s.kind=='band')return `<svg viewBox="0 0 22 10"><rect x="0" y="1" width="22" height="8" rx="2" style="fill:${c};fill-opacity:.25"/></svg>`;
 if(s.kind=='points')return `<svg viewBox="0 0 22 10"><circle cx="11" cy="5" r="3.5" style="fill:${c}"/></svg>`;
 return `<svg viewBox="0 0 22 10"><line x1="0" x2="22" y1="5" y2="5" style="stroke:${c};stroke-width:2.2" stroke-dasharray="${d}"/></svg>`;}
function card(p){
 const X=p.x||{},Y=p.y||{},ser=p.series||[];
 const tx=v=>X.log?(v>0?Math.log10(v):null):v,ty=v=>Y.log?(v>0?Math.log10(v):null):v;
 let xl=1e300,xh=-1e300,yl=1e300,yh=-1e300;
 const ok=(a,b)=>a!=null&&b!=null&&isFinite(a)&&isFinite(b);
 for(const s of ser){const xs=s.x||[];
  (s.y||[]).forEach((y,i)=>{const a=tx(xs[i]),b=ty(y);if(ok(a,b)){xl=Math.min(xl,a);xh=Math.max(xh,a);yl=Math.min(yl,b);yh=Math.max(yh,b);}});
  if(s.kind=='step')xs.forEach(x=>{const a=tx(x);if(a!=null&&isFinite(a)){xl=Math.min(xl,a);xh=Math.max(xh,a);}});
  for(const k of['lo','hi'])(s[k]||[]).forEach((y,i)=>{const a=tx(xs[i]),b=ty(y);if(ok(a,b)){yl=Math.min(yl,b);yh=Math.max(yh,b);}});}
 const W=Wd(),H=W<480?250:290,ml=W<480?44:50,mr=12,mt=22,mb=40;
 let html=`<div class="card" data-part="${esc(p.part)}"><h2>${esc(p.id)} · ${esc(p.title)}</h2>`;
 if(xl>xh||yl>yh)return html+`<div class="note">그릴 점이 없다 (명세에 숫자가 없거나 로그 축에 0 이하만 있다)</div></div>`;
 if(xl==xh){xl-=.5;xh+=.5}if(yl==yh){yl-=.5;yh+=.5}
 const px=(xh-xl)*.03,py=(yh-yl)*.06;if(!X.log||true){xl-=px;xh+=px;}yl-=py;yh+=py;
 const iw=W-ml-mr,ih=H-mt-mb;
 const SX=a=>ml+((X.reverse?(xh-a):(a-xl))/(xh-xl))*iw,SY=b=>mt+(1-(b-yl)/(yh-yl))*ih;
 let g='';
 for(const t of (Y.log?logticks(yl,yh):nice(yl,yh).map(v=>[v,v]))){if(t[0]<yl||t[0]>yh)continue;g+=`<line x1="${ml}" x2="${W-mr}" y1="${SY(t[0])}" y2="${SY(t[0])}" stroke="var(--grid)"/><text x="${ml-5}" y="${SY(t[0])+4}" text-anchor="end">${fmt(t[1])}</text>`;}
 for(const t of (X.log?logticks(xl,xh):nice(xl,xh).map(v=>[v,v]))){if(t[0]<xl||t[0]>xh)continue;g+=`<line x1="${SX(t[0])}" x2="${SX(t[0])}" y1="${mt}" y2="${H-mb}" stroke="var(--grid)"/><text x="${SX(t[0])}" y="${H-mb+14}" text-anchor="middle">${fmt(t[1])}</text>`;}
 g+=`<text x="${ml}" y="12" text-anchor="start" style="font-size:11px">${esc(Y.label||'')}${Y.log&&!/로그/.test(Y.label||'')?' (로그)':''}</text><text x="${W-mr}" y="${H-6}" text-anchor="end">${esc(X.label||'')}${X.log&&!/로그/.test(X.label||'')?' (로그)':''}${X.reverse&&!/거꾸로/.test(X.label||'')?' (거꾸로)':''}</text>`;
 const hits=[];const order={band:0,line:1,step:2,points:3};
 const idx={};const si=ser.map(s=>{idx[s.group]=(idx[s.group]||0);return s.kind=='band'?0:idx[s.group]++;});
 const sorted=ser.map((s,i)=>[s,i]).sort((a,b)=>(a[0].group=='season'?-1:0)-(b[0].group=='season'?-1:0)||order[a[0].kind]-order[b[0].kind]);
 for(const [s,i] of sorted){const c=COL[s.group]||COL.data,xs=s.x||[],ys=s.y||[],nm=s.name||'';
  const ssn=s.group=='season';
  if(s.kind=='band'){let top=[],bot=[];ys.forEach((_,k)=>{const a=tx(xs[k]),lo=ty((s.lo||[])[k]),hi=ty((s.hi||[])[k]);if(ok(a,lo)&&ok(a,hi)){top.push(SX(a)+','+SY(hi));bot.push(SX(a)+','+SY(lo));hits.push([SX(a),SY((lo+hi)/2),nm,xs[k],`${fmt((s.lo||[])[k])} ~ ${fmt((s.hi||[])[k])}`]);}});
   if(top.length>1)g+=`<path d="M${top.join('L')}L${bot.reverse().join('L')}Z" style="fill:${c};fill-opacity:.2"/>`;continue;}
  if(s.kind=='points'){const sh=si[i]%3;ys.forEach((y,k)=>{const a=tx(xs[k]),b=ty(y);if(!ok(a,b))return;const cx=SX(a),cy=SY(b);
    g+=sh==0?`<circle cx="${cx}" cy="${cy}" r="3.2" style="fill:${c};fill-opacity:.8"/>`:sh==1?`<rect x="${cx-3}" y="${cy-3}" width="6" height="6" style="fill:${c};fill-opacity:.8"/>`:`<path d="M${cx} ${cy-4}L${cx+4} ${cy}L${cx} ${cy+4}L${cx-4} ${cy}Z" style="fill:${c};fill-opacity:.8"/>`;
    hits.push([cx,cy,nm,xs[k],y]);});continue;}
  const sw=ssn?1:2.2,op=ssn?.4:1,da=ssn?'':DASH[si[i]%DASH.length];
  let d='',pen=false;
  if(s.kind=='step'){ys.forEach((y,k)=>{const a=tx(xs[k]),a2=tx(xs[k+1]),b=ty(y);if(!ok(a,b)||a2==null||!isFinite(a2)){pen=false;return;}
    d+=pen?`L${SX(a)},${SY(b)}`:`M${SX(a)},${SY(b)}`;d+=`L${SX(a2)},${SY(b)}`;pen=true;hits.push([(SX(a)+SX(a2))/2,SY(b),nm,`${fmt(xs[k])} ~ ${fmt(xs[k+1])}`,y]);});}
  else{ys.forEach((y,k)=>{const a=tx(xs[k]),b=ty(y);if(!ok(a,b)){pen=false;return;}d+=(pen?'L':'M')+SX(a)+','+SY(b);pen=true;hits.push([SX(a),SY(b),nm,xs[k],y]);});}
  if(d)g+=`<path d="${d}" fill="none" stroke="${c}" stroke-width="${sw}" stroke-opacity="${op}" stroke-dasharray="${da}" stroke-linejoin="round"/>`;}
 g+=`<circle class="hl" r="5" cx="0" cy="0" fill="none" stroke="var(--t1)" stroke-width="1.5" visibility="hidden"/>`;
 html+=`<svg class="pl" viewBox="0 0 ${W} ${H}" data-i="${esc(p.id)}">${g}<rect x="${ml}" y="${mt}" width="${iw}" height="${ih}" fill="transparent"/></svg>`;
 // 범례 (시즌 선은 한 줄로)
 let lg='',ns=0;ser.forEach((s,i)=>{if(s.group=='season'){ns++;return;}lg+=`<span>${legendSvg(s,si[i])}${esc(s.name)}</span>`;});
 if(ns)lg+=`<span><svg viewBox="0 0 22 10"><line x1="0" x2="22" y1="5" y2="5" style="stroke:var(--sg);stroke-width:1.2;stroke-opacity:.5"/></svg>시즌별 ${ns}개 (누르면 이름)</span>`;
 html+=`<div class="legend">${lg}</div>`;
 if(p.note)html+=`<div class="note">${esc(p.note)}</div>`;
 const nt=NOTES[p.part]||[];
 if(nt.length)html+=`<details><summary>${esc(p.part)} 부분 관찰 ${nt.length}줄</summary><ul>${nt.map(n=>`<li>${esc(n)}</li>`).join('')}</ul></details>`;
 html+='</div>';
 cardHits[p.id]={hits,X,Y};
 return html;}
const cardHits={};const tip=$('tip');
function show(pt,h,X,Y){tip.innerHTML=`<b>${esc(h[2])}</b><br>${esc(X.label||'x')} ${esc(typeof h[3]=='number'?fmt(h[3]):h[3])}<br>${esc(Y.label||'y')} ${esc(typeof h[4]=='number'?fmt(h[4]):h[4])}`;tip.style.display='block';
 tip.style.left=Math.max(8,Math.min(pt.clientX+12,innerWidth-tip.offsetWidth-8))+'px';tip.style.top=Math.max(8,pt.clientY-tip.offsetHeight-12)+'px';}
function hide(sv){tip.style.display='none';if(sv){const h=sv.querySelector('.hl');if(h)h.setAttribute('visibility','hidden');}}
function wire(){document.querySelectorAll('svg.pl').forEach(sv=>{const C=cardHits[sv.dataset.i],vb=sv.viewBox.baseVal;let tm;
 const mv=e=>{const b=sv.getBoundingClientRect(),k=vb.width/b.width,mx=(e.clientX-b.left)*k,my=(e.clientY-b.top)*k;let best=null,bd=1e9;
  for(const h of C.hits){const d=(h[0]-mx)*(h[0]-mx)+(h[1]-my)*(h[1]-my);if(d<bd){bd=d;best=h;}}
  if(!best||bd>40*40){hide(sv);return;}const hl=sv.querySelector('.hl');hl.setAttribute('cx',best[0]);hl.setAttribute('cy',best[1]);hl.setAttribute('visibility','visible');show(e,best,C.X,C.Y);clearTimeout(tm);if(e.pointerType=='touch')tm=setTimeout(()=>hide(sv),2500);};
 sv.addEventListener('pointermove',mv);sv.addEventListener('pointerdown',mv);sv.addEventListener('pointerleave',e=>{if(e.pointerType!='touch')hide(sv)});});}
function render(){$('parts').innerHTML=[['all','전체'],...parts.map(p=>[p,p])].map(([k,l])=>`<button class="${k==cur?'on':''}" data-p="${esc(k)}">${esc(l)}</button>`).join('');
 $('cards').innerHTML=PLOTS.filter(p=>cur=='all'||p.part==cur).map(card).join('')||'<p class="sub">그림이 없다</p>';wire();}
document.addEventListener('click',e=>{const b=e.target.closest('button');if(b&&b.dataset.p!=null){cur=b.dataset.p;render();}});
let lw=0;addEventListener('resize',()=>{const w=Wd();if(Math.abs(w-lw)>2){lw=w;render();}});lw=Wd();
render();
</script></body></html>
"""


def make_html(merged, argv):
    plots = merged["plots"]
    notes = {k: (v.get("notes") or []) for k, v in merged.items() if k not in RESERVED and isinstance(v, dict)}
    h = merged["header"]
    meta = {"sub": f"침수·컷 곡선의 꼴 (서술 조사). 자료 판 {h.get('data_ver')} · {h.get('date')} · 그림 {len(plots)}장.",
            "src": "출처: " + " · ".join(p["file"] for p in h["parts"]) + " — tools/solo_raid/lab7/wp12_plot.py"}

    def js(o):
        return json.dumps(o, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    return HTML.replace("__PLOTS__", js(plots)).replace("__NOTES__", js(notes)).replace("__META__", js(meta))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--parts", nargs="+", required=True)
    ap.add_argument("--json", default=None)
    ap.add_argument("--md", default=None)
    ap.add_argument("--html", default=None)
    a = ap.parse_args()
    merged = merge(a.parts)
    argv = ["python3", "-I", "tools/solo_raid/lab7/wp12_plot.py"] + sys.argv[1:]
    for path, fn in ((a.json, lambda: json.dumps(merged, ensure_ascii=False, indent=1) + "\n"),
                     (a.md, lambda: make_md(merged, argv)), (a.html, lambda: make_html(merged, argv))):
        if not path:
            continue
        text = fn()
        if path == a.html and len(text.encode("utf-8")) > MAX_HTML:
            sys.exit(f"HTML이 5MB를 넘는다 ({len(text) // 1024}KB) — plots를 줄여야 한다")
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"{path}: {len(text.encode('utf-8')) // 1024}KB")
    print(f"항목 {[k for k in merged if k not in RESERVED]} · 그림 {len(merged['plots'])}장")


if __name__ == "__main__":
    main()
