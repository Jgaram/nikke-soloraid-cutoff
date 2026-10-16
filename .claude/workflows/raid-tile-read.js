export const meta = {
  name: 'raid-tile-read',
  description: '솔로레이드 판독 시트를 배치로 나눠 raid-vision(기본 Haiku) 에이전트로 읽고 배치별 JSONL을 쓴다',
  whenToUse: '솔로레이드 2단계 타일 판독. tile_sheets.py로 시트를 만든 뒤, Haiku 기반이라 상시 허락 (AGENTS.md, 2026-10-15)',
  phases: [{ title: '판독', detail: '시트 약 19장마다 raid-vision 에이전트 하나', model: 'haiku' }],
}

// 세션 D(2026-10-15)에 쓴 판독 Workflow를 저장한 것. 자세한 배경은
// experiments/2026-10-15-solo-raid-low-cost/README.md 「결과 — 세션 D」.
//
// args (모든 경로는 절대 경로)
//   dir     시트 폴더 (tile_sheets.py의 OUTDIR, sheets.jsonl과 *.png가 있는 곳)
//   strip   띠 시트 수. tile_sheets.py 출력의 sheets.jsonl에서 kind=strip 줄 수
//   whole   통째 시트 수 (TILE_FULL=1이면 0)
//   out     결과 폴더. 배치마다 reads_NN.jsonl을 쓴다
//   prompt  지시문 파일 (보통 tools/tile_prompt.md, 원본 판독은 data/stage2d/orig_prompt.md 같은 것)
//   batch   배치당 시트 수 (기본 19)
//   model   기본 'haiku'. 모델을 비교할 때만 'sonnet' 등으로 바꾼다
//
// 시트 이름은 tile_sheets.py 규칙을 따른다: 번호는 종류를 가로질러 1부터 이어진다
// (strip_01 … strip_{strip}, whole_{strip+1} …).
// 끝나면 조율자는 결과를 열지 말고 tools/tile_check.py로 칸 수만 센다.

const a = args || {}
for (const k of ['dir', 'out', 'prompt']) {
  if (!a[k]) throw new Error(`args.${k}가 없다`)
}
const nStrip = a.strip || 0
const nWhole = a.whole || 0
const per = a.batch || 19
const model = a.model || 'haiku'
const pad = n => (n < 10 ? '0' + n : String(n))

const sheets = []
for (let i = 1; i <= nStrip; i++) sheets.push(`${a.dir}/strip_${pad(i)}.png`)
for (let i = nStrip + 1; i <= nStrip + nWhole; i++) sheets.push(`${a.dir}/whole_${pad(i)}.png`)
if (!sheets.length) throw new Error('시트가 0장이다 (args.strip·args.whole)')

const batches = []
for (let i = 0; i < sheets.length; i += per) batches.push(sheets.slice(i, i + per))
log(`시트 ${sheets.length}장 → ${batches.length}배치, 모델 ${model}`)

phase('판독')
const replies = await parallel(batches.map((b, i) => () => {
  const out = `${a.out}/reads_${pad(i + 1)}.jsonl`
  return agent(
    `지시문 파일: ${a.prompt}\n` +
    `이 파일의 \`---\` 아래를 그대로 따른다. {SHEETS}와 {OUT}은 다음 값이다.\n` +
    `{SHEETS}: ${b.join(', ')}\n` +
    `{OUT}: ${out}`,
    { label: `판독 ${pad(i + 1)} (${b.length}장)`, phase: '판독', agentType: 'raid-vision', model },
  ).then(r => ({ batch: i + 1, out, sheets: b.length, reply: r }))
}))

const failed = replies.map((r, i) => (r ? null : i + 1)).filter(Boolean)
if (failed.length) log(`응답 없는 배치: ${failed.join(', ')} — 그 배치만 다시 돌린다`)
return { model, batches: batches.length, failed, replies: replies.filter(Boolean) }
