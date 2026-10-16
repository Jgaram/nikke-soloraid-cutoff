export const meta = {
  name: 'raid-window',
  description: '솔로레이드 기간 실행 한 번의 Haiku 단계: 판독 시트·글 숫자 분류·시즌 문맥을 raid-vision 에이전트로 한꺼번에 돌린다',
  whenToUse: '솔로레이드 4단계. tools/run_window.sh가 DONE을 쓴 뒤, Haiku 기반이라 상시 허락 (AGENTS.md, 2026-10-15)',
  phases: [
    { title: '판독', detail: '시트 약 19장마다 raid-vision 에이전트 하나', model: 'haiku' },
    { title: '글 분류', detail: '추출 줄 200줄마다 하나', model: 'haiku' },
    { title: '시즌 문맥', detail: '글 150개마다 하나', model: 'haiku' },
  ],
}

// 세션 F1(2026-10-15)에 만든 기간 실행용 Workflow. 배경은
// experiments/2026-10-15-solo-raid-low-cost/README.md 「세션 F1 — 4단계 도구」·「결과 — 세션 F1」.
//
// args (경로는 절대 경로)
//   run          run_window.sh의 실행 폴더 (RUN). DONE 줄의 run 값
//   sheets       시트 수 (DONE의 sheets). 시트는 RUN/sheets/strip_01.png …
//   text         글 분류 배치 수 (DONE의 text_batches). 입력 RUN/text/in_NN.jsonl
//   ctx          시즌 문맥 배치 수 (DONE의 ctx_batches). 입력 RUN/ctx/in_NN.jsonl
//   tile_prompt  판독 지시문 (tools/tile_prompt.md)
//   batch        배치당 시트 수 (기본 19)
//   model        기본 'haiku'
//
// 결과: RUN/reads/reads_NN.jsonl, RUN/text/out_NN.jsonl, RUN/ctx/out_NN.jsonl.
// 끝나면 조율자는 결과를 열지 말고 tools/merge_obs.py STATE RUN으로 합친다 (빠진 칸·줄을 센다).

const a = args || {}
for (const k of ['run', 'tile_prompt']) {
  if (!a[k]) throw new Error(`args.${k}가 없다`)
}
const per = a.batch || 19
const model = a.model || 'haiku'
const pad = n => (n < 10 ? '0' + n : String(n))

const sheets = []
for (let i = 1; i <= (a.sheets || 0); i++) sheets.push(`${a.run}/sheets/strip_${pad(i)}.png`)
const tiles = []
for (let i = 0; i < sheets.length; i += per) tiles.push(sheets.slice(i, i + per))
log(`시트 ${sheets.length}장 → ${tiles.length}배치, 글 분류 ${a.text || 0}배치, 시즌 문맥 ${a.ctx || 0}배치, 모델 ${model}`)

const jobs = []
tiles.forEach((b, i) => jobs.push({
  phase: '판독', label: `판독 ${pad(i + 1)} (${b.length}장)`, out: `${a.run}/reads/reads_${pad(i + 1)}.jsonl`,
  prompt: `지시문 파일: ${a.tile_prompt}\n이 파일의 \`---\` 아래를 그대로 따른다. {SHEETS}와 {OUT}은 다음 값이다.\n` +
    `{SHEETS}: ${b.join(', ')}\n{OUT}: ${a.run}/reads/reads_${pad(i + 1)}.jsonl`,
}))
for (const [key, n, ph, file] of [['text', a.text || 0, '글 분류', 'text_prompt.md'], ['ctx', a.ctx || 0, '시즌 문맥', 'ctx_prompt.md']]) {
  for (let i = 1; i <= n; i++) {
    jobs.push({
      phase: ph, label: `${ph} ${pad(i)}`, out: `${a.run}/${key}/out_${pad(i)}.jsonl`,
      prompt: `지시문 파일: ${a.run}/${file}\n이 파일의 \`---\` 아래를 그대로 따른다. {IN}과 {OUT}은 다음 값이다.\n` +
        `{IN}: ${a.run}/${key}/in_${pad(i)}.jsonl\n{OUT}: ${a.run}/${key}/out_${pad(i)}.jsonl`,
    })
  }
}
if (!jobs.length) throw new Error('할 일이 없다 (args.sheets·text·ctx)')

// 동시 실행 상한은 Workflow마다 min(16, CPU-2)다 (4코어 컨테이너면 2). 같은 args에 parts·part를 주어
// Workflow 여러 개를 동시에 띄우면 일감을 나눠 맡는다 (세션 F2, 2026-10-15).
// skip: 이미 끝난 결과 파일 (out 경로, 또는 run 아래 상대 경로 "reads/reads_01.jsonl")
const skip = new Set(a.skip || [])
const parts = a.parts || 1
const mine = jobs.filter(j => !skip.has(j.out) && !skip.has(j.out.slice(a.run.length + 1)))
  .filter((j, i) => i % parts === (a.part || 0))
if (parts > 1 || skip.size) log(`나눠 맡기 ${(a.part || 0) + 1}/${parts}, 건너뜀 ${skip.size} → ${mine.length}개`)
jobs.length = 0
jobs.push(...mine)

const replies = await parallel(jobs.map(j => () =>
  agent(j.prompt, { label: j.label, phase: j.phase, agentType: 'raid-vision', model })
    .then(r => ({ out: j.out, reply: r }))))

const failed = jobs.filter((j, i) => !replies[i] || !replies[i].reply).map(j => j.out)
if (failed.length) log(`응답 없는 배치 ${failed.length}개 — merge_obs.py가 빠진 것을 센다. 그 배치만 다시 돌린다`)
return { model, jobs: jobs.length, failed, replies: replies.filter(Boolean) }
