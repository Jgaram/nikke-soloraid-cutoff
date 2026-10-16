# 솔로레이드 진행 중 시즌 (6시간 자동 수집·예측·웹 → 끝나면 보강)

2026-10-16 사용자와 정한 운영 방식. S42부터 **Claude 예약 작업**이 6시간마다 새 세션(Sonnet)을 하나 띄워 받기 → 판독 → 예측 → 커밋·push를 하고,
push가 공개 웹(<https://jgaram.github.io/nikke-soloraid-cutoff/>)을 바로 갱신한다 (docs/publish.md).
처음에는 세션 하나가 `send_later`로 정해진 시점(T = 12·18·24·36·48·72·96)마다 깨는 방식이었다가 같은 날 바꿨다 (사용자 결정).

- **진행 중:** 목적은 진행 중 레이드 예측이다. 6시간 칸마다 수집 → 예측 → 커밋 → 웹.
- **끝난 뒤:** 마지막 수집(종료 후 인증)과 후처리까지는 자동이다. 데이터 점검·예외 처리·분석 갱신·자기 예측 채점은 사람이 시킨다 (아래 「3. 끝난 뒤」).

수집·판독 도구는 [일괄 수집 지시문](solo-raid-backfill.md)의 「시즌 세션 지시문」과 같은 것(`run_window.sh` → `raid-window` Workflow → `merge_obs.py`)을
`tools/solo_raid/live_tick.sh`가 묶어 부른다. 이 문서는 일정·켜기·자동 실행 한 번·끝난 뒤 순서를 정한다.

## 정한 것 (사용자 결정, 2026-10-16)

- **실행 위치:** Claude 예약 작업 하나 (「솔로레이드 6시간 자동 수집」, 실행마다 새 세션, 이 환경). GitHub Actions에서는 Haiku 판독을 돌릴 수 없어서다.
  조율 세션 모델은 **Sonnet** (판독은 `raid-vision` Haiku — Haiku 기반 Workflow라 상시 허락, AGENTS.md).
- **칸:** 챌린지 시작(`c_start`, 05:00)부터 6시간마다 — 한국 시간 05·11·17·23시가 칸 경계다.
  구간은 **게시 시각으로 경계에서 자른다** (`[지난 성공 끝, 경계)`). 그래서 경계 뒤에 올라온 인증은 늦게 깨어도 이번 칸에 들지 않고 다음 칸으로 간다.
  실행을 경계 15분 뒤로 미루는 안은 자료는 그대로이고 웹만 늦어서 버렸다 (사용자 지적). 예약 작업은 **경계 1분 뒤(05:01·11:01·17:01·23:01)** 에 깬다
  (정각은 예약 작업이 붐비고, 몇 분 일찍 깨도 `live_tick.py`가 같은 칸을 잡고 경계까지 기다린다).
- **예측 시점:** T = 6, 12, 18, …, T_end (종료를 덮는 첫 경계, 대개 120). **매 칸 예측을 내고 웹에 올린다** (T=6부터, 사용자 결정).
  T=6은 % 변환 외삽이라 등급 C 이하로 나온다 (S41 사본으로 확인, 2026-10-16). 웹은 C·D면 가운데 숫자를 흐리게 보인다.
  채점 기준 T(`data/solo-raid/analysis/criteria.json`의 12·18·24·36·48·72·96)는 6시간 칸에 모두 들어 있어 `criteria.json`은 바꾸지 않는다.
  `score_live.py`·`site.py`는 `predict/T*/`를 모두 읽는다 (다른 T도 그대로).
- **예측 기록:** `data/solo-raid/S{n}/predict/T{T}/`(predict_run.sh 출력 그대로)를 커밋한다. 끝난 뒤 정직하게 채점하려는 것이다 (AGENTS.md 「결과물」 예외).
- **기본 도구:** 그때의 `predict_run.sh` 기본(prior·서버 미상 처리)을 그대로 쓴다. 진행 중에 결과를 보고 도구나 설정을 바꾸지 않는다.
  기본을 바꾸는 판정은 분석 문서의 기준(개발 시즌 8개 이상, `criteria.json`)으로만 한다.
- **시작은 사람, 끝은 자동.** 시즌마다 사용자가 「S{n} 자동 수집 켜」라고 한다 (아래 「1. 켜기」). 종료 뒤에는 마지막 수집 한 번으로 스스로 꺼진다.
- **종료 후 수집 끝 `e_end` = 종료 + 4일 05:00** (S42부터, 사용자 결정). 종료가 목요일이면 월요일 05:00이다.
  토요일이 보상 들어오는 날이라 그날과 주말에도 최종 인증이 많이 올라온다 (사용자 설명). S33~S40은 종료 + 2일 05:00(대개 토요일 05:00, 보상 직전)까지만 받았다.
  더 길게 받은 S41에서 종료 뒤 % 관측을 12시간 단위로 세면 0–12h 133 · 12–24h 16 · 24–36h 0 · 36–48h 4 · **48–60h(토 05:00~) 34** · 60–72h 13 · 72–96h 10이었다
  (토요일 05:04부터 종료 화면 인증이 다시 몰림, 종료 뒤 관측의 약 27%가 토요일 05:00 뒤).
- **종료 후 수집은 e_end에 한 번** (사용자 결정). 예측·웹에는 영향이 없고 채점·분석용 정답 자료다.
- **시험 시즌이어도 예측은 낸다** (`split.py role S{n}` — S42·S43은 개발, S44·S46·S48 시험). 다만 끝난 뒤 채점·분석은 하지 않는다.
- **예외 처리:** [예외 처리 규칙](solo-raid-backfill.md#예외-처리-규칙)대로 끝난 뒤 점검에서 `S{n}/discard.json`에 남긴다. 자동 실행은 예외를 판단하지 않는다.
- 사이트·UA·작업자·간격은 일괄 수집과 같다 (디시·네이버 게임 라운지·블라블라링크, 자기 이름 UA, 작업자 4·간격 0.5초).

## 일정 예 (S42, KST)

| | 시각 |
|---|---|
| 이벤트 오픈 | 10-15(목) 12:00 (공지 2026-10-05 「10월 8일 업데이트 공지」) |
| 챌린지 시작 `c_start` | 10-17(토) 05:00 (3일차 05:00) |
| 첫 예측 T=6 | 10-17 11:01 실행 → 웹 11:30~11:50쯤 |
| 마지막 예측 T=120 | 10-22(목) 05:01 실행 (종료 04:59:59를 덮는 칸) |
| 종료 후 수집 | 10-26(월) 05:01 실행 (`e_end` 05:00) → 예약 작업이 저절로 꺼짐 |

예측 20번 + 종료 후 수집 1번 = 세션 21개(+ T=0 칸 05:01의 빈 실행 1개). 지난 시즌 2일 구간 기록(새 글 1,100~3,100개, 시트 300~950장)으로 어림하면
6시간 칸 하나는 새 글 150~400개·시트 40~120장, 판독 에이전트 3~10개, 20~40분이다.

## 1. 켜기 (시즌 시작 — 사용자가 「S{n} 자동 수집 켜」라고 할 때)

1. 공식 공지 아카이브 `notices/`에서 이번 솔로 레이드 공지(보스·기간)를 찾는다. 없거나 보스 이름이 없으면 사용자에게 묻는다.
2. `data/solo-raid/S{n}/state/season.json`:
   `name`·`boss`·`start`·`end`, `c_start`(기본 3일차 05:00 — 공지가 다르게 말하면 그것, 근거를 README에), `e_start` = `c_start`, `e_end` = 종료 + 4일 05:00.
3. `data/solo-raid/S{n}/README.md`를 만든다 (일괄 수집 「시즌 요약」 형식의 머리 + 몫 `python3 -I tools/solo_raid/split.py role S{n}`).
   **맨 끝 절은 「자동 실행」 표로 둔다** — 실행마다 `live_tick.py record`가 README 맨 끝에 줄을 덧붙인다.
4. 예약 작업을 켠다.
   - 있으면(`data/solo-raid/live.json`의 `trigger`, 없으면 `list_triggers`에서 이름 「솔로레이드 6시간 자동 수집」): `update_trigger`로
     `cron_expression` = `CRON_TZ=Asia/Seoul 1 5,11,17,23 * * *`, `enabled` = true (지난 시즌 끝에 한 번짜리로 바뀌어 꺼져 있다).
   - 없으면 `create_trigger`: 이름 「솔로레이드 6시간 자동 수집」, 위 cron, `create_new_session_on_fire` = true, `notifications` = `{"push": true}`,
     `initiation` = human_request, `prompt`는 아래 「예약 작업 프롬프트」 그대로. 만든 뒤 `update_trigger`로 `model` = Sonnet(`claude-sonnet-5-5`).
   - 응답의 `next_run_at`이 다음 칸 1분 뒤인지 확인한다 (분이 바뀌어 있으면 고친다).
5. `data/solo-raid/live.json`: `{"season": "S{n}", "trigger": "trig_…", "cron": "…", "on": "켠 KST 시각", "by": "세션 id"}`.
6. 커밋(`S{n} 자동 수집 켬`) → push. `season.json`이 올라가 웹에 「챌린지 진행 중 · 첫 예측은 6시간 뒤」가 뜬다 (시작 뒤부터).
7. 사용자에게 첫 실행 시각(KST)을 알린다.

### 예약 작업 프롬프트

```
솔로레이드 자동 수집 한 번. docs/solo-raid-live.md의 「2. 자동 실행 한 번」을 처음부터 끝까지 그대로 따른다.
사람이 보고 있지 않으니 묻지 말고, 그 절이 멈추라고 하는 곳에서만 멈춘다. 마지막 답은 그 절의 형식대로 한국어로 쓴다.
```

절차는 이 문서에 두므로, 절차를 바꿀 때 예약 작업은 고치지 않는다.

## 2. 자동 실행 한 번 (예약 작업이 띄운 세션이 따른다)

규칙: 사람이 보고 있지 않다 — 묻지 않는다. `tools/`·`.claude/`·문서를 고치지 않는다. Workflow 결과 파일을 열어 해석하지 않고 숫자를 고치지 않는다.
커밋·push는 `live_tick.sh`가 한다 (main에 바로). 시각은 KST로 쓴다.

1. **판정:** `bash tools/solo_raid/live_tick.sh plan` → JSON 한 줄 (`action`·`T`·`start`·`end_w`·`after`·`trigger`·`reason`).
   - `idle` → 6으로 (보통 할 일 없음. `after`가 있으면 예약 작업만 손본다).
   - `stop` → `update_trigger(enabled=false)` 뒤 7로 (「멈춤: {reason}」).
   - `run`(예측 칸) · `final`(종료 후 수집) → 2로.
2. **받기:** `bash tools/solo_raid/live_tick.sh collect`를 **Bash `run_in_background`(timeout 7200000)** 로 띄우고 완료 알림을 기다린다 (대기 루프를 짜지 않는다).
   칸 경계 전이면 스스로 기다린다. 실패하면 같은 인자로 한 번 더 돈다.
   - 종료 0 → 출력 끝 줄 JSON(`WORK/workflow.json`과 같음)으로 3. `needed`가 false면 4·5를 건너뛰고 5의 `finish`로.
   - 종료 4 (403·429) → `bash tools/solo_raid/live_tick.sh fail collect "403·429"` → `update_trigger(enabled=false)` → 7 (「멈춤: 403·429 — 이 시즌 받기를 멈췄다」).
     다시 켜는 것은 사용자가 정한다.
   - 그 밖 → `bash tools/solo_raid/live_tick.sh fail collect` → 7 (다음 칸이 이어 받는다).
3. **판독:** Workflow를 **4개 한 번에** 띄운다 — `Workflow(scriptPath=".claude/workflows/raid-window.js", args={workflow.json의 args, part: 0})` … `part: 3`.
   (`args`에 `parts: 4`가 들어 있다.)
4. **합치기:** `bash tools/solo_raid/live_tick.sh merge`.
   - 종료 0 → 5.
   - 종료 3 (빠진 배치) → 출력 끝 줄 JSON 목록을 `args.skip`으로 더해 3을 **한 번만** 다시 → `merge` 다시. 또 3이면 그대로 5로 (빠진 칸은 merge.json에 남는다).
   - 종료 5 (final만, 원본 대기 사진) → `bash tools/solo_raid/live_tick.sh collect --orig`(2처럼 background) → 3 → 4 (한 번만).
   - 그 밖 → `bash tools/solo_raid/live_tick.sh fail merge` → 7.
5. **마무리:** `bash tools/solo_raid/live_tick.sh finish`.
   - `run`: 내보내기·커밋(`S{n} 받기 T{T}`)·push → 예측(`predict_run.sh`, 몇 초~1분) → `predict/T{T}/`·후처리 파일·README 줄 커밋(`S{n} 예측 T{T}`)·push.
     push가 `publish.yml`을 돌려 웹이 몇 분 안에 바뀐다. 출력 끝에 `[요약]` 줄과 경고(`!!`)가 나온다.
   - `final`: 내보내기·커밋·push → `post_all.sh` → `live/final_done.json`, `live.json`의 `season`을 비움 → 커밋(`S{n} 종료 후 수집 끝`)·push.
   - 종료 1 (예측 실패) → 이미 기록·커밋됐다. 7로.
6. **예약 작업 손보기** (1의 `after`):
   - `set_once {시각}` → `update_trigger(trigger_id={trigger}, run_once_at={시각})`. 반복 일정이 e_end 한 번짜리로 바뀐다 (T_end 칸, 또는 종료 뒤 `idle`인데 아직 반복일 때).
     `get_trigger`로 이미 그 시각 한 번짜리면 건너뛴다.
   - `finish` → 할 것 없음 (한 번짜리는 실행 뒤 저절로 꺼진다).
   - 실패하면 7의 답에 「예약 작업을 못 바꿨다 — 사람이 바꿔야 함」을 적는다.
7. **마지막 답** (세션 기록과 push 알림에 남는다. 해석을 덧붙여 확신을 키우지 않는다):
   - `run`: 첫 줄 `S{n} T{T} 예측 — {칸 시각} 칸, {올린 시각} 올림`. 다음에 `finish` 출력의 `[요약]` 줄과 `!!` 경고를 그대로, 받은 글·시트·요청 수를 한 줄로.
     맨 끝에 「검증 안 된 예측」.
   - `final`: 받은 글·관측 수 한 줄과 「남은 일: 데이터 점검·수집 끝 표시·채점 (docs/solo-raid-live.md 「3. 끝난 뒤」)」.
   - `idle`: `reason` 한 줄 (예약 작업을 손봤으면 그것도).
   - 실패·멈춤: 어느 단계에서 왜(`run.log`·FAILED 끝 몇 줄 요약), 다음 칸이 이어 받는지.

### 놓친 칸·겹침

- 구간은 「지난 성공 끝」(`state/runs.jsonl`)부터라 실행 하나가 실패해도 자료는 빠지지 않는다. 다음 칸이 그만큼 길게 받는다.
- 놓친 T의 예측은 나중에 메우지 않는다 (그 뒤 글이 섞인다). README 「자동 실행」 표에 실패로 남는다.
- 같은 칸을 두 번 하지 않는다: `plan`은 `predict/T{T}/`가 있으면 `idle`, `finish`는 main에 그 예측이 먼저 올라왔으면 예측을 건너뛴다.
- 리허설: `LIVE_DATA`(사본 `data/solo-raid`)·`LIVE_NOW`(가짜 지금)·`LIVE_DRY=1`(커밋·push 안 함)·`LIVE_WORK`(작업 폴더)로 레포를 건드리지 않고 돌린다.

## 3. 끝난 뒤 (e_end 이후)

1. **남은 구간·후처리:** 자동이다 (종료 후 수집 실행). `[지난 끝, e_end)`를 받고 판독·합치기, 원본 대기 실행, `post_all.sh`까지 한다.
2. 아래 3~7은 사용자가 「S{n} 끝난 뒤 보강 해」라고 시킨 세션이 한다.
3. **데이터 점검:** [일괄 수집 「7. 데이터 점검」](solo-raid-backfill.md#7-데이터-점검-s34-이하실시간)과 같다.
4. **수집 끝 표시:** `$DEST/collect_done.json` (`{"done": "KST 시각", "by": "세션 id"}`) → 커밋. 이 표시가 있어야 분석 도구가 이 시즌을 쓴다.
5. **개발 시즌만:**
   - [일괄 수집 「8. 분석 갱신」](solo-raid-backfill.md#8-분석-갱신)의 3~5와 같다.
   - 자기 예측 채점: `python3 -I tools/solo_raid/score_live.py S{n}` → `predict/score.json` 커밋, 표를 채팅에 옮긴다.
6. **시험 시즌:** 채점·분석을 하지 않는다. README·채팅에 정답 점·컷 수치를 쓰지 않는다 (예측 수치는 진행 중에 이미 낸 것이라 괜찮다).
7. README 「시즌 요약」을 채우고 (일괄 수집 형식 + 「예측」 칸: 시점마다 커밋 시각과 등급 — 「자동 실행」 표에 있다) 사용자에게 끝을 알린다.

## 바뀐 기록

- 2026-10-16: 처음 작성 (사용자 결정 — 실시간은 진행 중 예측이 목적, 끝나면 보강으로 전환, Opus 주도, 시점마다 예측 커밋 + 채팅).
- 2026-10-16: 시즌 세션 모델을 Opus → Sonnet으로 (사용자 결정).
- 2026-10-16: 받기 끝 `e_end`를 종료 + 2일 → + 4일 05:00으로 (사용자 결정, 일괄 수집과 같게).
- 2026-10-16: 6시간 자동 수집으로 바꿈 (사용자 결정) — `send_later` 시점 방식 대신 예약 작업이 칸마다 새 세션, 예측 T=6부터 6시간마다·매번 웹 반영,
  시작은 사람·끝은 자동, `e_end`를 종료 + 2일 → 4일 05:00, 종료 후 수집은 e_end에 한 번. 도구 `tools/solo_raid/live_tick.py`·`live_tick.sh`.
