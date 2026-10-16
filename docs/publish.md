# 공개 레포와 웹 (nikke-soloraid-cutoff)

2026-10-16 사용자 결정. 솔로레이드 **코드와 파이프라인**을 공개 레포로 나누고, 커트라인을 **웹**으로 보여 준다.
이 레포(nikke-community)는 비공개로 남아 정본이 되고, 수집한 자료도 여기에만 둔다.

| 무엇 | 어디 |
|---|---|
| 공개 레포 | <https://github.com/jgaram/nikke-soloraid-cutoff> (`main`: 코드·문서, `gh-pages`: 웹) |
| 웹 | <https://jgaram.github.io/nikke-soloraid-cutoff/> — 진행 중 시즌 예측 · 지난 시즌 컷, `analysis.html`(분석 페이지) |
| 라이선스 | CC0 1.0 (사용자 결정) |
| 이름 | 처음 `nikke-solo-percent`로 만들었다가 같은 날 `nikke-soloraid-cutoff`로 바꿨다 (사용자 결정). GitHub가 옛 레포 주소는 넘겨 주지만 Pages 주소는 넘겨 주지 않는다 |
| 만드는 도구 | `tools/publish/build_public.py`(공개 레포 내용) · `tools/publish/site.py`(웹) · `.github/workflows/publish.yml`(자동 복사) |

## 정한 것

- **공개 범위는 솔로레이드 파이프라인만이다.** 문서 5개(`docs/solo-raid-*.md`·`community-access.md`)와 이 문서를 공개한다.
  코드는 `tools/solo_raid/`·`tools/publish/`, 수집 단계 `experiments/2026-10-15-solo-raid-low-cost/`의 `tools/`·`README.md`다.
  그 밖에 `.claude`의 raid-vision 에이전트·워크플로, 설정 `split.json`·`criteria.json`을 공개한다.
  목록은 `build_public.py`의 `ALLOW`가 정본이다 (허용 목록 방식 — 새 파일을 공개하려면 여기에 더한다).
- **웹에는 숫자와 개수만 싣는다.** 근거는 「커뮤니티 N건」으로만 쓰고, 원글 링크·글 번호·사진·닉네임은 넣지 않는다 (사용자 결정).
- **시험 시즌 정답 컷은 공개하지 않는다.** `site.py`는 split 감시를 켜고 `split.load_seasons()`로만 컷을 읽는다.
  시험 시즌은 「잠김」으로 표시하고, 진행 중 예측의 채점(`score.json`)도 개발 시즌 것만 싣는다.
  공개 웹을 분석 세션이 읽어도 블라인드가 깨지지 않게 하려는 것이다. 최종 시험(할 일 3.7)이 끝나면 풀지 다시 정한다.
- **갱신은 Actions 자동 복사다.** 이 레포 `main`에 관련 경로(워크플로의 `paths`)가 push되면, `publish.yml`이 공개 레포 `main`과 `gh-pages`를 통째로 다시 만든다.
  공개 레포에 직접 커밋한 것은 다음 복사 때 사라진다.
  - 웹의 기준 시각은 웹에 들어가는 자료·도구를 마지막으로 바꾼 커밋 시각이다. 그래서 자료가 그대로면 `gh-pages`에 커밋이 생기지 않는다.

## 내보내기 전에 손보는 것과 검사 (`build_public.py`)

- 공개하지 않는 파일(AGENTS.md, 실험 기록, `data/` 등)로 가는 마크다운 링크는 링크를 풀고 글자만 남긴다.
- 특정 글을 가리키는 값은 가린다: 디시 글 번호(`board/gov/{번호}`, `no={번호}`, `글 {번호}`)와 첨부 파일명의 촬영 시각(`Screenshot_YYYYMMDD_HHMMSS`).
- 다음 중 하나라도 걸리면 아무것도 쓰지 않고 멈춘다. Actions run이 빨갛게 뜬다.
  - 사진·동영상·jsonl·csv·tsv 파일
  - 1MB가 넘는 파일
  - 가린 뒤에도 남은 글 번호와 디시 이미지 주소
  - 웹 파일 안의 커뮤니티 글 주소
- 원문, 사진, OCR 결과(`ocr.jsonl` 등 — 랭킹 화면의 닉네임이 들어 있음)는 허용 목록에 없으므로 나가지 않는다.

손으로 확인: `python3 -I tools/publish/build_public.py --out {임시}/pub --site {임시}/site`

## 처음 설정 (사용자가 한 번 함)

1. **토큰:** GitHub → Settings → Developer settings → Fine-grained tokens → Generate new token.
   - Repository access: Only select repositories → `jgaram/nikke-soloraid-cutoff`
   - Permissions: Contents → Read and write
   - 만료일은 원하는 대로 정한다. 만료되면 복사가 빨갛게 뜨므로 새로 만들어 바꾼다.
2. **이 레포 Secret:** nikke-community → Settings → Secrets and variables → Actions → New repository secret.
   이름은 `PUBLIC_REPO_TOKEN`, 값은 1번의 토큰이다. 없는 동안 `publish.yml`은 아무것도 하지 않고 초록으로 끝난다.
3. **Pages:** nikke-soloraid-cutoff → Settings → Pages → Build and deployment → Source: Deploy from a branch → `gh-pages` / `/ (root)`.
4. 처음 한 번은 Actions → publish → Run workflow로 돌린다. 그다음부터는 push마다 돈다.

## 주의

- 수집기 UA `nikke-community-research/0.1 (+https://github.com/jgaram/nikke-community)`의 주소는 비공개 레포라 남에게는 404다.
  공개 레포 주소로 바꿀지는 따로 정한다 (2026-10-16 기준 그대로). 바꾸면 `docs/community-access.md`와 수집 지시문도 같이 고친다.
- 공개 문서에도 세션 ID, 실험 날짜 같은 작업 기록이 남는다. 개인정보는 아니라서 그대로 둔다.
