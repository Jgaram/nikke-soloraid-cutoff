# 니케 커뮤니티 접근 방법 조사

- 조사일: 2026-10-07
- 재검토일: 2026-10-07 — 주소·API를 다시 요청해 확인하고, robots.txt·신원 표시 문제를 고치고, 사이트별 수집 비용(토큰)을 추가했다. → [재검토](#재검토-2026-10-07)
- 테스트 환경: Claude Code 클라우드 컨테이너 (IP가 다른 환경에서는 결과가 다를 수 있음)
- 사이트 구조와 API는 예고 없이 바뀔 수 있으므로, 수집 코드를 만들 때 다시 확인할 것

2026-10-14 Codex 환경 재확인: 아래 2026-10-07 표는 당시 Claude 환경의 결과다. 자기 이름의 UA를 사용한 이번 조사에서는 디시·네이버 본문과 이미지, 블라블라 공개 검색·본문 API가 동작했다. 아카라이브는 목록 200, 검색 503, 본문 429였고, 에펨코리아는 robots.txt 429로 게시판을 요청하지 않았다. 접근 실패를 자료 부재로 해석하지 않는다. 실행별 요청·실패 기록은 솔로레이드 수집 방법 실험에 보관한다.

2026-10-15 무인 실행 조건 시험 (Claude 클라우드 컨테이너, 자기 이름 UA `nikke-community-research/0.1 (+https://github.com/jgaram/nikke-community)`, 요청 약 60개, 간격 1초):
네이버 게임 라운지(목록·본문·이미지)와 블라블라링크(목록·검색·본문·댓글·답글·이미지)가 모두 200으로 받아졌고 403·429는 없었다.
robots.txt도 다시 봤다 — `game.naver.com` `*` 허용, `comm-api.game.naver.com`·`nng-phinf.pstatic.net`·`api.blablalink.com`·`sg-cdn.blablalink.com`은 없음(404), `apis.naver.com`은 그대로 전체 차단.
두 곳 모두 사람 없이 도는 수집기를 붙일 수 있다. 사이트별 세부는 [네이버](#4-네이버-게임-라운지--승리의-여신-니케)·[블라블라링크](#5-블라블라링크-글로벌-공식-커뮤니티)의 「2026-10-15 확인」.

## 한눈에 보기

| 커뮤니티 | 목록 | 본문 | 이미지 | 동영상·움짤 | 댓글 | 답글 | 형태 |
|---|---|---|---|---|---|---|---|
| 디시인사이드 | ✅ | ✅ | ✅ | ✅ mp4 (업로드 영상은 Referer 필요) | ✅ | ✅ | HTML + 댓글 JSON |
| 아카라이브 | ✅ | ✅ | ✅ (1시간 만료 URL) | ✅ mp4 (1시간 만료 URL) | ✅ | ✅ | HTML (`Claude-User` 필요) |
| 에펨코리아 | ✅ | ✅ | ✅ | ✅ 움짤 mp4 (업로드 영상 미확인) | ✅ | ✅ | HTML (`Claude-User` 필요) |
| 네이버 게임 라운지 | ✅ | ✅ | ✅ | ⛔ 재생 API가 robots.txt 차단 | ⛔ robots.txt 차단 | ⛔ robots.txt 차단 | JSON API |
| 블라블라링크 | ✅ | ✅ (번역본/원문) | ✅ | YouTube 링크만 | ✅ | ✅ | JSON API (POST) |

⛔: 요청하면 받아지지만(첫 조사에서 확인) robots.txt가 막는 주소라 쓰지 않는다. 네이버는 댓글 수만 목록·본문 응답의 `comment`에서 알 수 있다.

모든 곳에서 로그인 없이 접근할 수 있었다. Referer가 필요한 건 디시의 원본 첨부파일과 업로드 동영상뿐이었고, 나머지 이미지·동영상은 Referer 없이 받아졌다.

수집 비용: 코드로 필요한 필드만 뽑으면 댓글 1개에 30~40토큰으로 사이트 간 차이가 거의 없다. 원본 HTML을 그대로 모델에 넣으면 디시 글 1개가 13만 토큰을 넘는다. → [수집 비용](#수집-비용-토큰-추정)

## 공통 원칙

- **robots.txt를 따른다.** robots.txt가 막은 주소는 요청하지 않는다. 같은 사이트라도 API 도메인이 따로 있으면 그 도메인의 robots.txt도 확인한다 (네이버 `apis.naver.com`은 전체 차단).
- **신원 표시:** 실제 요청 주체에 맞는 UA를 쓴다. Claude가 사용자 요청을 받아 가져올 때는 `User-Agent: Claude-User`, 다른 에이전트·수집기는 자기 이름의 UA를 쓴다. 2026-10-07 Claude 환경에서는 아카라이브·에펨코리아가 `Claude-User`를 허용했지만 이를 다른 실행 주체에 그대로 적용하지 않는다. 브라우저나 다른 서비스로 위장해 보안 시스템을 우회하지 않는다. (2026-10-14 명확화)
- **`Claude-User`는 Claude가 사용자 요청을 받아 가져올 때만 쓴다.** 사람 없이 도는 수집기(정기 실행 등)가 이 이름을 쓰면 신원을 속이는 것이다. 그런 수집기는 자기 이름의 UA를 쓰고, 접근 가능 범위는 해당 UA의 robots 규칙과 현재 응답으로 다시 확인한다. 무인 수집을 하더라도 빈도는 낮게 유지한다.
- **AI 학습에 쓰지 않는다:** 디시인사이드·에펨코리아 robots.txt는 AI 학습 크롤러(ClaudeBot, GPTBot 등)를 막는다. 수집한 글을 모델 학습 데이터로 쓰지 않는다.
- **요청 간격:** 요청 제한(rate limit)은 확인하지 않았다. 간격은 작업마다 정하고, 403·429가 나오면 멈춘다. (고정 「1초 이상」 규칙은 2026-10-15 사용자 결정으로 뺐다.)
- **개인정보:** 댓글에는 닉네임, 회원 ID, IP 앞자리(디시 유동닉) 등이 들어 있다. 요약 결과물에는 필요한 것만 남긴다.
- **외부 임베드:** YouTube·X(트위터) 등 외부 서비스 영상은 각 커뮤니티에 링크나 iframe으로만 들어 있다. YouTube는 약관상 영상 파일을 내려받지 않고, 링크·영상 ID·제목·썸네일만 기록한다.
- **만료 URL:** 아카라이브 본문 이미지·동영상(1시간)은 서명이 붙은 URL이 만료된다. 글을 받은 직후 바로 내려받거나, 나중에 필요하면 글을 다시 받아 새 URL을 얻는다.

### 사이트별 robots.txt와 User-Agent

재확인 2026-10-07. 목록 1페이지를 요청한 결과.

| 커뮤니티 | robots.txt | `Claude-User` | 자기 이름 UA¹ | 일반 브라우저 UA² |
|---|---|---|---|---|
| 디시인사이드 | `*` 허용. AI 학습 크롤러(ClaudeBot, anthropic-ai, Claude-Web, GPTBot 등)만 차단 | 200 | 200 | 200 |
| 아카라이브 | `*` 대부분 허용 | 200 | 403 (Cloudflare) | 403 (Cloudflare) |
| 에펨코리아 | `Claude-User` 전체 허용. `*`는 첫 화면·`/best`·`/best2`·`/humor`만 허용 (`/nikke` 불가) | 200 | 시험 안 함 (robots.txt 불허) | 430 |
| 네이버 게임 라운지 | `game.naver.com` 허용, `comm-api.game.naver.com` 없음(404), **`apis.naver.com` 전체 차단** | 200 (comm-api) | 200 (comm-api) | — |
| 블라블라링크 | 없음 (`www`는 앱 HTML을 돌려주고, `api`는 404) | 200 | 200 | — |

1. `nikke-community-research/0.1 (+https://github.com/jgaram/nikke-community)`
2. 첫 조사 결과. 공통 원칙에 따라 다시 시험하지 않았다

---

## 수집 비용 (토큰 추정)

측정 2026-10-07. 사이트마다 목록 1페이지와 댓글이 달린 최근 글 2~3개(디시는 댓글 123개 글 1개 추가)를 받아 쟀다. 표본이 작으니 자릿수를 보는 용도로 쓴다. 토큰은 모델에 텍스트를 넣을 때 드는 입력 토큰이다.

### 재는 방법

- 토크나이저 없이 추정했다 (오차 ±20% 정도). 한국어 텍스트·HTML·JSON 표본을 Claude 컨텍스트에 실제로 넣어 줄어든 토큰 수를 재고, 거기에 맞춘 식으로 계산했다:
  `토큰 ≈ 1.5 × (한글·한자·가나 글자) + 0.55 × (그 밖의 공백 아닌 문자) + 0.2 × (공백 한 칸) + 1 × (줄바꿈·들여쓰기 덩어리)`
- 어림셈: 원본 HTML·JSON은 **UTF-8 2~4바이트당 1토큰** (들여쓰기가 많으면 4, ID·URL·이스케이프가 빽빽하면 2). 추출한 한국어 본문은 **글자 수 × 1.1**.
- 세 단계로 비교했다:
  - **원본**: 응답을 그대로 모델에 넣을 때
  - **태그 제거**: HTML에서 `script`·`style`을 빼고 텍스트만 남긴 것 (사이트 메뉴·안내문 포함)
  - **추출**: 코드로 필요한 필드만 뽑은 것. 글은 `제목 / 작성자 | 시각 / 본문`, 댓글은 한 줄에 `작성자 | 시각 | 내용`, 이미지·이모티콘은 `[이미지]` 같은 자리표시

### 목록 1페이지

| 커뮤니티 | 응답 크기 | 글 수 | 원본 | 태그 제거 | 추출 | 추출 시 글 1개당 |
|---|---|---|---|---|---|---|
| 디시인사이드 | 216KB HTML | 50 (공지 9) | ~102,000 | ~9,900 | ~2,800 | ~56 |
| 아카라이브 | 151KB HTML | 53 (공지 10) | ~44,000 | ~3,700 | ~2,700 | ~51 |
| 에펨코리아 | 76KB HTML | 27 (공지 7) | ~40,000 | ~3,200 | ~1,300 | ~48 |
| 네이버 게임 라운지 | 180KB JSON | 25 | ~100,000 | — | ~1,500 | ~61 |
| 블라블라링크 | 120KB JSON | 10 | ~66,000 | — | ~620 | ~62 |

- 네이버·블라블라링크는 목록 응답에 글마다 본문 전체와 작성자 정보가 들어 있어서 원본이 크다 (글 1개당 네이버 ~4,000, 블라블라 ~6,600).

### 글 1개 (본문 + 댓글)

비용 ≈ 글 고정분 + 댓글 수 × 댓글 1개당

| 커뮤니티 | 요청 수 | 원본: 글 고정분 | 원본: 댓글 1개당 | 태그 제거: 글 페이지 | 추출: 댓글 1개당 |
|---|---|---|---|---|---|
| 디시인사이드 | 2 (글 HTML + 댓글 JSON), 댓글 100개마다 +1 | ~135,000 (300KB) | ~690 (JSON) | ~11,000 | ~40 |
| 아카라이브 | 1 (댓글이 글 HTML 안), 댓글 페이지마다 +1 | ~49,000 (165KB) | ~550 (HTML) | ~4,100 | ~36 |
| 에펨코리아 | 1 (댓글이 글 HTML 안), 댓글 50개마다 +1 | ~42,000 (85KB) | ~780 (HTML) | ~3,000 | ~34 |
| 네이버 게임 라운지 | 1 (+ 댓글 API) | ~2,000 + 본문 HTML | 측정 안 함 | — | 측정 안 함 |
| 블라블라링크 | 2~3 (본문 + 댓글 + 답글) | ~2,400 + 본문 HTML | ~1,470 (JSON) | — | ~31 |

- 추출한 글 고정분(제목·작성자·시각)은 어디나 30~60토큰이고, 여기에 본문 글자 수 × 1.1이 붙는다.
- 네이버·블라블라 본문 HTML은 꾸밈 태그가 많아 추출 텍스트의 약 10배다 (블라블라 긴 글 하나: 원본 ~30,000 → 추출 ~2,400).
- 블라블라링크는 댓글마다 작성자 정보(`user`: 업적·아바타 등)가 통째로 붙어서 댓글 원본이 가장 크다.
- 네이버 댓글은 `apis.naver.com`(robots.txt 전체 차단)에 있어서 요청하지 않았다.

**예: 댓글 20개가 달린 짧은 글 1개**

| 커뮤니티 | 원본 그대로 | 추출 후 |
|---|---|---|
| 디시인사이드 | ~149,000 | ~860 |
| 아카라이브 | ~60,000 | ~780 |
| 에펨코리아 | ~58,000 | ~740 |
| 블라블라링크 | ~32,000 | ~680 |

### 정리

- **원본을 그대로 모델에 넣지 않는다.** 디시 글 페이지 하나가 13만 토큰을 넘는다 (아래쪽에 갤러리 목록이 통째로 같이 들어 있음). 태그만 지워도 사이트 메뉴·안내문 때문에 디시 1만, 아카 4천, 에펨 3천 토큰이 남는데, 그중 본문은 수십 토큰이다.
- **코드로 필드만 뽑으면 사이트 간 차이가 거의 사라진다** (원본의 1/50~1/170). 그 뒤 비용은 글·댓글 양에 비례하고 댓글 1개에 30~40토큰이다.
- 추출 후에도 남는 사이트 차이는 요청 수와 받는 데이터 양(시간·트래픽)이다. 글 1개에 디시 ~300KB·2회, 아카 ~170KB·1회, 에펨 ~90KB·1회, 네이버 4~9KB·1회, 블라블라 5~60KB·2~3회.
- 이미지·동영상은 모델에 보여주지 않으면 토큰이 들지 않는다. 모델에 이미지를 보여주면 이미지 크기에 따라 따로 토큰이 든다 (이번에 재지 않음).

---

## 1. 디시인사이드 — 니케 마이너 갤러리

- 갤러리 ID: **`gov`** (`id=nikke`는 폐쇄된 갤러리)
- 요청: `User-Agent: Claude-User`로 GET 200 (재확인 2026-10-07). 첫 조사에는 "일반 브라우저 UA로 GET 가능"이라고 적었는데, 공통 원칙(브라우저로 위장하지 않음)에 맞지 않아 고쳤다
- robots.txt: AI 학습 크롤러(ClaudeBot, anthropic-ai, Claude-Web, GPTBot 등)는 차단. `Claude-User`는 목록에 없어 `*`(허용)이 적용됨
- 자기 이름 UA(`nikke-community-research/0.1 …`, 사람 없이 도는 자동 실행용): 목록·본문·본문 사진(`dcimg*.dcinside.co.kr`)·원본 첨부(`image.dcinside.com/download.php`, Referer 필요) 모두 200.
  사진 sha1도 `Claude-User`로 받았을 때와 같았다. 사진 호스트 두 곳에는 robots.txt가 없다(404). 요청 660번 실패 0 (재확인 2026-10-15, [솔로레이드 저비용 수집 세션 F1](../experiments/2026-10-15-solo-raid-low-cost/README.md#결과--세션-f1-4단계-도구-2026-10-15))
  - 본문 사진 서버 중 `dcimg8.dcinside.co.kr`은 Referer가 없으면 403이고, `-e https://gall.dcinside.com/`(갤러리 쪽 Referer)를 주면 200이다.
    같은 때 `dcimg1`·`dcimg4`는 Referer 없이 200. 그래서 본문 사진도 글 주소를 Referer로 보낸다 (확인 2026-10-16, 솔로레이드 S37 글 … 사진, 요청 3번).

### 목록
- 전체글: `GET https://gall.dcinside.com/mgallery/board/lists/?id=gov`
- 개념글: `...&exception_mode=recommend`
- 페이지: `...&page=N`
- 파싱: `tr.ub-content[data-no]` 한 줄이 글 하나
  - 제목 `.gall_tit a`, 조회수 `.gall_count`, 추천수 `.gall_recommend`, 작성시각 `.gall_date[title]`, 댓글 수 `.reply_num`
  - 작성자 `.gall_writer`의 `data-nick`(닉네임), `data-uid`(고정닉 ID), `data-ip`(유동닉 IP 앞자리)
  - 공지는 `tr`의 `data-type="icon_notice"` (일반 글은 `icon_pic`·`icon_txt`·`icon_movie`·`icon_recomimg` 등 글 종류 아이콘)

2026-10-14 역대 게시물 조사에서 확인:

- 제목 검색은 `s_type=search_subject&s_keyword={검색어}`를 사용한다. 검색 결과는 글 번호 구간으로 나뉘며, 첫 구간의 결과가 비어 있다고 전체 기간에 글이 없다는 뜻은 아니다.
- 페이지 하단의 `이전 검색`·`다음 검색` 링크에 있는 `search_pos`를 따라 과거 구간으로 이동한다. 특정 구간 안의 페이지 이동과 검색 구간 이동은 별개다. 구간 크기를 고정값으로 가정하지 말고 실제 링크와 게시 시각을 확인한다.

2026-10-16 역대 컷 정리 조사에서 확인 (실험):

- 한 구간은 글 번호 1만 개다. 갤러리 첫 글까지 가려면 키워드 하나에 구간 631개(요청 약 630번, 1.2초 간격이면 약 30분)가 든다. 다음 구간 주소는 `a.search_next`의 `search_pos`.
- 띄어 쓴 검색어(`솔레 컷`)는 붙인 말(`솔레컷`)과 결과가 다르다. 둘 다 돌린다.
- 2023-08~2024-09 글의 본문 사진은 대부분 지워져 디시 기본 이미지(550×350 GIF)만 오고, 원본 첨부도 「해당 파일이나 경로가 존재하지 않습니다」다. 이 시기 값은 본문 글자에서만 얻는다.
- 유동닉은 닉이 모두 「ㅇㅇ」이니 사람을 가를 때 닉만으로 묶지 않는다 (IP 앞자리나 글 단위로).

### 본문
- `GET https://gall.dcinside.com/mgallery/board/view/?id=gov&no={글번호}`
- 본문 영역: `div.write_div`
- 제목 `.title_subject`, 작성자 `.gall_writer[data-nick]`, 작성시각 `.gall_date[title]`
  - 글 페이지 아래쪽에 갤러리 목록이 통째로 같이 들어 있어서 `.gall_writer`·`.gall_date`가 50개 넘게 잡힌다. 첫 번째 것이 이 글의 값이다

2026-10-14에는 공개 모바일 주소 `https://m.dcinside.com/board/gov/{글번호}`에서도 일부 글을 확인했다. 응답이 PC 형태일 수 있어 `.write_div` 등 실제 구조를 확인한다. 본문 수신 뒤 첨부 이미지 일부가 실패해도 이미 얻은 본문·출처·다른 이미지를 버리지 않고, 실패한 첨부만 별도로 표시한다. 각 호스트의 robots.txt와 공통 요청 간격을 지킨다.

### 이미지
- 본문 이미지: `https://dcimg{N}.dcinside.co.kr/viewimage.php?id=...&no=...` (`dcimg{N}.dcinside.com`도 있음)
  - Referer 없이 다운로드됨 (재확인 2026-10-07)
  - `Content-Type`이 `application/octet-stream`이라 확장자는 파일 헤더로 판별해야 함
  - 지연 로딩: 이미지가 많은 글은 `<img class="lazy" src="…gallview_loading_ori.gif" data-original="{실제 주소}">` 꼴이다. `data-original`을 먼저 보고 없으면 `src`를 쓴다 (재확인 2026-10-07)
  - Range 요청은 응답 본문이 요청한 범위보다 짧게 오는 등 불안정했다. 파일 전체를 받는다
- 원본 첨부파일: `https://image.dcinside.com/download.php?no=...&f_no={파일명}` (본문 하단 첨부 목록)
  - **`Referer: https://gall.dcinside.com/` 필요** (없으면 빈 응답)
  - `f_no`에 올린 사람의 원래 파일명이 남고, 원본 파일에는 EXIF가 남아 있기도 하다. 본문 표시용 이미지는 다시 인코딩돼 EXIF가 대부분 없다 (2026-10-15)
  - 2026-10-15 Claude 클라우드 환경: 이미지 호스트(`dcimg*`, `image.dcinside.com`, 네이버 `nng-phinf.pstatic.net`) 요청의 약 15~35%가 TLS 단계에서 끊겼다(`curl: (35) Connection reset`). 다시 요청하면 대개 받아졌다. Python `urllib`은 이때 한 장에 수십 초씩 걸려 curl로 받았다. 실패한 요청도 1.2초 간격 한 칸을 쓴다
    - 같은 날 다른 세션(02~08시 KST)에서는 목록·본문·이미지·원본 13,569번 중 재시도가 필요한 것이 161번(1.2%)뿐이었다 (curl, 최대 4번 시도). 시간대·세션마다 다를 수 있다
- 본문 표시용 이미지는 원본을 **가로 850px로 줄인 사본**이다. 긴 캡처 속 작은 숫자(11~12자리 점수)는 사본에서 한 자리씩 틀리게 읽혔다(약 3%). 숫자를 정확히 옮겨야 하면 원본 첨부를 받는다 (2026-10-15, 실험)
  - 첨부 목록과 본문 이미지는 **개수가 같을 때만** 순서로 짝지을 수 있다. 디시콘·붙여 넣은 사진이 섞이면 어긋난다 (사진 글 4,096개 중 2,936개가 같았다, 2026-10-15)
  - 첨부 파일명(`Screenshot_YYYYMMDD_HHMMSS_NIKKE.jpg` 등)에 찍은 시각이 남는다. 시즌 전 날짜면 지난 시즌 화면이다. `gov-…`는 디시 앱이 붙인 이름이라 시각으로 쓰지 않는다

### 동영상
디시에는 형태가 세 가지 있다.

**1) 움짤 (GIF가 mp4로 변환된 것)**
- 본문에 바로 들어 있음: `<video autoplay loop muted ...><source src="https://dcimg{N}.dcinside.co.kr/viewimage.php?id=...&no=..." type="video/mp4"></video>`
- 이미지와 같은 `viewimage.php` 주소이고, Referer 없이 mp4로 받아짐
- `<img class="gif-mp4" src="…viewimage.php?…">`로 들어간 것도 있다. 이 주소는 GIF 파일을 돌려줬다 (재확인 2026-10-07)

**2) 업로드 동영상**
- 목록에서 아이콘이 `icon_movie`인 글
- 본문에는 플레이어 iframe만 있음: `<iframe src="https://gall.dcinside.com/board/movie/movie_view?no={영상번호}">`
- 플레이어 페이지와 영상 파일 모두 **Referer가 필요**하다 (`Referer: https://gall.dcinside.com/` 또는 글 URL)

| 요청 | Referer 없음 | Referer 있음 |
|---|---|---|
| 플레이어 `movie_view?no=...` | 200이지만 빈 응답 | 200, `<video>` HTML |
| 재생용 `https://dcm{N}.dcinside.co.kr/viewmovie.php?code=...&key=...&type=mp4` | 403 | mp4 (Range 요청 지원) |
| 다운로드용 `https://dcm{N}.dcinside.co.kr/download.php?code=...&key=...&f_no={파일명}.mp4` | 빈 응답 | mp4 원본 |

- 썸네일: 재생용 주소의 `type=mp4`를 `type=jpg`로 바꾼 것 (`<video poster>`)

**3) YouTube 임베드**
- `<iframe src="https://www.youtube.com/embed/{영상ID}">` → 링크와 ID만 기록

### 댓글
- `POST https://gall.dcinside.com/board/comment/`
- 헤더: `X-Requested-With: XMLHttpRequest`, `Referer: {글 URL}`
- 폼 데이터: `id=gov&no={글번호}&cmt_id=gov&cmt_no={글번호}&e_s_n_o={토큰}&comment_page={N}&_GALLTYPE_=M`
  - `e_s_n_o`: 글 페이지 HTML의 `<input name="e_s_n_o" value="...">` 값
  - 글 페이지에서 받은 쿠키를 유지한 상태로 요청함
- 응답(JSON):
  - `total_cnt`: 전체 댓글 수
  - `comments[]`: `no`, `name`, `user_id`, `ip`(유동닉만), `reg_date`(`MM.DD HH:MM:SS`, 연도 없음), `memo`(HTML)
  - 답글: `depth`가 1이고 `c_no`가 부모 댓글 번호
  - `pagination`: 페이지 링크 HTML
- 페이지 (재확인 2026-10-07):
  - 페이지당 댓글 100개 (답글 포함, 댓글돌이는 따로 더 붙음)
  - `comment_page=1`이 **가장 최근 100개**이고, 숫자가 커질수록 오래된 댓글이다. 한 페이지 안에서는 오래된 순
  - 페이지 수 = `total_cnt` ÷ 100 올림. 예: 댓글 123개 글 → 1페이지 100개(최근), 2페이지 23개(가장 오래된 것)
  - 이 글에서 답글 51개는 모두 부모 댓글과 같은 페이지에 있었다
- 주의:
  - `no`가 0인 "댓글돌이" 항목(디시 뉴스 광고)이 섞여 있으니 걸러야 함 (댓글 20개 글에 3개, 100개 페이지에 8개)
  - 디시콘(이모티콘)은 `memo` 안의 `<img>`/`<video>` (`dcimg5.dcinside.com/dccon.php?...`)

---

## 2. 아카라이브 — 승리의 여신 니케 채널

- 채널: **`nikketgv`** (`/b/nikke`는 2022년 이후 비활성 채널)
- 요청: 일반 브라우저 UA는 Cloudflare 403. **`User-Agent: Claude-User`면 200** (재확인 2026-10-07). 자기 이름 UA도 403
- robots.txt: 대부분 허용

### 목록
- 전체글: `GET https://arca.live/b/nikketgv`
- 개념글: `?mode=best`
- 페이지: `?p=N`
- 파싱: `a.vrow.column` 한 개가 글 하나
  - 링크가 `/b/nikketgv/{글번호}?p=1` 꼴이라 `href`를 `^/b/nikketgv/(\d+)`로 매칭한다 (재확인 2026-10-07: 첫 조사의 `href="/b/nikketgv/{글번호}"` 완전 일치로는 안 잡힘)
  - 제목 `.title`, 댓글 수 `.comment-count`, 작성자 `.user-info [data-filter]`, 작성시각 `time[datetime]`
  - 공지는 `notice` 클래스 (공지 제목은 `.col-title`)

2026-10-16 확인 (`Claude-User`, 실험): 채널 검색(`?target=title|all&keyword=`)은 「전산 점검 중」 503이 몇 시간 계속됐다.
대신 `?category=솔로레이드`(「솔레/뮤지엄」, 2024-07부터 글 있음)와 `before={ISO 시각}`으로 원하는 날짜 근처 목록을 바로 열 수 있었다 (카테고리·개념글 목록 모두).
이미지에 `&type=orig`를 붙이면 302라 빈 파일이 된다 — 서명 URL 그대로 받는다.
  - 같은 `a.vrow.column`으로 다른 채널 광고(`notice-service`, 링크 `https://arca.live/b/{다른 채널}/...`)와 "숨겨진 공지 펼치기"(링크 없음)도 들어 있으니 걸러야 함

### 본문
- `GET https://arca.live/b/nikketgv/{글번호}`
- 본문 영역: `div.article-content`
- 제목 `.article-head .title`, 작성자 `.article-head .user-info`, 작성시각 `.article-head time[datetime]`

### 이미지·동영상
- 형식: `//ac.arca.live/{날짜}sac/{해시}.{확장자}?expires={유닉스시각}&key={서명}`
- 서명(`expires`, `key`)이 있어야 받아짐 (빼면 403). Referer는 필요 없음 (재확인 2026-10-07)
- **본문 이미지·동영상 URL은 페이지를 받은 뒤 1시간 후 만료** → 글을 받은 직후 바로 다운로드해야 함
  - 만료 시각은 URL의 `expires`(유닉스 시각)에 있으니 그 값으로 판단
  - 재확인 2026-10-07: `expires`가 페이지를 받은 시각 + 정확히 3600초였다
  - 댓글의 아카콘은 만료가 약 24일로 더 길었다
- 확장자가 `.png`여도 실제로는 WebP로 내려옴. `&type=orig`를 붙이면 원본으로 302 리다이렉트

**동영상**
- 본문 동영상: `<video src="//ac.arca.live/{날짜}sac/{해시}.mp4?expires=...&key=..." poster="...thumbnail.webp?...">`
  - 이미지와 같은 서명 URL 방식이고, Referer 없이 mp4로 받아짐 (서명 빼면 403)
  - GIF를 올린 경우 `data-orig="gif"`가 붙고, `data-originalurl`에 원본 GIF 주소(`//ac-o.arca.live/...gif?expires=...&key=...&type=orig`)가 있음
  - 확인한 샘플은 GIF가 변환된 mp4였음. 처음부터 동영상 파일로 올린 글은 확인하지 못함
- 아카콘(이모티콘)도 `<video class="arca-emoticon" data-store-id="...">`로 들어가니, 본문 동영상과는 `arca-emoticon` 클래스로 구분
- 외부 임베드:
  - YouTube: `<iframe src="https://www.youtube.com/embed/{영상ID}">`
  - X(트위터): `<iframe class="safeframe tweet" src="https://safefra.me/twitter/?url={트윗 URL}">` → `url` 파라미터에서 트윗 주소를 꺼내 기록

### 댓글
- 글 HTML 안에 포함
  - 항목: `div.comment-item#c_{댓글ID}`
  - 작성자 `.user-info a[data-filter]`, 시각 `<time datetime="ISO8601">`(UTC, `...Z`), 내용 `.message .text`
  - 이모티콘만 단 댓글은 `.text`가 없고 `.message .emoticon-wrapper`에 `img.emoticon`(또는 `video`)만 있음 (재확인 2026-10-07)
  - `.message` 안의 "펼쳐보기▼"(`.btn-more`)는 버튼이니 내용에서 뺀다
- 페이지: `?cp=N`
  - 파라미터가 없으면 **마지막 페이지(최신 댓글)**가 나옴
  - 페이지 링크(`?cp=N#comment`)에서 최대 번호를 읽어 1부터 순회
- 답글: `div.comment-wrapper`가 중첩되는 깊이로 구분 (깊이 1 = 댓글, 2 이상 = 답글)
- 아카콘(이모티콘)과 댓글 이미지도 `ac.arca.live` 서명 URL

---

## 3. 에펨코리아 — 승리의 여신 니케 게시판

- 게시판: `https://www.fmkorea.com/nikke` (`mid=nikke`)
- 요청: 일반 브라우저 UA는 430(에펨코리아 보안 시스템), 헤드리스 Chromium도 실패. **`User-Agent: Claude-User`면 200** (재확인 2026-10-07)
- robots.txt: `Claude-User`는 모든 페이지 허용. 그 밖의 봇(`*`)은 첫 화면·`/best`·`/best2`·`/humor`만 허용하고 `/nikke`는 막는다. AI 학습 크롤러(ClaudeBot, GPTBot 등)는 첫 화면만
  - 그래서 `Claude-User`가 아닌 수집기로는 이 게시판을 받을 수 없다 (공통 원칙 참고)

### 목록
- 전체글: `GET https://www.fmkorea.com/nikke` 또는 `https://www.fmkorea.com/index.php?mid=nikke`
- 인기순: `...&sort_index=pop&order_type=desc`
- 페이지: `...&page=N`
- 파싱: 글 링크 `td.title > a:not(.replyNum)`, 댓글 수 `a.replyNum`
- 검색 (2026-10-16 확인): `https://www.fmkorea.com/search.php?mid=nikke&search_target=title&search_keyword={검색어}&page=N` (`title_content`는 본문까지). `index.php`로 보내면 302.
  한 검색어에 보통 4~30쪽까지만 나오고 2023~2024 전반 글은 거의 걸리지 않는다. 이미지는 본문의 `.png.webp` 주소만 받아지고 `.png`만 요청하면 404.
  - 재확인 2026-10-07: 첫 조사의 `td.title > a[href]`는 댓글 수 링크(`a.replyNum`, `...#{글번호}_comment`)까지 잡아서 댓글 있는 글이 두 번 나온다
  - 공지는 `tr.notice`

### 본문
- `GET https://www.fmkorea.com/{글번호}`
- 본문 영역: `<article>` 안의 `.xe_content`
- 제목 `h1 .np_18px_span`
- 작성시각은 절대시각(`2026.10.07 14:06`)으로 나옴

### 이미지
- `//image.fmkorea.com/files/attach/new5/{날짜}/{글번호}_..._{해시}.{확장자}`
  - 본문에는 `.jpeg.webp`처럼 WebP로 바꾼 주소가 들어 있음 (재확인 2026-10-07)
- Referer 없이 다운로드됨 (재확인 2026-10-07)

### 동영상
- 움짤(GIF가 mp4로 변환된 것):
  ```html
  <video class="mediaelement_video ..." poster="//image.fmkorea.com/files/attach/.../{파일}.gif.thumb.webp">
    <source src="//media.fmkorea.com/files/attach/new5/{날짜}/{글번호}_..._{해시}.gif.mp4?d" type="video/mp4" />
  ```
  - 동영상은 `media.fmkorea.com`에서 내려옴. Referer 없이, `Claude-User`·일반 브라우저 UA 모두 mp4로 받아짐
- 최근 글 44개 중 동영상이 있는 글은 이 움짤 1건뿐이었다. 직접 올린 동영상과 YouTube 임베드가 어떻게 들어가는지는 확인하지 못함

### 댓글
- 글 HTML 안에 포함
  - 항목: `li[id^="comment_"]`
  - 작성자 `.member_plate`, 시각 `.date`, 내용 `.xe_content`
- 페이지: `GET https://www.fmkorea.com/index.php?mid=nikke&document_srl={글번호}&cpage=N`
  - 페이지당 50개
  - 파라미터가 없으면 마지막 페이지가 나옴
  - 범위를 넘는 `cpage`도 마지막 페이지를 돌려주므로, 댓글 ID가 겹치면 끝으로 판단
- **댓글 시각이 상대표기**("7 시간 전", "1 분 전")라 수집 시각 기준으로 환산해야 함
- 이미지콘(이모티콘): `image.fmkorea.com/filesn/imagecon/...`

### 답글
댓글과 같은 `li[id^="comment_"]` 목록에 섞여 나오고, 부모 댓글 바로 아래에 트리 순서로 붙는다 (작성 시간순이 아님).

| 구분 | 일반 댓글 | 답글 |
|---|---|---|
| `li`의 class | `fdb_itm clear comment-…` | `fdb_itm clear re bg0`/`bg1 comment-…` (`re` 포함) |
| `li`의 style | 없음 | `margin-left:{2×깊이}%` (2% = 1단계, 4% = 2단계 …) |
| 부모 링크 | 없음 | `.xe_content` 맨 앞의 `a.findParent` |

- 부모 댓글 ID: `a.findParent`의 `onclick="return findComment({부모댓글ID}, event);"` 또는 `href`의 `#comment_{부모댓글ID}`
- `a.findParent`의 텍스트는 부모 댓글 작성자 닉네임이다. 본문 텍스트를 뽑을 때 이 링크를 먼저 제거해야 닉네임이 내용 앞에 붙지 않는다
- 예시:
  ```html
  <li id="comment_10422534077" class="fdb_itm clear re bg1   comment-10422534083" style="margin-left:2%">
    ...
    <div class="comment_10422534077_1555157335 xe_content">
      <a class="findParent" href="https://www.fmkorea.com/10422528645/10422532731#comment_10422532731"
         onclick="return findComment(10422532731, event);">욘사리코</a> 월 30000원임
    </div>
  ```
- 검증 범위: 글 12개의 댓글 첫 페이지에서 답글 73개를 확인했다. 모두 `re` 클래스와 `a.findParent`가 함께 있었고, 깊이는 최대 5단계(`margin-left:10%`)까지 나왔다 (재확인 2026-10-07: 글 3개, 답글 3개에서도 같음)
- 샘플에서는 부모 댓글이 모두 같은 페이지에 있었다. 답글과 부모가 다른 페이지로 갈리는 경우는 확인하지 못했지만, 부모 ID가 링크에 있으므로 페이지를 모두 모은 뒤 ID로 연결하면 된다

### 미확인
- 직접 업로드한 동영상, YouTube 임베드

---

## 4. 네이버 게임 라운지 — 승리의 여신: 니케

- 라운지 ID: `nikke` (내부 ID `originalLoungeId`: `NIKKE_The_Goddess_of_Victory`)
- 요청: 로그인 없이 JSON API. Referer/Origin 없이도 응답함
- robots.txt (재확인 2026-10-07):
  - `game.naver.com`: 일반 허용 (네이버 검색봇 Yeti만 `/lounge/` 차단)
  - `comm-api.game.naver.com`(목록·본문 API): robots.txt 없음(404)
  - **`apis.naver.com`(댓글·답글·동영상 재생 API): `User-agent: *` / `Disallow: /`로 전체 차단** → 이 API들은 쓰지 않는다. 첫 조사는 `game.naver.com`만 확인했었다. 아래 동영상·댓글·답글 설명은 첫 조사 기록으로만 남긴다

API 기본 주소: `https://comm-api.game.naver.com/nng_main/v1`

### 목록
- 글 목록: `GET /community/lounge/nikke/feed?boardId={0=전체}&buffFilteringYN=N&limit=25&offset=0&order=NEW`
  - 항목: `content.feeds[]`의 `feed`(`feedId`, `title`, `createdDate`, `buff`, `iconTypes`), `user`(`nickname`), `board`(`boardName`), `comment`(`totalCount`, `commentCount`, `replyCount`)
  - 응답에 본문(`feed.contents`)도 들어 있는데, 목록의 `contents`는 HTML이 아니라 **SmartEditor JSON 문자열**(`{"document":{"components":[...]}}`)이다. 본문 API의 `contents`는 HTML이다 (재확인 2026-10-07)
- 게시판 목록(boardId 확인용): `GET /lounge/nikke/board`
- 라운지 인기글: `GET /lounge/nikke/new/popularFeeds`

2026-10-14 확인: 위 글 목록 API의 `offset`은 페이지 번호처럼 동작했다. `limit=25`라고 해서 0→25→50으로 이동하면 중간 페이지를 건너뛸 수 있으므로 0→1→2의 반환 글 ID·게시 시각을 확인한다. 시즌41 당시 자료는 이번 조사 시점의 자유게시판(`boardId=4`) 페이지 25~44 부근이었다. 이 위치는 시간이 지나면 바뀐다. 댓글 API의 offset 규칙과 혼동하지 않는다.

2026-10-15 확인 (자기 이름 UA): `boardId=0` 페이지 0·1이 각각 25개, 두 페이지가 약 하루치(10-14 20:19~10-15 19:30)였다. `boardId=4&offset=40`은 09-24 글이었다 → 시즌 하나 거슬러 가는 데 목록 수십 번이면 된다.
목록 1페이지 응답은 약 18만~20만 바이트(`contents` SmartEditor JSON 포함)라 필요한 필드만 뽑는다.
오래된 글: `boardId=0`에서 `offset`을 건너뛰어 요청하면 바로 그 시점 글이 온다 — 100→2026-09-26, 300→2026-09-05, 1000→2026-06-19, 3000→2025-11-06, 10000→2023-10-30, 30000→빈 목록. 원하는 날짜는 offset을 이분 탐색하면 몇 번 만에 찾는다.

검색 (2026-10-16 확인, 웹 스크립트 `index-*.js`에서 찾은 경로): `GET https://comm-api.game.naver.com/nng_main/v2/search/feeds?keyword={검색어}&limit=50&offset={N}&loungeId=nikke&searchOption=COMMUNITY_FEED_TITLE&orderType=LATEST`
(`COMMUNITY_FEED_TITLE_CONTENT`면 본문까지). 응답에 `totalCount`·제목·작성일·미리보기가 있다. 검색어 하나에 최근 2,000건까지만 나온다 (「솔레」 3,025건 중 2,000건).

### 본문
- `GET /community/lounge/nikke/feed/{feedId}`
- `content.feed.contents`: 본문 HTML (SmartEditor)
- `buff` / `nerf`: 추천 / 비추천 수
- `createdDate`: `YYYYMMDDHHMMSS`

### 이미지
- `https://nng-phinf.pstatic.net/...?type=w1678` (본문 HTML의 `<img>`, `repImageUrl`)
- Referer 없이 다운로드됨 (재확인 2026-10-07)
- 2026-10-15 확인: 자기 이름 UA로 8장 모두 받아졌다. 다만 연결이 응답 없이 끊기는 경우(상태 0)가 2번 있었고 다시 요청하면 받아졌다 → 재시도를 둔다. `nng-phinf.pstatic.net/robots.txt`는 없음(404)

### 동영상
⛔ 재생 정보 API가 `apis.naver.com`(robots.txt 전체 차단)이라 쓰지 않는다. 아래는 첫 조사 기록.

목록·본문의 `iconTypes`에 `VIDEO`가 있으면 네이버에 직접 올린 동영상, `SNIPPET`이 있으면 링크 미리보기(YouTube 등)가 들어 있다.

**1) 네이버에 올린 동영상 (`se-video`)**
- 본문 HTML의 `<script class="__se_module_data" data-module='{"type":"v2_video", ...}'>` 안에 `vid`, `inkey`, `thumbnail`이 있음
- 재생 정보: `GET https://apis.naver.com/rmcnmv/rmcnmv/vod/play/v2.0/{vid}?key={inkey}`
  - `videos.list[]`: 화질별 mp4 (`encodingOption.name` = `270P`/`480P`/`720P`, `source` = mp4 주소, `size` = 바이트)
  - `streams[]`: HLS 주소
  - `expireTime`: 호출 시각 + 30분
- mp4 주소(`*.akamaized.net/...`)는 `__gda__` 서명이 붙어 **30분 뒤 만료**. Referer 없이 받아짐 (Range 요청 지원)
- 썸네일: `https://video-phinf.pstatic.net/...` (Referer 없이 받아짐). 썸네일은 본문 HTML에 있으니 robots.txt와 상관없이 기록할 수 있다

**2) 외부 링크 임베드 (`se-oembed`)**
- `data-module`의 `"type":"v2_oembed"` 안에 `inputUrl`(원래 링크), `providerName`(예: `YouTube`), `title`, `thumbnailUrl`이 있음 → 링크와 메타정보만 기록. 본문 HTML 안에 있으므로 robots.txt 문제 없음

### 댓글
⛔ `apis.naver.com`(robots.txt 전체 차단)이라 쓰지 않는다. 아래는 첫 조사 기록이고, 재검토 때는 요청하지 않았다. 댓글 수는 목록·본문 응답의 `comment.totalCount`로 알 수 있다.

- `GET https://apis.naver.com/nng_main/nng_comment_api/v1/type/FEED/id/{feedId}/comments?limit=30&offset={0,30,60,…}&orderType=ASC&originalLoungeId=NIKKE_The_Goddess_of_Victory&pagingType=PAGE`
  - `offset`은 건너뛸 개수
  - 더 이상 댓글이 없으면 `content.comments` 키가 빠짐 → 끝
- 항목: `comment.commentId`, `content`, `createdDate`, `replyCount`, `user.userNickname`, `buffNerf`
- 댓글 첨부(이미지·GIF): `comment.attaches[].attachValue`

### 답글
⛔ 댓글과 같은 이유로 쓰지 않는다. 첫 조사 기록:

- `GET https://apis.naver.com/nng_main/nng_comment_api/v1/type/FEED/id/{feedId}/comments/{commentId}/replyComments?originalLoungeId=NIKKE_The_Goddess_of_Victory&offset=0`
- 항목의 `commentType`이 `REPLY`, `parentCommentId`가 부모 댓글

---

## 5. 블라블라링크 (글로벌 공식 커뮤니티)

- 요청: 로그인 없이 JSON API. 전부 `POST`, `Content-Type: application/json` (재확인 2026-10-07)
- robots.txt: 없음. `www.blablalink.com/robots.txt`는 앱 HTML을 돌려주고, `api.blablalink.com/robots.txt`는 404 (재확인 2026-10-07)
- 공통 헤더:
  - `x-channel-type: 2`
  - `x-language: ko` (받고 싶은 번역 언어)
  - `x-common-params: {"game_id":"","area_id":"","source":"pc_web","intl_game_id":"","language":"ko","env":"prod"}`

API 기본 주소: `https://api.blablalink.com/api/ugc/direct/standalonesite/Dynamics`

### 목록
- `POST /GetPostListNew`
  ```json
  {"search_type":0,"plate_id":46,"plate_unique_id":"recommend","order_by":1,"limit":"10","regions":[]}
  ```
- 다음 페이지: 응답의 `data.page_info.next_page_cursor`를 요청 body의 `next_page_cursor`로 전달
- 항목: `post_uuid`, `title`, `content`, `language`, `browse_count`(조회), `comment_count`, `upvote_count`, `created_on`(유닉스 초), `pic_urls`
- `order_by`는 1, 2 모두 동작함 (정확한 의미는 미확인)
- 2026-10-15 확인 (자기 이름 UA):
  - **`"regions":["ko"]`면 한국어 글만 온다.** `["KR"]`은 0개. `"limit":"50"`도 받아진다
  - `regions:["ko"]`·`limit:"50"`으로 커서를 따라가면 `created_on` 내림차순, 중복 없이 11페이지 만에 09-22까지 닿았다 (한국어 글 하루 약 25개) → 기간 수집은 이 목록으로 한다
  - `"limit":"100"`까지 받아지고, `"200"`은 10개만 왔다. 커서는 건너뛸 수 없어 차례로 넘겨야 한다.
    `regions:["ko"]`·`limit:"100"`으로 30페이지(3,000개, 중복 0) 넘기니 2026-07-17까지 닿았다. 끝이 아니라 시험 상한에서 멈췄다 → 석 달 전이면 요청 30번쯤
  - 숫자 필드 일부가 문자열이다 (`comment_count: "0"`). 그대로 참·거짓으로 쓰면 틀린다
- **다국어 혼합:** 일본어·대만 글이 섞여 있고, `x-language: ko`면 제목이 한국어 자동번역으로 옴. `language`로 원래 언어를 구분할 수 있음

2026-10-14 확인: 같은 API 기본 주소의 `POST /SearchPost`로 공개 검색을 할 수 있었다. 요청은 `{"keyword":"솔로레이드","plate_unique_id":"","next_page_cursor":"","limit":10,"regions":[]}` 형태이며, 반환 `list`와 `page_info`를 확인한다. 검색 목록의 번역 제목·본문과 `GetPost`의 `original_content:1` 원문을 구분한다. 검색어 한 개의 결과가 없다고 해당 기간의 자료 전체가 없다고 결론 내리지 않는다.

2026-10-15 확인: 검색 결과는 날짜순이 아니다 (「솔로레이드」 1·2페이지가 2025-06~2026-09에 섞여 있음). `order_by` 1·2를 넣어도 같은 결과였다. 「컷」처럼 짧은 말은 「컷신」「네컷」 등이 섞인다. 기간을 정한 수집에는 쓰기 어렵고 키워드 확인용으로만 쓴다.

### 본문
- `POST /GetPost`
  ```json
  {"post_uuid":"...","browse_post":2,"original_content":0}
  ```
- `original_content: 0` → `x-language`로 번역된 본문, `1` → 원문
- `language` / `original_language`로 원래 언어 확인

### 이미지
- `pic_urls[]` 또는 본문 HTML의 `<img>`
- `https://sg-cdn.blablalink.com/standalonesite/ugc/public/image/{uuid}.{확장자}?height=..&width=..`
- Referer 없이 다운로드됨

### 동영상
- 글의 `type`이 `3`이면 동영상 글이다 (최근 글 120개 중 10개). `1`과 `2`의 차이는 확인하지 못함
- 동영상 정보는 `ext_info`에 **JSON 문자열**로 들어 있음 (배열):
  ```json
  [{"platform":"youtubeshort","video_id":"Fn8rJDujMS4","video_url":"https://www.youtube.com/shorts/Fn8rJDujMS4",
    "video_title":"...","video_desc":"<p>...</p>","video_cover":"https://sg-cdn.blablalink.com/lip/ugc/public/image/....jpg"}]
  ```
  - `platform`: 샘플에서는 `youtube`, `youtubeshort`만 나옴
  - `video_cover`(썸네일)는 Referer 없이 받아짐 (재확인 2026-10-07)
- 샘플 120개 중 블라블라링크에 직접 올린 동영상은 없었고, 본문 HTML에 `<video>`/`<iframe>`이 들어간 글도 없었다 → 영상 자체는 YouTube에 있으므로 링크·ID·제목·썸네일만 기록

### 댓글
- `POST /GetPostCommentsV2`
  ```json
  {"post_uuid":"...","limit":10,"page_type":0,"next_page_cursor":"","previous_page_cursor":"","comment_reply_limit":3,"order_by":2}
  ```
- 응답: `data.list[]` (`comment_uuid`, `content`(HTML), `language`, `created_on`, `upvote_count`, `user`, `replies`), `data.page_info.is_finish` / `next_page_cursor`
- 댓글은 번역되지 않고 원문 그대로 옴
- 항목마다 `user`(업적·아바타 등)가 통째로 붙어 있어 응답이 크다 (댓글 1개 원본 ~1,500토큰)
- 이모티콘: `sg-cdn.blablalink.com/cms/nrft/feeds/pic/...?imgtype=emoji`
- 2026-10-15 확인: 댓글 190개 글에서 10개와 `is_finish: false`가 왔다. 원본 응답은 댓글 1개에 약 2,200바이트. 표본 10개 모두 `replies`가 비어 있었다

### 답글
- `POST /BatchGetPostCommentReplies`
  ```json
  {"comment_uuids":["...","..."],"limit":3,"order_by":2}
  ```
- 응답: `data.replies_map[{댓글ID}].data_list[]`
- 2026-10-15 확인: 댓글 3개를 넣자 그중 하나의 답글 1개가 `replies_map[{댓글ID}]`(`data_list`, `page_info`)로 왔다. 댓글 목록의 `replies`가 비어 있어도 답글은 이 API로 받아야 한다
- 재확인 2026-10-07: 표본 댓글에 답글이 없어서 `replies_map`이 빈 객체로만 왔다. 댓글 항목의 `replies` 필드(표본은 모두 null)에 `comment_reply_limit`만큼 답글이 같이 오는지도 확인하지 못함

### 미확인
- 답글이 `limit`보다 많을 때 다음 페이지 받는 방법
- 댓글 응답의 `replies`와 답글 API의 관계

---

## 그 밖의 사이트 (2026-10-16)

역대 컷 정리 조사에서 쓴 사이트와 지킬 점.

- `cdn.syndication.twimg.com`(X 트윗 JSON)은 robots.txt가 `Disallow: /`다. 이 조사에서 10건을 받은 뒤에야 알았다 — 쓰지 않는다. 트윗은 검색 결과에 나온 글자만 쓴다.
- `www.gamemeca.com`은 robots.txt `Crawl-delay: 30`. 30초 간격을 지킨다 (`trees.gamemeca.com`은 지연 요구 없음).
- note.com(일본 블로그)·`wiki3.jp/nikke`(JP 위키)·티스토리·다모앙은 `Claude-User`로 받아졌다. 연결 끊김은 재시도로 풀렸다.
- WebSearch 도구는 reddit.com·naver.com(블로그·카페) 도메인을 거부했다.

## 재검토 (2026-10-07)

문서의 주소·API를 `User-Agent: Claude-User`로 다시 요청해 확인했다 (robots.txt 포함 요청 약 50회, 간격 1초 이상). 사이트마다 목록 1페이지와 댓글이 달린 최근 글 2~3개를 받았다.

### 적절성 문제 → 고침

1. **네이버 댓글·답글·동영상 API가 robots.txt 차단 주소였다.** 이 API들이 있는 `apis.naver.com`은 robots.txt로 전체를 막는다. 첫 조사는 `game.naver.com`의 robots.txt만 봤다. → ⛔로 표시하고 쓰지 않는 것으로 바꿨다. 재검토에서도 요청하지 않았다.
2. **디시 요청 방법이 공통 원칙과 어긋났다.** "일반 브라우저 UA로 GET"이라고 적혀 있었다. `Claude-User`로도 200이 나와서 그렇게 고쳤다.
3. **`Claude-User`를 쓸 수 있는 경우를 좁혔다.** 이 이름은 Claude가 사용자 요청을 받아 가져올 때의 신원이다. 무인 수집기가 쓰면 신원을 속이는 것이다. 자기 이름의 UA로 시험해 보니 디시·네이버(comm-api)·블라블라링크는 200, 아카라이브는 403이었다. 에펨코리아는 robots.txt가 막고 있어 시험하지 않았다. → 아카라이브·에펨코리아는 사용자가 요청할 때 Claude가 가져오는 용도로만 쓸 수 있다.
4. **robots.txt·AI 학습 원칙을 공통 원칙에 추가했다.**

### 내용 수정

- 디시: 댓글은 페이지당 100개이고 `comment_page=1`이 가장 최근 100개 (첫 조사의 미확인 항목). 지연 로딩 이미지는 `data-original`에 주소가 있음. `img.gif-mp4`는 GIF로 받아짐. Range 요청 불안정
- 아카라이브: 목록 링크에 `?p=1`이 붙어 첫 조사의 선택자로는 안 잡힘. 광고·숨김 공지 행 걸러야 함. 이모티콘만 단 댓글은 `.text`가 없음. 댓글 시각은 UTC
- 에펨코리아: 첫 조사의 목록 선택자가 댓글 수 링크까지 잡아 글이 두 번 나옴
- 네이버: 목록의 `contents`는 HTML이 아니라 SmartEditor JSON
- 블라블라링크: 댓글에 `replies` 필드가 있음
- 수집 비용(토큰) 절 추가

### 그대로 맞았던 것

- 다섯 곳의 목록·본문 주소와 파싱 위치, 디시 댓글 API와 `e_s_n_o`, 댓글돌이(`no`가 0)
- 아카라이브 서명 URL 만료(받은 시각 + 3600초)와 아카콘의 긴 만료
- 에펨코리아 답글 구조(`re` 클래스, `a.findParent`)
- 블라블라링크 헤더와 요청 형식
- 이미지가 Referer 없이 받아지는 것 (디시·아카라이브·에펨코리아·네이버·블라블라 썸네일)

### 다시 확인하지 않은 것

- 동영상: 디시 업로드 영상과 Referer, 아카라이브·에펨코리아 동영상 (표본 글에 없거나 확인 범위 밖)
- 아카라이브 `?cp=`·에펨코리아 `cpage` 댓글 페이지 (표본 글의 댓글이 한 페이지 안이었음)
- 블라블라링크 답글 (표본에 답글이 없었음)
- 각 사이트 이용약관의 자동 수집 조항 — robots.txt만 봤다. 무인 수집기를 만들 거라면 따로 확인해야 한다

---

## 참고: 함께 찾은 오래된 주소

| 주소 | 상태 |
|---|---|
| `gall.dcinside.com/mgallery/board/lists/?id=nikke` | 폐쇄 ("매니저의 요청으로 폐쇄") |
| `arca.live/b/nikke` | 2022년 이후 글 없음, 채널 설명에 `nikketgv`로 안내 |
