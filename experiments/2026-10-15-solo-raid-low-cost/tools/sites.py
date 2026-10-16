"""세 사이트 글 번호 도우미 (2026-10-15, 디시·네이버·블라블라링크 묶기).

- 디시 글 번호는 숫자, 네이버는 `n{feedId}`, 블라블라링크는 `b{post_uuid}` 문자열이다 (other_sites.py).
- 파일(nos.txt 등)에서 읽은 번호는 `parse_no`로 바꾼다: 숫자면 int, 아니면 문자열 그대로. 글 JSON·목록 줄의 `no`와 같은 형이 된다.
"""


def parse_no(x):
    x = str(x).strip()
    return int(x) if x.isdigit() else x


def site_of(no):
    s = str(no)
    if s.isdigit():
        return "dc"
    return {"n": "naver", "b": "blabla"}.get(s[:1], "unknown")


def sort_key(no):
    """디시(숫자) 먼저 번호 순, 그다음 문자열."""
    s = str(no)
    return (0, int(s), "") if s.isdigit() else (1, 0, s)
