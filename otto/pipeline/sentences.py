"""Text plumbing: sentence segmentation for streaming TTS + speech clean-up."""
from __future__ import annotations

import re

# abbreviations that end with a period but do not end a sentence
_ABBREV = {
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "vs", "etc", "inc", "ltd",
    "co", "corp", "dept", "est", "approx", "fig", "no", "vol", "p", "pp", "al",
    "e.g", "i.e", "a.m", "p.m", "u.s", "u.k", "am", "pm", "min", "max", "sec",
    "avenue", "ave", "blvd", "rd", "apt",
}

_CLAUSE_CHARS = ",;:—–-"
_HARD_TERMINATORS = ".!?…\n"

_MD_LINK = re.compile(r"\[([^\]]+)\]\((?:[^)]+)\)")
_MD_IMAGE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_BARE_URL = re.compile(r"https?://\S+")
_MD_TOKENS = re.compile(r"[*_`#>~]+")
_BULLET = re.compile(r"^\s*(?:[-–•]|\d+[.)])\s+", re.MULTILINE)
_STAGE = re.compile(r"\((?:laughs?|sighs?|chuckles?|smiles?|pauses?|beat|clears? throat)\)", re.I)
_BRACKET_STAGE = re.compile(r"\[(?:laughs?|sighs?|chuckles?|smiles?|pauses?|beat|music|applause)[^\]]*\]", re.I)
_EMOJI = re.compile(
    "[" "\U0001F000-\U0001FAFF" "\U00002600-\U000027BF" "\U0001F1E6-\U0001F1FF" "\U0000FE0F" "\U00002190-\U000021FF" "]"
)
_SPACES = re.compile(r"[ \t\xa0]+")
_MULTI_NEWLINE = re.compile(r"\n{2,}")

_REPLACEMENTS = {
    "&": " and ",
    "->": " to ",
    "→": " to ",
    "≈": " about ",
    "<=": " less than or equal to ",
    ">=": " greater than or equal to ",
    "—": ", ",
    "–": ", ",
}


def clean_for_speech(text: str) -> str:
    """Turn model output into something a TTS engine should actually say."""
    if not text:
        return ""
    out = _MD_IMAGE.sub(" ", text)
    out = _MD_LINK.sub(r"\1", out)
    out = _BARE_URL.sub(" ", out)
    out = _BRACKET_STAGE.sub(" ", out)
    out = _MD_TOKENS.sub("", out)
    out = _BULLET.sub("", out)
    out = _STAGE.sub(" ", out)
    out = _EMOJI.sub("", out)
    for src, dst in _REPLACEMENTS.items():
        out = out.replace(src, dst)
    out = _MULTI_NEWLINE.sub("\n", out)
    out = _SPACES.sub(" ", out)
    out = "\n".join(line.strip() for line in out.split("\n")).strip()
    if out and not re.search(r"\w", out):
        return ""
    return out


class SentenceBuffer:
    """Accumulates streamed tokens and yields speakable chunks early.

    The first chunk is cut aggressively (comma if available, or a short word
    boundary) so the very first audio byte lands quickly; later chunks wait for
    full sentences.
    """

    def __init__(self, first_max_chars: int = 60, max_chars: int = 220, min_chars: int = 0) -> None:
        self.first_max_chars = max(20, first_max_chars)
        self.max_chars = max(self.first_max_chars, max_chars)
        #: don't hand a 3-word fragment to TTS on its own — let the next sentence
        #: join it, so the voice keeps its prosody across the boundary
        self.min_chars = max(0, min_chars)
        self.buf = ""
        self.emitted = 0

    # ------------------------------------------------------------------ public
    def feed(self, delta: str) -> list[str]:
        if delta:
            self.buf += delta
        chunks: list[str] = []
        while True:
            cut = self._find_cut()
            if cut is None:
                break
            chunk = clean_for_speech(self.buf[:cut])
            self.buf = self.buf[cut:].lstrip()
            if chunk:
                chunks.append(chunk)
                self.emitted += 1
        return chunks

    def flush(self) -> list[str]:
        chunk = clean_for_speech(self.buf)
        self.buf = ""
        if chunk:
            self.emitted += 1
            return [chunk]
        return []

    @property
    def pending(self) -> str:
        return self.buf

    # ------------------------------------------------------------------ internals
    def _soft_limit(self) -> int:
        return self.first_max_chars if self.emitted == 0 else int(self.max_chars * 0.75)

    def _find_cut(self) -> int | None:
        buf = self.buf
        if not buf.strip():
            return None
        limit = self.first_max_chars if self.emitted == 0 else self.max_chars

        # 1) real sentence terminator
        for i, ch in enumerate(buf):
            if ch == "\n":
                if buf[:i].strip() and (len(buf[:i].strip()) >= self.min_chars or len(buf) > self.max_chars):
                    return i + 1
                continue
            if ch not in ".!?…":
                continue
            nxt = buf[i + 1] if i + 1 < len(buf) else None
            if nxt is not None and nxt not in " \n\t\"'”’)]}":
                continue  # mid-token punctuation ("3.5", "a.m.", "end.Next")
            if ch == ".":
                prev = buf[i - 1] if i > 0 else ""
                after = nxt or ""
                if prev.isdigit() and after.isdigit():
                    continue
                word = re.split(r"[\s(\[\"']", buf[:i].strip())[-1].lower().strip(".")
                if word in _ABBREV or (len(word) == 1 and word.isalpha()):
                    continue
                if nxt is None and len(buf.strip()) < self._soft_limit():
                    # sentence-final period at the edge of the stream: wait for
                    # one more token before committing
                    continue
            if len(buf[: i + 1].strip()) >= self.min_chars or len(buf) > self.max_chars:
                return i + 1
            continue

        # 2) clause boundary once we have enough text for the first chunk
        if len(buf) >= self._soft_limit():
            window = buf[: max(self._soft_limit(), limit)]
            for i in range(len(window) - 1, 0, -1):
                if window[i] in _CLAUSE_CHARS and len(window[:i].strip()) >= max(12, self.min_chars):
                    return i + 1
            # 3) hard wrap on a word boundary
            if len(buf) > limit:
                space = buf.rfind(" ", 0, limit)
                if space > 10:
                    return space + 1
                return limit
        return None
