"""E3/E3g/E4 prompt edits, questions and answer parsers (extension §3).

Every edit works on a logged prompt (a list of chat messages); the turn
context is the tail of the last message. What a player sees (X1 §2.2):
- dev: `Player B says: <msg>` lines (own `Player A (you) says: ...`), joined
  by newlines and closed by a blank line, then notices and boards;
- final: `MESSAGE BOARD (the last 4 rounds, oldest first):` and lines
  `Player Green said "<msg>"` (whitespace collapsed), closed by a blank line;
  an empty board is left out;
- both end with the response block `Your turn. Respond with: ...`.
"""
import copy
import hashlib
import json
import re

from analysis.messages import corpus, grammar

RESPONSE = "Your turn. Respond with:"
FINAL_HEAD = "MESSAGE BOARD (the last 4 rounds, oldest first):\n"
LEAD = "This turn you answer one question and make no move."


def q_loc(letter):
    return f"{LEAD} Which cell is object {letter} on now? Answer `cell: (Rr,Cc)` or `cell: unknown`."


def q_goal(letter):
    return f"{LEAD} Which cell is the target cell of object {letter}? Answer `cell: (Rr,Cc)` or `cell: unknown`."


def sha(msgs):
    return hashlib.sha256(json.dumps(msgs, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def _tail(msgs, fn):
    """Copy of msgs with fn applied to the last message's content."""
    out = copy.deepcopy(msgs)
    out[-1]["content"] = fn(out[-1]["content"])
    return out


def ask(msgs, question):
    """Replace the response block (from the last `Your turn. Respond with:` to the end) with a question."""
    def f(c):
        i = c.rfind(RESPONSE)
        if i < 0:
            raise ValueError("no response block")
        return c[:i] + question
    return _tail(msgs, f)


def entry(arm, seat, msg):
    """The line a message occupies in a context."""
    return f"{seat} says: {msg}" if arm == "dev" else f'{seat} said "{corpus.norm(msg)}"'


def _span(c, arm, seat, msg):
    line = entry(arm, seat, msg)
    i = c.find(line)
    if i < 0 or c.find(line, i + 1) >= 0:
        raise ValueError(f"entry not found exactly once: {line[:60]!r}")
    return i, i + len(line)


def delete(msgs, arm, seat, msg):
    """Delete one message entry. The only line of a block takes the block's
    closing blank line (and, final arm, the MESSAGE BOARD header) with it."""
    def f(c):
        i, j = _span(c, arm, seat, msg)
        head = FINAL_HEAD if arm == "final" else ""
        first = c[:i].endswith(FINAL_HEAD) if arm == "final" else (i == 0 or c[:i].endswith("\n\n"))
        if first and c[j:].startswith("\n\n"):
            return c[:i - len(head)] + c[j + 2:]
        assert c[j:].startswith("\n"), "an entry line ends with a newline"
        return c[:i] + c[j + 1:]
    return _tail(msgs, f)


def strip_messages(msgs, arm, entries):
    """Every message entry removed, the no-channel arm of E3. dev deletes each
    visible entry [(label, text)], own `(you)` lines included; final drops the
    whole MESSAGE BOARD block. Notices and boards stay."""
    if arm == "final":
        def f(c):
            i = c.find(FINAL_HEAD)
            return c if i < 0 else c[:i] + c[c.index("\n\n", i) + 2:]
        return _tail(msgs, f)
    for seat, msg in entries:
        msgs = delete(msgs, arm, seat, msg)
    return msgs


def replace(msgs, arm, seat, msg, new):
    """Put another message text in the same entry (the swap)."""
    def f(c):
        i, j = _span(c, arm, seat, msg)
        return c[:i] + entry(arm, seat, new) + c[j:]
    return _tail(msgs, f)


def swap_cell(msg, arm, c1, c2):
    """Every G-cell surface form of c1 in msg rewritten to c2, digits only, so
    the text keeps its length and form. Returns (new text, forms replaced)."""
    out, n = list(msg), 0
    for t in grammar.tokenize(msg, arm, grammar.DEFAULT):
        if t.kind == "cell" and t.val == c1:
            nums = iter((str(c2[0] + 1), str(c2[1] + 1)))
            new = re.sub(r"\d+", lambda m: next(nums), t.text)
            assert len(new) == len(t.text), "same digit count on the board"
            out[t.i0:t.i1] = list(new)
            n += 1
    return "".join(out), n


# ---------------------------------------------------------------- answers

THINK_END = "</think>"
CELL_ANS = re.compile(r"cell\s*:\s*[`*\s]*(?:\(?\s*R\s*(\d+)\s*,?\s*C\s*(\d+)\s*\)?|(unknown)\b)", re.I)
_DONE_ONLY = re.compile(r"\A\s*done:\s*(?P<done_reason>.+)", re.I | re.S)  # master.py DONE_ONLY_PATTERN


def answer_text(text):
    """The answer part of a reply: after the last bare `</think>` (Nemotron's
    inline reasoning), else the whole text. reasoning_content is never seen here."""
    return (text or "").rsplit(THINK_END, 1)[-1].strip()


def parse_cell(text):
    """(r, c) zero-based, 'unknown', or None (unparseable). The last answer counts."""
    m = list(CELL_ANS.finditer(answer_text(text)))
    if not m:
        return None
    m = m[-1]
    return "unknown" if m.group(3) else (int(m.group(1)) - 1, int(m.group(2)) - 1)


def parse_move(text):
    """(object, direction), 'done', or None, with the master's own regexes."""
    t = answer_text(text)
    m = corpus._M["COMM_MOVE_PATTERN"].match(t)
    if m:
        return m.group("object").upper(), m.group("direction").lower()
    return "done" if corpus._M["COMM_DONE_PATTERN"].match(t) or _DONE_ONLY.match(t) else None
