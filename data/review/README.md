# Hand review of the `unexecuted` bucket

`hand_review_2026-09-11.csv` — the setter's review of every thinking-arm derivation on `nat50`
that was right-but-invalid or `unexecuted` (ANALYSIS_PLAN §2.3: L1 ∧ L3 ∧ ¬L2a is hand-reviewed).
53 rows, keyed by clue id and system; one row per (id, system) since the arms ran at k = 1.
No clue text and no answers: those are the setter's and stay out of the repository.

Classes, as proposed by Claude on 2026-09-11 and **agreed by the setter in every case the same
day**:

- **A** — correct parse; the checker's grammar or vocabulary cannot express it (nested container,
  anagram chained into a charade, deletion of a named letter, multi-word selection phrase, two
  indicators in one field, &lit untagged, possessive trimmed from a span). 34 rows.
- **B** — correct parse; the model misused the template's syntax (an identity "anagram", a
  definition span swallowing a fodder word, a non-syntax deletion). 5 rows. Stays invalid.
- **C** — wrong parse; the derivation does not produce the answer. 14 rows. Stays invalid.

Two rows were proposed as split (A/C, A/B) and resolved A; the `note` column says which.

`analysis/metrics.py` reads this file into the `reviewed_validity` table: class A rows are
re-scored valid, giving a second, reviewed validity and M2 beside the mechanical ones. The
frozen tables are untouched; the mechanical numbers remain primary.

# Blind adjudication of answer strings

`adjudication_2026-09-11.csv` — the setter's rulings on distinct wrong-by-exact-match answer
strings (PLAN §2.4), keyed by clue id and the string as extracted (A–Z only). `accept` is `Y`
for a string the setter accepts as a correct answer to the clue as set, `N` otherwise. Only
ruled strings appear; a string not in the file is unadjudicated and scores as exact match does.
Contains model outputs, not the setter's clues or answers.

**Complete, 2026-09-11.** The setter graded the seven strings one letter off a gold answer
first (all **N**), then the 39 other priority strings (all **N**), then the long tail: **no
wrong-by-exact-match string is accepted.** Every distinct wrong string from the `nat` and `nat50`
arms is in the file with `accept = N` (909 rows), so the adjudicated accuracy equals the
exact-match accuracy on every system. `metrics.py` reads the file into the
`adjudicated_accuracy` table, which shows that, and would credit any string later changed to Y.
