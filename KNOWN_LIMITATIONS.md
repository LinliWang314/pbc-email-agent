# Known limitations & unverified assumptions

An honest audit of where the system relies on assumptions I have **not** validated
against real held-out data, plus known gaps. Kept explicit rather than hidden — in a
regulated domain, knowing the boundary matters as much as the capability.

## Verified during audit (real findings)

1. **Free-form PBC parsing depends on the LLM path.** The held-out PBC list is free-form
   prose. The LLM parser handles it correctly (tested: 5/5 prose items → structured), but
   the **regex fallback returns 0 items on free-form** — so a cold run **without** an API
   key would produce an empty tracker. Mitigation: ensure `ANTHROPIC_API_KEY` is set for
   any real run (offline mode is only for UI inspection).

2. **Scanned / image-only PDFs are not OCR'd.** `parse_pdf` reads the text layer only. In
   the sample, `Board_Minutes_2025-09-18.pdf` extracts **0 characters** (it's a scan) — and
   the brief hints held-out has more ("board secretary scanned old ones"). Today the agent
   still classifies these from filename + email body, but cannot cite their contents.
   Proper fix: fall back to rasterize-then-OCR when the text layer is empty (needs
   `pdf2image` + tesseract; not installable on this build machine).

3. **PDF citation verification is uneven.** Excel citations verify reliably (numeric-aware
   matching); PDF citations range 0.0–1.0 depending on extraction quality (0.0 for the
   scanned minutes above). Confidence reflects this honestly rather than overclaiming.

4. **"Under review" vs the scorer.** The brief lists 5 statuses incl. Under review and
   Complete, but the handout ground truth only uses 3 (Not started / Received /
   Insufficient). So any item we route to "Under review" is counted wrong by an exact-match
   scorer on that file. Whether the held-out scorer credits Under review is an open question
   (asked). If not, mapping Under review → Received/Insufficient at output is a one-line change.

## Fixed during this audit

- Concurrent emails raced to lazily build the TF-IDF index — now pre-built before the
  thread pool starts.

## Not validated (no held-out data)

- The status-accuracy fixes (TF-IDF classification, email-direction awareness) are
  validated on the handout sample and a self-built 44-email adversarial stress set, **not**
  on the real held-out mailbox. The stress set is my proxy for held-out shape; it is not the
  real distribution.
- `expected tool-call sequence match` (a brief eval metric) is not implemented.
- Embedding-based matching uses TF-IDF + synonym expansion, not a neural embedding model
  (portability: no compiled deps on the target machine). A neural embedding is a drop-in
  upgrade to `classification._build_index`.

## Deliberate scope choices

- Single in-memory run (one audit at a time) — multi-tenancy is per-engagement state +
  a queue (README covers the 10/100-concurrent path).
- No auth on the deployed app (synthetic data).
