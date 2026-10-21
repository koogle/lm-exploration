# EU corpus discovery

Read README.md and sources.json before working on a source. This is an independent
uv project; run commands from this directory. It does not change the training UI
dataset registry or share the latent-workspace environment.

Discovery agents own one assigned collection at a time. Follow linked departments,
groups, meetings and attachments; look for public bulk inventories, feeds and
native full text. Record tested URLs, UTC observation times, HTTP outcomes and
unresolved gaps. A registry record or metadata response is not full document text.
Use bounded requests and respect per-host limits and robots exclusions. Preserve
negative findings rather than guessing endpoints or treating an empty crawl as
proof that a collection has no data.

Write source findings to research/<source-id>.md and research/<source-id>.json.
Keep downloaded bodies, extracted documents, Parquet and large manifests under
ignored artifacts/. Source reports and reproducible code belong in Git; bulk
corpus content does not. Retain source notices as metadata during discovery.
Do not infer publication eligibility from public access.

Prefer original XML/HTML/TXT, then document containers and text-bearing PDFs.
Do not rewrite source text with an LLM. Distinguish document language from UI
language, original speech from interpretation, and official text from ASR.
Preserve document-language-version relationships and exact retrieval provenance.

No collector is production-ready solely because one sample succeeds. State which
pagination, historic coverage, updates and extraction paths were actually tested.
Do not deploy schedules, launch unbounded crawls, or publish a corpus as a side
effect of source investigation. Parent integration owns commits and pushes.
