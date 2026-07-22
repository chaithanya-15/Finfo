import json, time, sys
from pathlib import Path
from src.data_processing.ingest import extract_document_text

docs = {json.loads(l)["doc_name"]: 1 for l in open("data/financebench_document_information.jsonl")}
names = sorted(docs)
t0 = time.time()
ok = fail = 0
for i, n in enumerate(names, 1):
    p = f"data/raw_pdfs/{n}.pdf"
    if not Path(p).exists():
        print(f"[{i}/{len(names)}] MISSING {n}", flush=True); fail += 1; continue
    t = time.time()
    txt, extra = extract_document_text(p, "pdfplumber")
    if not txt.strip():
        print(f"[{i}/{len(names)}] EMPTY {n}", flush=True); fail += 1; continue
    ok += 1
    print(f"[{i}/{len(names)}] {n:42} {len(txt):8d} chars {len(extra):4d} tables {time.time()-t:5.1f}s", flush=True)
print(f"DONE ok={ok} fail={fail} elapsed={(time.time()-t0)/60:.1f} min", flush=True)
