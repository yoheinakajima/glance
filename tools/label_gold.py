"""A ten-minute labeling page for your own photos: the clean holdout the public benchmarks cannot give us.

Put 50 to 100 photos that were never online into gold/photos/ (any subjects; jpg, png, webp, heic is NOT supported),
then run this and open the address it prints. Every answer is appended to gold/human_gold.jsonl in the format the
`human_gold` suite reads. Everything stays on this machine: the server binds to 127.0.0.1, `gold/` is gitignored, and
the frontier baseline refuses these images unless you pass --allow-upload-gold.

uv run python tools/label_gold.py            # then open http://127.0.0.1:8078
uv run glance eval --suite human_gold --model vlm --model siglip --prefix-cache
"""
import argparse
import json
import pathlib

from flask import Flask, abort, jsonify, request, send_file

ROOT = pathlib.Path(__file__).resolve().parent.parent
SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}

# Objective questions a person can answer at a glance about any photo. Keys: keyboard shortcuts on the page.
QUESTIONS = [
    {"id": "person", "type": "noul", "instructions": "Is there at least one person visible in `img0`?"},
    {"id": "indoors", "type": "noul", "instructions": "Was `img0` taken indoors?"},
    {"id": "text", "type": "noul", "instructions": "Is there any readable text in `img0`?"},
    {"id": "subject", "type": "choice", "instructions": "What is the main subject of `img0`?",
     "criteria": {"person": "One or more people", "animal": "An animal", "food": "Food or drink",
                  "document_or_screen": "A document, sign, or screen", "object": "An object or product",
                  "scene": "A landscape, building, room, or street scene", "other": None}},
    {"id": "light", "type": "choice", "instructions": "What is the lighting in `img0`?",
     "criteria": {"daylight": "Natural daylight", "artificial": "Artificial indoor lighting", "low_light": "Night or low light"}},
]

PAGE = """<!doctype html><meta charset=utf-8><title>gold labels</title>
<style>body{font:16px system-ui;margin:24px;background:#f6f6f4;color:#222}img{max-width:min(900px,95vw);max-height:60vh;border-radius:8px}
button{font:inherit;margin:4px 6px 4px 0;padding:8px 14px;border-radius:8px;border:1px solid #999;background:#fff;cursor:pointer}
button:hover{background:#e9e9e4}#q{font-weight:600;margin:14px 0 6px}#meta{color:#666;font-size:13px;margin-top:10px}</style>
<div id=app>loading...</div>
<script>
let state=null;
async function load(){state=await (await fetch('/next')).json();render()}
function render(){const a=document.getElementById('app');
 if(state.done){a.innerHTML='<h2>All done. '+state.labeled+' answers saved to gold/human_gold.jsonl.</h2>';return}
 const q=state.question;let opts=q.type==='noul'?[['true','Yes (y)'],['false','No (n)']]:Object.keys(q.criteria).map((k,i)=>[k,(i+1)+'. '+k.replaceAll('_',' ')]);
 a.innerHTML='<img src="/image?path='+encodeURIComponent(state.image)+'"><div id=q>'+q.instructions.replaceAll('`img0`','this photo')+'</div>'+
  opts.map(o=>'<button onclick="answer(\\''+o[0]+'\\')">'+o[1]+'</button>').join('')+'<button onclick="answer(\\'__skip__\\')">skip (s)</button>'+
  '<div id=meta>'+state.progress+' &middot; '+state.image+'</div>'}
async function answer(v){await fetch('/answer',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify({image:state.image,qid:state.question.id,value:v})});load()}
document.addEventListener('keydown',e=>{if(!state||state.done)return;const q=state.question;
 if(e.key==='s')return answer('__skip__');
 if(q.type==='noul'){if(e.key==='y')answer('true');if(e.key==='n')answer('false')}
 else{const k=Object.keys(q.criteria)[parseInt(e.key)-1];if(k)answer(k)}});
load();
</script>"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--photos", default="gold/photos")
    parser.add_argument("--out", default="gold/human_gold.jsonl")
    parser.add_argument("--port", type=int, default=8078)
    args = parser.parse_args()
    photos_dir, out_path = ROOT / args.photos, ROOT / args.out
    photos_dir.mkdir(parents=True, exist_ok=True)
    skipped: set[tuple[str, str]] = set()

    def photos() -> list[str]:
        return sorted(str(p.relative_to(ROOT)) for p in photos_dir.rglob("*") if p.suffix.lower() in SUFFIXES)

    def done() -> set[tuple[str, str]]:
        rows = [json.loads(line) for line in out_path.read_text().splitlines() if line.strip()] if out_path.exists() else []
        return {(r["image"], r.get("qid", "")) for r in rows} | skipped

    app = Flask(__name__)

    @app.get("/")
    def index():
        return PAGE

    @app.get("/next")
    def next_item():
        finished, todo = done(), [(img, q) for img in photos() for q in QUESTIONS]
        remaining = [(img, q) for img, q in todo if (img, q["id"]) not in finished]
        if not remaining:
            return jsonify({"done": True, "labeled": len(finished - skipped)})
        img, q = remaining[0]
        return jsonify({"done": False, "image": img, "question": q, "progress": f"{len(todo) - len(remaining) + 1} of {len(todo)}"})

    @app.get("/image")
    def image():
        path = (ROOT / request.args["path"]).resolve()
        if photos_dir.resolve() not in path.parents or path.suffix.lower() not in SUFFIXES:
            abort(404)
        return send_file(path)

    @app.post("/answer")
    def answer():
        body = request.get_json()
        q = next(q for q in QUESTIONS if q["id"] == body["qid"])
        if body["value"] == "__skip__":
            skipped.add((body["image"], q["id"]))
            return jsonify({"ok": True})
        label = (body["value"] == "true") if q["type"] == "noul" else body["value"]
        if q["type"] == "choice" and label not in q["criteria"]:
            abort(400)
        question = {k: v for k, v in q.items() if k != "id"}
        with open(out_path, "a") as f:
            f.write(json.dumps({"image": body["image"], "qid": q["id"], "question": question, "label": label, "annotators": {"owner": label}}) + "\n")
        return jsonify({"ok": True})

    n = len(photos())
    print(f"{n} photos in {photos_dir.relative_to(ROOT)}/ x {len(QUESTIONS)} questions. Open http://127.0.0.1:{args.port}  (Ctrl-C to stop; progress is saved)")
    app.run(host="127.0.0.1", port=args.port, debug=False)


if __name__ == "__main__":
    main()
