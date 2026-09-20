import sys, time, collections
from glance.config import load_config
cfg = load_config()
from glance.evals.suites import SUITES, SuiteSkipped
for name in sys.argv[1:]:
    t = time.time()
    try:
        items = SUITES[name].build(cfg, cfg.eval.manifest_n)
    except SuiteSkipped as e:
        print(f"{name}: SKIPPED ({e})"); continue
    splits = collections.Counter(i.split for i in items)
    labels = collections.Counter(str(i.label) for i in items)
    print(f"{name}: {len(items)} items in {time.time()-t:.0f}s | splits {dict(splits)} | {len(labels)} labels, top {labels.most_common(4)}")
    print("   e.g.", items[0].item_id, "|", items[0].question["instructions"], "|", items[0].label, "|", items[0].image_path.split('.cache/')[-1])
