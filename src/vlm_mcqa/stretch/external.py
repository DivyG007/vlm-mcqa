"""Guarded IconQA ingestion and built-in dataset-neutral behavior/lens/head study.

No model downloads or imports at module import. All new runtime source is unrun.
External text-choice tasks have no semantic counterfactual image pairs: head
transfer here is same-image position transfer, never semantic-image evidence.
"""
from __future__ import annotations

from collections import defaultdict, OrderedDict
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

from .task_runtime import canonical_hash, grouped_effects, execution_identity
from .protocol import load_frozen_heads


class _ExternalSession:
    """Read-only verified dataset view sharing the calibrated model, no output dir."""
    def __init__(self, parent: Any, root: str | Path, *, max_tokens: int):
        self.adapter = parent.adapter
        self.calibration_data_root = parent.data_dir.resolve()
        if type(max_tokens) is not int or max_tokens < 1:
            raise ValueError("explicit positive external max_tokens required")
        self.max_tokens = max_tokens
        self.data_dir = Path(root).resolve()
        self.manifest = json.loads((self.data_dir / "dataset_manifest.json").read_text())
        release = json.loads((self.data_dir / "release_manifest.json").read_text())
        hashes = self.manifest["sha256"]
        if hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()[:16] != self.manifest["dataset_hash"]:
            raise ValueError("external dataset inventory hash mismatch")
        for name, expected in hashes.items():
            path = (self.data_dir / name).resolve()
            if self.data_dir not in path.parents or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise ValueError("external metadata changed")
        if release.get("image_aware_release_hash") != canonical_hash({k: v for k, v in release.items() if k not in {"errors", "image_aware_release_hash"}}) or release.get("errors"):
            raise ValueError("external image-aware release invalid")
        for name, expected in release["final_image_sha256"].items():
            path = (self.data_dir / "images" / name).resolve()
            if self.data_dir not in path.parents or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise ValueError("external image changed")
        self.items = {r["item_id"]: r for r in self._read("items.jsonl")}
        self.variants = {r["variant_id"]: r for r in self._read("variants.jsonl")}
        scene_splits = {}
        for row in [*self.items.values(), *self.variants.values()]:
            scene = row["scene_id"]
            if scene in scene_splits and scene_splits[scene] != row["split"]:
                raise ValueError("external source image crosses splits")
            scene_splits[scene] = row["split"]
            path = Path(row["image"])
            if not path.parts or path.parts[0] != "images" or ".." in path.parts or str(Path(*path.parts[1:])).replace("\\", "/") not in release["final_image_sha256"]:
                raise ValueError("external referenced image lacks verified inventory")
        self._prepared = OrderedDict()

    def _read(self, name):
        rows = [json.loads(line) for line in (self.data_dir / name).read_text().splitlines() if line.strip()]
        identity = "item_id" if name == "items.jsonl" else "variant_id"
        if len({r[identity] for r in rows}) != len(rows):
            raise ValueError("duplicate external metadata identity")
        return rows

    def items_in(self, split):
        return sorted((v for v in self.items.values() if v["split"] == split), key=lambda v: v["item_id"])

    def variants_of(self, item_id, *, schemes):
        return sorted((v for v in self.variants.values() if v["item_id"] == item_id and v["label_scheme"] in schemes
                       and v["variant_kind"] == "position"), key=lambda v: v["variant_id"])

    def image(self, relative):
        from PIL import Image
        path = (self.data_dir / relative).resolve()
        if self.data_dir not in path.parents:
            raise ValueError("external image outside target dataset")
        with Image.open(path) as loaded:
            return loaded.convert("RGB")

    def prepared(self, variant_id):
        if variant_id not in self._prepared:
            variant = self.variants[variant_id]
            self._prepared[variant_id] = self.prepare_image(variant, self.image(variant["image"]))
            while len(self._prepared) > 4:
                self._prepared.popitem(last=False)
        return self._prepared[variant_id]

    def prepare_image(self, variant, image):
        inputs, rendered = self.adapter.encode(image, variant["prompt"])
        if inputs["input_ids"].shape[1] > self.max_tokens:
            raise ValueError("external input exceeds explicit token budget before model forward")
        return self.adapter.prepare_inputs(variant, inputs, rendered)

    def label_ids(self, prepared, labels):
        ids = [prepared.label_ids[label] for label in labels]
        if any(value is None for value in ids) or len(set(ids)) != 4:
            raise ValueError("external labels are not distinct single-token continuations")
        return ids


def _view(session, config, context):
    if context.get("execute") is not True:
        raise PermissionError("external dataset/model preparation requires explicit execute=True")
    execution_identity(session, context)
    return _ExternalSession(session, config["target_dataset_path"], max_tokens=config["protocol"]["max_tokens"])


def import_iconqa(root: str | Path, output: str | Path, *, source_split: str,
                  target_items: int, seed: int, dataset_id: str,
                  phase_counts: Mapping[str, int]) -> dict:
    """Exclusive derivative import; all choices/truth validated before writing.

    Official choose_txt records only. Source image directory is the clustering
    unit; question/template grouping absent upstream is not invented here.
    """
    source, out = Path(root).resolve(), Path(output).resolve()
    if type(target_items) is not int or target_items < 1 or type(seed) is not int or not dataset_id:
        raise ValueError("explicit positive item count, seed and dataset identity required")
    if (set(phase_counts) != {"discovery", "confirmation"}
            or any(type(n) is not int or n < 1 for n in phase_counts.values())
            or sum(phase_counts.values()) != target_items):
        raise ValueError("explicit positive discovery/confirmation image-unit quotas must sum to target_items")
    if not isinstance(source_split, str) or not source_split or Path(source_split).name != source_split:
        raise ValueError("source split must be a single directory component")
    if out.exists() or source == out or source in out.parents or out in source.parents:
        raise ValueError("new output outside immutable IconQA source required")
    candidates, rejected = [], []
    for directory in sorted((source / source_split / "choose_txt").iterdir()):
        if not directory.is_dir():
            continue
        metadata_path, image_path = directory / "data.json", directory / "image.png"
        raw = metadata_path.read_bytes()
        record = json.loads(raw)
        options = record.get("choices")
        answer = record.get("answer")
        if isinstance(answer, str) and answer.isdigit():
            answer = int(answer)
        if (not isinstance(options, list) or len(options) != 4
                or any(not isinstance(x, str) or not x.strip() for x in options)
                or len(set(x.strip() for x in options)) != 4
                or type(answer) is not int or not 0 <= answer < 4
                or not isinstance(record.get("question"), str) or not record["question"].strip()):
            rejected.append(directory.name)
            continue
        image_hash = hashlib.sha256(image_path.read_bytes()).hexdigest()
        candidates.append((directory.name, record, answer, raw, image_path, image_hash))
    candidates.sort(key=lambda x: canonical_hash([seed, x[0]]))
    unique_images = {}
    for candidate in candidates:
        unique_images.setdefault(candidate[5], candidate)
    candidates = list(unique_images.values())
    if len(candidates) < target_items:
        raise ValueError("insufficient eligible external items; target not silently reduced")
    selected = candidates[:target_items]
    out.mkdir(parents=True, exist_ok=False)
    (out / "images").mkdir()
    items, provenance = [], []
    for index, (qid, record, answer, raw, image_path, image_hash) in enumerate(selected):
        image = f"images/iconqa_{qid}.png"
        with (out / image).open("xb") as stream:
            stream.write(image_path.read_bytes())
        options = [x.strip() for x in record["choices"]]
        items.append({"item_id": f"iconqa_{qid}", "source": "iconqa", "dataset_version": dataset_id,
                      "split": "discovery" if index < phase_counts["discovery"] else "confirmation", "scene_id": "iconqa_" + image_hash,
                      "counterfactual_pair_id": None, "pair_member": None, "image": image,
                      "question": record["question"].strip(), "option_contents": options,
                      "semantic_answer": options[answer], "answer_type": "text",
                      "question_family": record.get("ques_type", "choose_txt"), "program": [], "program_depth": 0})
        provenance.append({"source_question_id": qid, "image_sha256": image_hash,
                           "metadata_sha256": hashlib.sha256(raw).hexdigest()})
    # Historical expand writes only inside this newly created derivative tree.
    from ..clevr_mcq4.build import expand
    manifest = expand(out, items, seed=seed)
    release = {"schema": "stretch.external-release.v1", "dataset_hash": manifest["dataset_hash"],
               "final_image_sha256": {Path(item["image"]).name: record["image_sha256"] for item, record in zip(items, provenance)},
               "source_split": source_split, "phase_counts": dict(phase_counts)}
    release["image_aware_release_hash"] = canonical_hash(release)
    with (out / "release_manifest.json").open("x", encoding="utf-8") as stream:
        json.dump(release, stream, sort_keys=True, indent=2)
    report = {"schema": "stretch.iconqa-import.v1", "dataset_id": dataset_id,
              "source_split": source_split, "target_items": target_items, "seed": seed,
              "phase_counts": dict(phase_counts), "release": release,
              "selected": len(selected), "rejected_question_ids": rejected,
              "source_records": provenance, "manifest": manifest,
              "capabilities": ["behavior", "label_lens", "position_head_transfer"],
              "semantic_counterfactual_pairs": False}
    report["import_sha256"] = canonical_hash(report)
    with (out / "external_import.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    return report


def _configuration(session: Any, config: Mapping, context: Mapping) -> tuple[dict, list[dict], dict | None]:
    if context.get("execute") is not True:
        raise PermissionError("external study requires explicit execute=True")
    if config.get("name", config.get("study")) != "external":
        raise ValueError("external study selector required")
    if config.get("model_key") != session.adapter.spec.key or session.adapter.spec.family not in {"qwen_vl", "paligemma", "internvl", "llava_onevision"}:
        raise ValueError("explicit supported calibrated Core-7 model required")
    core = context.get("core_calibration", {})
    required = {"cached_matches_full", "capture_is_inert", "identity_patch", "batched_matches_single",
                "replay_deterministic", "residual_decomposition", "lens_final_matches", "symbols_single_token", "token_spans"}
    if (core.get("all_passed") is not True or core.get("adapter") != session.adapter.describe()
            or any(core.get("gates", {}).get(k, {}).get("passed") is not True for k in required)):
        raise ValueError("all real Core-7 calibration gates for current adapter required")
    p = dict(config["protocol"])
    for key in ("target_units", "minimum_units", "calibration_variants", "bootstrap_samples"):
        if type(p.get(key)) is not int or p[key] < 1:
            raise ValueError(f"explicit positive {key} required")
    if p["minimum_units"] > p["target_units"] or type(p.get("seed")) is not int:
        raise ValueError("invalid target/minimum/seed")
    if p["bootstrap_samples"] < 100:
        raise ValueError("at least 100 bootstrap draws required before external inference")
    if type(p.get("head_transfer")) is not bool:
        raise ValueError("explicit head_transfer capability choice required")
    layers = p.get("lens_layers", [])
    if not layers or len(set(layers)) != len(layers) or any(type(l) is not int or not 0 <= l < session.adapter.num_layers for l in layers):
        raise ValueError("explicit distinct lens layers required")
    for key in ("tolerance", "minimum_accuracy", "minimum_image_gap"):
        if type(p.get(key)) not in (float, int) or not math.isfinite(p[key]) or p[key] < 0:
            raise ValueError(f"finite nonnegative {key} required")
    if p["minimum_accuracy"] > 1 or not isinstance(p.get("schemes"), list) or not p["schemes"]:
        raise ValueError("invalid behavioral threshold or empty schemes")
    if config.get("phase") not in {"discovery", "confirmation"} or not config.get("split"):
        raise ValueError("explicit phase and dataset split required")
    items = session.items_in(config["split"])
    by_unit = defaultdict(list)
    for item in items:
        by_unit[item["scene_id"]].append(item)
    units = sorted(by_unit, key=lambda u: canonical_hash([p["seed"], u]))[:p["target_units"]]
    selected = [v for unit in units for item in sorted(by_unit[unit], key=lambda i: i["item_id"])
                for v in session.variants_of(item["item_id"], schemes=p["schemes"])]
    if not selected:
        raise ValueError("no external variants match explicitly requested split/schemes")
    frozen = None
    if p["head_transfer"]:
        if any(type(p.get(k)) is not int or p[k] < 1 for k in ("random_head_sets", "minimum_correct_transfer_units")):
            raise ValueError("head transfer requires explicit random controls and minimum correct transfer units")
        frozen = load_frozen_heads(config["head_freeze_path"], model_key=config["model_key"], dataset_hash=config["head_freeze_dataset_hash"])
        p["selected_heads"] = frozen["head_candidates"][:4]
        p["head_source_sha256"] = frozen["source_sha256"]
    p["model_key"] = config["model_key"]
    p["external_dataset_hash"] = session.manifest["dataset_hash"]
    return p, selected, frozen


def calibrate_study(session: Any, config: Mapping, context: Mapping) -> dict:
    session = _view(session, config, context)
    p, variants, frozen = _configuration(session, config, context)
    # Calibration variants always come from discovery, never confirmation.
    _, calibration_variants, _ = _configuration(session, {**config, "phase": "discovery", "split": "discovery"}, context)
    from ..core7.hooks import Site, Intervention
    checks = {}
    checked = []
    if len(calibration_variants) < p["calibration_variants"]:
        raise ValueError("insufficient discovery-only external calibration variants")
    for i, variant in enumerate(calibration_variants[:p["calibration_variants"]]):
        prepared = session.prepared(variant["variant_id"])
        logits, _ = session.adapter.decoder_forward(prepared)
        error = float((logits[0].cpu() - prepared.clean_logits).abs().max().item())
        checks[f"external_replay_{i}"] = {"passed": error <= p["tolerance"], "measured": error, "threshold": p["tolerance"]}
        for layer in p["lens_layers"]:
            site = Site("resid_post", layer)
            _, capture = session.adapter.decoder_forward(prepared, capture=[site], capture_positions=[prepared.final_position])
            identity, _ = session.adapter.decoder_forward(prepared, interventions=[Intervention(site, [prepared.final_position], capture[site.key][0])])
            error = float((identity - logits).abs().max().item())
            checks[f"external_identity_{i}_{layer}"] = {"passed": error <= p["tolerance"], "measured": error, "threshold": p["tolerance"]}
        checked.append(variant["variant_id"])
    return {"schema": "stretch.external-calibration.v1", "study": "external", "checks": checks,
            "all_passed": bool(checks) and all(v["passed"] for v in checks.values()),
            "adapter": session.adapter.describe(), "protocol_sha256": canonical_hash(p),
            "dataset_hash": session.manifest.get("dataset_hash"), "checked_units": checked,
            "execution_identity_sha256": execution_identity(session, context),
            "core_calibration_sha256": canonical_hash(context["core_calibration"])}


def run_study(session: Any, config: Mapping, context: Mapping) -> dict:
    session = _view(session, config, context)
    p, variants, frozen = _configuration(session, config, context)
    calibration = context.get("extension_calibration", {})
    if (calibration.get("all_passed") is not True or calibration.get("adapter") != session.adapter.describe()
            or calibration.get("protocol_sha256") != canonical_hash(p)
            or calibration.get("dataset_hash") != session.manifest.get("dataset_hash")
            or calibration.get("core_calibration_sha256") != canonical_hash(context["core_calibration"])):
        raise ValueError("matching measured external calibration required")
    if calibration.get("execution_identity_sha256") != execution_identity(session, context):
        raise ValueError("external calibration execution identity changed")
    units = sorted({v["scene_id"] for v in variants})
    if config["phase"] == "confirmation":
        freeze = json.loads(Path(config["freeze_path"]).read_text(encoding="utf-8"))
        if freeze.get("freeze_sha256") != canonical_hash({k: v for k, v in freeze.items() if k != "freeze_sha256"}):
            raise ValueError("external freeze content hash mismatch")
        if freeze.get("protocol_sha256") != canonical_hash(p) or set(freeze.get("discovery_units", [])) & set(units):
            raise ValueError("external confirmation freeze mismatch or reused source images")
    out = Path(context["output_dir"]).resolve()
    protected = [session.data_dir.resolve(), session.calibration_data_root]
    if out.exists() or any(root == out or root in out.parents or out in root.parents for root in protected):
        raise ValueError("new external attempt outside dataset required")
    out.mkdir(parents=True, exist_ok=False)
    from ..core7.scoring import symbol_readout
    from ..core7.hooks import Site, Intervention
    from PIL import Image
    # Behavioral eligibility precedes observational lenses and causal patches.
    preflight_rows, preflight_missing = [], []
    for v in variants:
        prepared = session.prepared(v["variant_id"])
        clean = symbol_readout(prepared.clean_logits, session.label_ids(prepared, v["labels"]), v["correct_index"])
        image = session.image(v["image"])
        candidates = [w for w in variants if w["scene_id"] != v["scene_id"] and w["semantic_answer"] != v["semantic_answer"]]
        row = {"variant_id": v["variant_id"], "scene_id": v["scene_id"], "clean": clean}
        if not candidates:
            preflight_missing.append(v["variant_id"])
        else:
            donor = sorted(candidates, key=lambda w: canonical_hash([p["seed"], v["variant_id"], w["variant_id"]]))[0]
            for op, changed in (("blank", Image.new("RGB", image.size, "white")),
                                ("shuffled", session.image(donor["image"]))):
                control = session.prepare_image(v, changed)
                row[op] = symbol_readout(control.clean_logits, session.label_ids(control, v["labels"]), v["correct_index"])
        preflight_rows.append(row)
    clusters = defaultdict(list)
    for row in preflight_rows:
        clusters[row["scene_id"]].append(row)
    accuracy_pre = sum(sum(r["clean"]["correct"] for r in rs) / len(rs) for rs in clusters.values()) / len(clusters)
    ready_clusters = [rs for rs in clusters.values() if all("shuffled" in r for r in rs)]
    gap = sum(sum(r["clean"]["correct"] - r["shuffled"]["correct"] for r in rs) / len(rs) for rs in ready_clusters) / len(ready_clusters) if ready_clusters else None
    behavior = {"accuracy": accuracy_pre, "image_gap": gap, "unit": "source_image_equal_weight",
                "missing_controls": preflight_missing, "rows": preflight_rows}
    with (out / "behavioral_preflight.json").open("x", encoding="utf-8") as stream:
        json.dump(behavior, stream, sort_keys=True, allow_nan=False)
    if (len(units) != p["target_units"] or preflight_missing or accuracy_pre < p["minimum_accuracy"]
            or gap is None or gap < p["minimum_image_gap"]):
        report = {"schema": "stretch.external-result.v1", "study": "external", "status": "insufficient",
                  "coverage": {"target_units": p["target_units"], "observed_units": len(units), "variants": len(variants)},
                  "scientific_verdict": "behaviorally_ineligible_no_lens_or_head_experiments", "behavior": behavior,
                  "protocol_sha256": canonical_hash(p), "identity": context["identity"],
                  "artifacts": {"behavior": str(out / "behavioral_preflight.json")}}
        from .reporting import write_study_artifacts
        report["artifacts"]["exports"] = write_study_artifacts(out, preflight_rows, report)
        with (out / "report.json").open("x", encoding="utf-8") as stream:
            json.dump(report, stream, sort_keys=True, allow_nan=False)
        return report
    rows, excluded = [], []
    with (out / "raw.jsonl").open("x", encoding="utf-8") as stream:
        def emit(v, operation, prepared, logits, **extra):
            score = symbol_readout(logits.cpu(), session.label_ids(prepared, v["labels"]), v["correct_index"])
            if not all(math.isfinite(x) for x in score["label_logits"]):
                raise ValueError("nonfinite external readout")
            row = {"variant_id": v["variant_id"], "pair_id": v["scene_id"], "scene_id": v["scene_id"],
                   "item_id": v["item_id"], "scheme": v["label_scheme"], "position": v["correct_index"],
                   "operation": operation, "split": config["split"], **score, **extra}
            rows.append(row)
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
            return score
        for v in variants:
            prepared = session.prepared(v["variant_id"])
            clean_logits, _ = session.adapter.decoder_forward(prepared)
            clean = emit(v, "clean", prepared, clean_logits[0])
            image = session.image(v["image"])
            alternatives = [w for w in variants if w["scene_id"] != v["scene_id"] and w["semantic_answer"] != v["semantic_answer"]]
            for op in ("blank", "shuffled"):
                if op == "shuffled" and not alternatives:
                    excluded.append({"variant_id": v["variant_id"], "reason": "no_shuffled_image"})
                    continue
                modified = Image.new("RGB", image.size, "white") if op == "blank" else session.image(sorted(alternatives, key=lambda w: canonical_hash([p["seed"], v["variant_id"], w["variant_id"]]))[0]["image"])
                control = session.prepare_image(v, modified)
                emit(v, op, control, control.clean_logits)
            for layer in p["lens_layers"]:
                site = Site("resid_post", layer)
                _, captures = session.adapter.decoder_forward(prepared, capture=[site], capture_positions=[prepared.final_position])
                lens = session.adapter.handle.readout(captures[site.key][0, 0]).float().cpu()
                emit(v, f"lens-L{layer}", prepared, lens, diagnostic="vocabulary_alignment_not_computation_time")
            if p["head_transfer"]:
                donors = [w for w in variants if w["item_id"] == v["item_id"] and w["label_scheme"] == v["label_scheme"]
                          and w["correct_symbol"] != v["correct_symbol"]]
                if not donors:
                    excluded.append({"variant_id": v["variant_id"], "reason": "no_position_donor"})
                    continue
                donor = sorted(donors, key=lambda w: canonical_hash([p["seed"], v["variant_id"], w["variant_id"]]))[0]
                donor_prepared = session.prepared(donor["variant_id"])
                donor_logits, _ = session.adapter.decoder_forward(donor_prepared)
                donor_scores = symbol_readout(donor_logits[0].cpu(), session.label_ids(donor_prepared, donor["labels"]), donor["correct_index"])
                interventions, self_patch = [], []
                for layer in sorted({l for l, _ in p["selected_heads"]}):
                    site = Site("head", layer)
                    _, captures = session.adapter.decoder_forward(donor_prepared, capture=[site], capture_positions=[donor_prepared.final_position])
                    _, own = session.adapter.decoder_forward(prepared, capture=[site], capture_positions=[prepared.final_position])
                    heads = [h for l, h in p["selected_heads"] if l == layer]
                    interventions.append(Intervention(site, [prepared.final_position], captures[site.key][0], heads=heads))
                    self_patch.append(Intervention(site, [prepared.final_position], own[site.key][0], heads=heads))
                donor_idx = v["labels"].index(donor["correct_symbol"])
                label_ids = session.label_ids(prepared, v["labels"])
                baseline_margin = float(clean_logits[0, label_ids[donor_idx]] - clean_logits[0, label_ids[v["correct_index"]]])
                condition_patches = [("position_frozen_heads", interventions), ("self_frozen_heads", self_patch)]
                reference_sets = {"all_selected_layers": [[l, h] for l in sorted({l for l, _ in p["selected_heads"]})
                                                         for h in range(session.adapter.handle.num_heads)]}
                for random_index in range(p["random_head_sets"]):
                    random_heads = []
                    for layer in sorted({l for l, _ in p["selected_heads"]}):
                        actual = {h for l, h in p["selected_heads"] if l == layer}
                        pool = sorted(set(range(session.adapter.handle.num_heads)) - actual,
                                      key=lambda h: canonical_hash([p["seed"], random_index, layer, h]))
                        if len(pool) < len(actual):
                            raise ValueError("insufficient layer-matched random heads")
                        random_heads.extend([[layer, h] for h in pool[:len(actual)]])
                    reference_sets[f"random_{random_index}"] = random_heads
                for set_name, head_set in reference_sets.items():
                    patches = []
                    for layer in sorted({l for l, _ in head_set}):
                        site = Site("head", layer)
                        _, donor_values = session.adapter.decoder_forward(donor_prepared, capture=[site], capture_positions=[donor_prepared.final_position])
                        patches.append(Intervention(site, [prepared.final_position], donor_values[site.key][0], heads=[h for l, h in head_set if l == layer]))
                    condition_patches.append(("position_" + set_name, patches))
                for operation, patches in condition_patches:
                    changed, _ = session.adapter.decoder_forward(prepared, interventions=patches)
                    margin = float(changed[0, label_ids[donor_idx]] - changed[0, label_ids[v["correct_index"]]])
                    emit(v, operation, prepared, changed[0], donor_variant_id=donor["variant_id"],
                         correct_correct=clean["correct"] and donor_scores["correct"],
                         margin_change=margin - baseline_margin, estimand="same_image_position_transfer")
    clean_rows = [r for r in rows if r["operation"] == "clean"]
    accuracy = sum(r["correct"] for r in clean_rows) / len(clean_rows)
    correct_transfer_units = {r["scene_id"] for r in rows if r["operation"] == "position_frozen_heads" and r.get("correct_correct") is True}
    complete = (len(units) == p["target_units"] and not excluded
                and (not p["head_transfer"] or len(correct_transfer_units) >= p["minimum_correct_transfer_units"]))
    report = {"schema": "stretch.external-result.v1", "study": "external", "status": "complete" if complete else "insufficient",
              "coverage": {"target_units": p["target_units"], "minimum_units": p["minimum_units"], "observed_units": len(units),
                           "variants": len(variants), "rows": len(rows), "exclusions": excluded,
                           "correct_transfer_units": len(correct_transfer_units)},
              "scientific_verdict": "behaviorally_ineligible" if accuracy < p["minimum_accuracy"] else "external_behavior_and_position_transfer_only",
              "accuracy": accuracy, "protocol_sha256": canonical_hash(p), "dataset_hash": session.manifest.get("dataset_hash"),
              "behavior": {k: v for k, v in behavior.items() if k != "rows"},
              "identity": context["identity"], "phase": config["phase"],
              "effects_correct_correct": grouped_effects([r for r in rows if r.get("correct_correct") is True], seed=p["seed"], bootstrap_samples=p["bootstrap_samples"]),
              "effects_all_cases_secondary": grouped_effects(rows, seed=p["seed"], bootstrap_samples=p["bootstrap_samples"]),
              "artifacts": {"raw": str(out / "raw.jsonl"), "report": str(out / "report.json")},
              "generalization_limit": "source_image_clusters; unknown shared templates may remain dependent; no semantic counterfactuals"}
    if complete and config["phase"] == "discovery":
        freeze = {"schema": "stretch.external-freeze.v1", "protocol_sha256": canonical_hash(p), "discovery_units": units}
        freeze["freeze_sha256"] = canonical_hash(freeze)
        with (out / "freeze.json").open("x", encoding="utf-8") as stream:
            json.dump(freeze, stream, indent=2, sort_keys=True)
        report["artifacts"]["freeze"] = str(out / "freeze.json")
    from .reporting import write_study_artifacts
    report["artifacts"]["exports"] = write_study_artifacts(out, rows, report)
    with (out / "report.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
    return report
