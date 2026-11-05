# Databricks notebook source
# MAGIC %md
# MAGIC # 08 — Avaliador Semântico de Equivalência de Respostas (Benchmark BEM)
# MAGIC
# MAGIC Avalia triplas `(question, reference_answer, candidate_answer)` em inglês comparando **Exact Match**, **Token-F1** e um **Adaptador Neural de Equivalência Semântica** (`databricks-qwen3-embedding-0-6b` + verificador estrutural de negação/números), calibrando o limiar no conjunto de calibração e medindo erros em paráfrases, alterações numéricas, negações e trocas de entidade no teste fechado.

# COMMAND ----------

import hashlib
import json
import math
import re
from datetime import datetime, timezone
import mlflow
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from databricks.sdk import WorkspaceClient
from pyspark.sql import SparkSession

spark = SparkSession.builder.getOrCreate()
w = WorkspaceClient()

TRIPLES = [
    # Calibration Split (15 triples: paraphrases, exact, numeric errors, negations, entity swaps)
    {"id": "T01", "split": "cal", "perturbation": "exact", "label": 1, "q": "What is the standard loan period at library LIB01?", "ref": "The standard loan period at LIB01 is 14 calendar days.", "cand": "The standard loan period at LIB01 is 14 calendar days."},
    {"id": "T02", "split": "cal", "perturbation": "paraphrase", "label": 1, "q": "What is the standard loan period at library LIB01?", "ref": "The standard loan period at LIB01 is 14 calendar days.", "cand": "Readers can borrow items from LIB01 for two weeks (14 days) by default."},
    {"id": "T03", "split": "cal", "perturbation": "paraphrase", "label": 1, "q": "How many consecutive renewals are allowed at LIB01?", "ref": "Up to 2 consecutive renewals are permitted if no reservation exists.", "cand": "Borrowers may renew an item up to 2 times in a row when nobody else has reserved it."},
    {"id": "T04", "split": "cal", "perturbation": "paraphrase", "label": 1, "q": "How long is a reserved book held for pickup at LIB02?", "ref": "Reserved books at LIB02 are held for 48 hours after notification.", "cand": "Once notified, you have 48 hours to pick up a reserved item at LIB02."},
    {"id": "T05", "split": "cal", "perturbation": "paraphrase", "label": 1, "q": "Does LIB01 provide wheelchair ramp access?", "ref": "Yes, LIB01 provides step-free ramp access and accessible materials.", "cand": "Affirmative, step-free entrance via ramp and accessible formats are available at LIB01."},
    {"id": "T06", "split": "cal", "perturbation": "paraphrase", "label": 1, "q": "What is the loan duration at branch LIB03?", "ref": "At branch LIB03, home loans last 21 calendar days.", "cand": "A standard borrowing period of 21 days applies to items checked out at LIB03."},
    {"id": "T07", "split": "cal", "perturbation": "paraphrase", "label": 1, "q": "Can a loan be renewed if another reader placed a hold?", "ref": "No, renewals are blocked whenever there is an active hold on the book.", "cand": "Items with an active reservation from another user cannot be renewed."},
    {"id": "T08", "split": "cal", "perturbation": "number_change", "label": 0, "q": "What is the standard loan period at library LIB01?", "ref": "The standard loan period at LIB01 is 14 calendar days.", "cand": "The standard loan period at LIB01 is 30 calendar days."},
    {"id": "T09", "split": "cal", "perturbation": "number_change", "label": 0, "q": "How many consecutive renewals are allowed at LIB01?", "ref": "Up to 2 consecutive renewals are permitted if no reservation exists.", "cand": "Up to 5 consecutive renewals are permitted if no reservation exists."},
    {"id": "T10", "split": "cal", "perturbation": "number_change", "label": 0, "q": "How long is a reserved book held for pickup at LIB02?", "ref": "Reserved books at LIB02 are held for 48 hours after notification.", "cand": "Reserved books at LIB02 are held for 24 hours after notification."},
    {"id": "T11", "split": "cal", "perturbation": "negation", "label": 0, "q": "Does LIB01 provide wheelchair ramp access?", "ref": "Yes, LIB01 provides step-free ramp access and accessible materials.", "cand": "No, LIB01 does not provide step-free ramp access."},
    {"id": "T12", "split": "cal", "perturbation": "negation", "label": 0, "q": "Can a loan be renewed when no hold is active?", "ref": "Yes, loans can be renewed online when there is no active hold.", "cand": "Loans cannot be renewed online even when there is no active hold."},
    {"id": "T13", "split": "cal", "perturbation": "entity_swap", "label": 0, "q": "What is the standard loan period at library LIB01?", "ref": "The standard loan period at LIB01 is 14 calendar days.", "cand": "The standard loan period at LIB03 is 14 calendar days."},
    {"id": "T14", "split": "cal", "perturbation": "entity_swap", "label": 0, "q": "How long is a reserved book held at LIB02?", "ref": "Reserved books at LIB02 are held for 48 hours.", "cand": "Reserved books at LIB01 are held for 48 hours."},
    {"id": "T15", "split": "cal", "perturbation": "unrelated", "label": 0, "q": "What is the loan duration at branch LIB03?", "ref": "At branch LIB03, home loans last 21 calendar days.", "cand": "The library catalog is indexed every night using Delta Lake tables."},

    # Closed Test Split (15 triples)
    {"id": "T16", "split": "test", "perturbation": "exact", "label": 1, "q": "How long is a reserved item kept at LIB01?", "ref": "Reserved items at LIB01 are kept for 24 hours after the alert.", "cand": "Reserved items at LIB01 are kept for 24 hours after the alert."},
    {"id": "T17", "split": "test", "perturbation": "paraphrase", "label": 1, "q": "How long is a reserved item kept at LIB01?", "ref": "Reserved items at LIB01 are kept for 24 hours after the alert.", "cand": "After receiving the alert, patrons at LIB01 have 24 hours to collect a reserved book."},
    {"id": "T18", "split": "test", "perturbation": "paraphrase", "label": 1, "q": "What is the initial borrowing window at LIB02?", "ref": "The initial borrowing window at LIB02 is 10 calendar days.", "cand": "Patrons at LIB02 may keep borrowed books for an initial period of 10 days."},
    {"id": "T19", "split": "test", "perturbation": "paraphrase", "label": 1, "q": "How many renewals are permitted at branch LIB02?", "ref": "Branch LIB02 allows 1 renewal per loan when no hold is pending.", "cand": "A single (1) renewal is allowed at LIB02 provided the item is not requested by another reader."},
    {"id": "T20", "split": "test", "perturbation": "paraphrase", "label": 1, "q": "What is the pickup deadline for reservations at LIB03?", "ref": "At LIB03, reservations must be picked up within 72 hours.", "cand": "Readers have a 72-hour window to claim reserved books at branch LIB03."},
    {"id": "T21", "split": "test", "perturbation": "paraphrase", "label": 1, "q": "Are tactile paving and priority seating available at LIB02?", "ref": "Yes, LIB02 features tactile paving and priority service desks.", "cand": "LIB02 is equipped with both tactile floor guidance and priority attendance."},
    {"id": "T22", "split": "test", "perturbation": "paraphrase", "label": 1, "q": "Are external email notifications sent automatically by the MVP bulletin job?", "ref": "No, the MVP bulletin pipeline writes verified reports to Delta without external sends (0 external emails).", "cand": "The MVP automation stores bulletins in Delta tables and performs 0 external dispatches."},
    {"id": "T23", "split": "test", "perturbation": "number_change", "label": 0, "q": "What is the initial borrowing window at LIB02?", "ref": "The initial borrowing window at LIB02 is 10 calendar days.", "cand": "The initial borrowing window at LIB02 is 21 calendar days."},
    {"id": "T24", "split": "test", "perturbation": "number_change", "label": 0, "q": "What is the pickup deadline for reservations at LIB03?", "ref": "At LIB03, reservations must be picked up within 72 hours.", "cand": "At LIB03, reservations must be picked up within 12 hours."},
    {"id": "T25", "split": "test", "perturbation": "number_change", "label": 0, "q": "How many renewals are permitted at branch LIB02?", "ref": "Branch LIB02 allows 1 renewal per loan when no hold is pending.", "cand": "Branch LIB02 allows 3 renewals per loan when no hold is pending."},
    {"id": "T26", "split": "test", "perturbation": "negation", "label": 0, "q": "Are tactile paving and priority seating available at LIB02?", "ref": "Yes, LIB02 features tactile paving and priority service desks.", "cand": "No, LIB02 does not have tactile paving or priority service desks."},
    {"id": "T27", "split": "test", "perturbation": "negation", "label": 0, "q": "Can reserved items be renewed by the current holder?", "ref": "No, items with an active reservation cannot be renewed.", "cand": "Yes, items with an active reservation can still be renewed online."},
    {"id": "T28", "split": "test", "perturbation": "entity_swap", "label": 0, "q": "What is the pickup deadline for reservations at LIB03?", "ref": "At LIB03, reservations must be picked up within 72 hours.", "cand": "At LIB01, reservations must be picked up within 72 hours."},
    {"id": "T29", "split": "test", "perturbation": "entity_swap", "label": 0, "q": "What is the initial borrowing window at LIB02?", "ref": "The initial borrowing window at LIB02 is 10 calendar days.", "cand": "The initial borrowing window at LIB03 is 10 calendar days."},
    {"id": "T30", "split": "test", "perturbation": "unrelated", "label": 0, "q": "How long is a reserved item kept at LIB01?", "ref": "Reserved items at LIB01 are kept for 24 hours after the alert.", "cand": "The hourly bike demand model uses a 2-hour causal availability lag."},
]

WORD_NUM_MAP = {"zero": "0", "one": "1", "single": "1", "two": "2", "three": "3", "four": "4", "five": "5"}

def normalize_tokens(text: str):
    cleaned = re.sub(r"[^a-z0-9\s]", " ", text.lower())
    return [WORD_NUM_MAP.get(tok, tok) for tok in cleaned.split() if tok]

def exact_match_score(ref: str, cand: str) -> float:
    return 1.0 if " ".join(normalize_tokens(ref)) == " ".join(normalize_tokens(cand)) else 0.0

def token_f1_score(ref: str, cand: str) -> float:
    r_toks = normalize_tokens(ref)
    c_toks = normalize_tokens(cand)
    if not r_toks or not c_toks:
        return 0.0
    common = set(r_toks) & set(c_toks)
    if not common:
        return 0.0
    prec = len(common) / len(set(c_toks))
    rec = len(common) / len(set(r_toks))
    return 2 * prec * rec / (prec + rec)

def extract_numbers(text: str):
    toks = normalize_tokens(text)
    return {t for t in toks if t.isdigit()}

def extract_entities(text: str):
    return set(re.findall(r"LIB\d{2}", text.upper()))

def has_negation(text: str) -> bool:
    neg_words = {"no", "not", "cannot", "never", "without", "blocked"}
    return bool(neg_words & set(normalize_tokens(text)))

# Obter embeddings densos via endpoint gerenciado databricks-qwen3-embedding-0-6b
EMBEDDING_ENDPOINT = "databricks-qwen3-embedding-0-6b"
ref_texts = [f"Question: {t['q']} Answer: {t['ref']}" for t in TRIPLES]
cand_texts = [f"Question: {t['q']} Answer: {t['cand']}" for t in TRIPLES]

resp_ref = w.api_client.do("POST", f"/serving-endpoints/{EMBEDDING_ENDPOINT}/invocations", body={"input": ref_texts})
resp_cand = w.api_client.do("POST", f"/serving-endpoints/{EMBEDDING_ENDPOINT}/invocations", body={"input": cand_texts})

ref_embs = [item["embedding"] for item in resp_ref["data"]]
cand_embs = [item["embedding"] for item in resp_cand["data"]]

def cosine_sim(v1, v2) -> float:
    a = np.array(v1, dtype=float)
    b = np.array(v2, dtype=float)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

for idx, t in enumerate(TRIPLES):
    t["em_score"] = exact_match_score(t["ref"], t["cand"])
    t["f1_score"] = round(token_f1_score(t["ref"], t["cand"]), 4)
    raw_cos = cosine_sim(ref_embs[idx], cand_embs[idx])

    # Guardrails estruturais do adaptador (números, entidades LIBxx e polaridade de negação)
    nums_ref = extract_numbers(t["ref"])
    nums_cand = extract_numbers(t["cand"])
    ents_ref = extract_entities(t["ref"])
    ents_cand = extract_entities(t["cand"])
    neg_ref = has_negation(t["ref"])
    neg_cand = has_negation(t["cand"])

    penalty = 0.0
    if nums_ref and nums_ref != nums_cand:
        penalty += 0.35
    if ents_ref and ents_ref != ents_cand:
        penalty += 0.35
    if neg_ref != neg_cand:
        penalty += 0.30

    t["raw_cosine"] = round(raw_cos, 4)
    t["bem_adapter_score"] = round(max(0.0, raw_cos - penalty), 4)

cal_triples = [t for t in TRIPLES if t["split"] == "cal"]
test_triples = [t for t in TRIPLES if t["split"] == "test"]

# Calibração do limiar no split cal: priorizar zero falsos equivalentes em number_change/negation e maximizar F1
best_thresh = 0.78
best_cal_f1 = -1.0
for th in [0.70, 0.75, 0.78, 0.80, 0.82, 0.85]:
    y_true_c = [t["label"] for t in cal_triples]
    y_pred_c = [1 if t["bem_adapter_score"] >= th else 0 for t in cal_triples]
    critical_fp = sum(
        1 for t, p in zip(cal_triples, y_pred_c)
        if t["perturbation"] in ("number_change", "negation", "entity_swap") and p == 1
    )
    f1_c = f1_score(y_true_c, y_pred_c)
    if critical_fp == 0 and f1_c > best_cal_f1:
        best_cal_f1 = f1_c
        best_thresh = th

def eval_method(triples_subset, score_key, threshold):
    y_true = [t["label"] for t in triples_subset]
    y_pred = [1 if t[score_key] >= threshold else 0 for t in triples_subset]
    critical_fp = sum(
        1 for t, p in zip(triples_subset, y_pred)
        if t["perturbation"] in ("number_change", "negation", "entity_swap") and p == 1
    )
    paraphrase_recall = sum(
        1 for t, p in zip(triples_subset, y_pred)
        if t["perturbation"] == "paraphrase" and p == 1
    ) / max(1, sum(1 for t in triples_subset if t["perturbation"] == "paraphrase"))
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "paraphrase_recall": round(float(paraphrase_recall), 4),
        "critical_false_equivalents": int(critical_fp),
    }

mlflow.set_experiment("/Shared/portfolio-estudos/08_modelo_kaggle_avaliador/experimentos_equivalencia")

with mlflow.start_run(run_name="K01_comparativo_em_f1_bem_adapter"):
    em_test = eval_method(test_triples, "em_score", 1.0)
    f1_test = eval_method(test_triples, "f1_score", 0.65)
    raw_emb_test = eval_method(test_triples, "raw_cosine", 0.85)
    bem_cal = eval_method(cal_triples, "bem_adapter_score", best_thresh)
    bem_test = eval_method(test_triples, "bem_adapter_score", best_thresh)

    mlflow.log_params({
        "embedding_endpoint": EMBEDDING_ENDPOINT,
        "calibrated_threshold": best_thresh,
        "cal_triples": len(cal_triples),
        "test_triples": len(test_triples),
    })
    mlflow.log_metrics({
        "test_em_f1": em_test["f1"],
        "test_token_f1": f1_test["f1"],
        "test_raw_cosine_critical_fp": raw_emb_test["critical_false_equivalents"],
        "test_bem_adapter_f1": bem_test["f1"],
        "test_bem_adapter_precision": bem_test["precision"],
        "test_bem_adapter_paraphrase_recall": bem_test["paraphrase_recall"],
        "test_bem_adapter_critical_fp": bem_test["critical_false_equivalents"],
    })

spark.sql("CREATE SCHEMA IF NOT EXISTS workspace.estudos_ai")
spark.createDataFrame(TRIPLES).write.mode("overwrite").saveAsTable("workspace.estudos_ai.aurora_avaliador_equivalencia")

summary_payload = {
    "measured_on": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
    "status": "SUCCESS",
    "embedding_endpoint": EMBEDDING_ENDPOINT,
    "total_triples": len(TRIPLES),
    "calibration_triples": len(cal_triples),
    "test_triples": len(test_triples),
    "calibrated_threshold": best_thresh,
    "delta_table": "workspace.estudos_ai.aurora_avaliador_equivalencia",
    "methods_test": {
        "exact_match": em_test,
        "token_f1": f1_test,
        "uncalibrated_embedding_cosine": raw_emb_test,
        "calibrated_bem_adapter": bem_test,
    },
    "calibration_metrics": bem_cal,
}

print("JSON_SUMMARY_START")
print(json.dumps(summary_payload, ensure_ascii=False))
print("JSON_SUMMARY_END")
