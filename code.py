import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import random
import os

def clean_text(x):
    if not isinstance(x, str):
        x = str(x)
    return x.strip().replace("\n"," ").replace("\r"," ")

def sample_values(series, k=30):
    vals = [str(v) for v in series.dropna().astype(str).values]
    if not vals:
        return []
    if len(vals) > k:
        random.seed(42)
        vals = random.sample(vals, k)
    return vals

def make_embed_text(header, values, max_chars=500):
    t = clean_text(header)
    if values:
        joined = " | ".join([clean_text(v) for v in values])
        t = f"{t} [SEP] {joined[:max_chars]}"
    return t

PREMIUM_SCHEMA = {
    "policy_source": ["Policy Source","Source","UW System Name"],
    "reporting_period": ["Reporting period","Reporting Period"],
    "master_policy_reference": ["Master Policy Reference","MASTER Policy Reference"],
    "master_policy_inception": ["MASTER Policy Inception","Master Policy Inception"],
    "master_policy_expiration": ["MASTER Policy Expiry","Master Policy Expiry","MASTER Policy Expiration"],
    "policy_number": ["Policy Number","Policy No","Número de póliza","N° DE POL","Local Policy Number"],
    "assured_legal_name": ["Assured Legal Name","Assured Name","Insured Name","Local Insured Name"],
    "issuing_branch": ["Issuing Branch","Producing Branch/entity","Producing Branch","Branch"],
    "line_of_business": ["Line of Business","LOB","Policy LOB","General Casualty","Casualty","Property"],
    "sub_class": ["Sub Class","Sub Class (Primary Casualty)","Subclass","SubClass"],
    "type_of_inward_risk_policy": ["Type of Inward risk Policy","Type of Inward risk","Type of inward risk policy","Facultative"],
    "insured_country": ["Country","Risk Location Country","Insured Location","Risk Country"],
    "coverage_of_risk": ["Coverage of Risk","Coverage","Cobertura"],
    "cover_inception_date": ["Effective Date","Cover Inception Date","Inception Date","Start Date","VIG DESDE"],
    "cover_expiration_date": ["Expiry Date","Cover Expiration Date","Expiration Date","End Date","VIG HASTA"],
    "policy_limits": ["Policy Limits","Sum Insured","Limit"],
    "original_currency": ["Original Currency","Claim Original Currency Code","Currency (Original)","Moneda Original"],
    "gross_premium_original": ["Gross Premium (Original Currency)","Gross Premium Original","Gross Premium (Original)"],
    "local_retention": ["Local Retention","Retention Local","Retención Local"],
    "gross_premium_after_localised_retention": ["Gross Premium after Localised Retention","Gross Premium after Localized Retention"],
    "ceded_share": ["Ceded Share %","Ceded Share","% Cesión","% CEDIDA","Ceded %"],
    "ceded_premium_original": ["Ceded Premium (Original)","Ceded Premium Original"],
    "ceding_commission_percentage": ["Ceding Commission %","Ceding commission %","Comisión Cesión %"],
    "ceding_commission": ["Ceding Commission","Ceding commission","Comisión Cesión"],
    "brokerage_percentage": ["Brokerage %","Brokerage Percentage","Comisión Broker %","Brokerage Commission %"],
    "brokerage": ["Brokerage","Brokerage Commission","Comisión de Corretaje","Broker Commission"],
    "tax_percentage": ["Tax %","Tax Percentage"],
    "premium_tax": ["Premium Tax","IPT","Impuesto sobre Primas","Montant IPT"],
    "other_deductions_percentage": ["Other Deductions %","Deducciones %","Otras Deducciones %"],
    "other_deductions": ["Other Deductions","Deducciones","Otras Deducciones"],
    "settlement_currency": ["Settlement Currency"],
    "roe": ["ROE","Rate of Exchange","Exchange Rate"],
    "net_premium": ["Net Premium","Prima Neta","P NETA","Premium Net"]
}

class HeaderMapper:
    def __init__(self, model_name, canonical_map):
        self.model = SentenceTransformer(model_name)
        self.alias_texts = []
        self.alias_keys = []
        for canon, aliases in canonical_map.items():
            fake_alias = canon.replace("_"," ")
            for a in [fake_alias] + aliases:
                t = clean_text(a)
                if t:
                    self.alias_texts.append(t)
                    self.alias_keys.append(canon)
        self.alias_emb = self.model.encode(self.alias_texts, convert_to_numpy=True, normalize_embeddings=True)

    def map_headers(self, df, use_values=True, threshold=0.75, topk=3):
        out = {}
        for h in df.columns:
            try:
                vals = sample_values(df[h], 30) if use_values else []
                txt = make_embed_text(str(h), vals)
                h_emb = self.model.encode([txt], convert_to_numpy=True, normalize_embeddings=True)
                sims = cosine_similarity(h_emb, self.alias_emb)[0]
                idx = np.argsort(-sims)[:topk]
                cands = [(self.alias_keys[i], float(sims[i])) for i in idx]
                best_canon, best_score = cands[0]
                chosen = best_canon if best_score >= threshold else "UNKNOWN"
                out[str(h)] = (chosen, best_score, cands)
            except Exception:
                out[str(h)] = ("ERROR", 0.0, [("ERR", 0.0)])
        return out

def load_gold(path_csv):
    df = pd.read_csv(path_csv)
    return {clean_text(r.orig_header): clean_text(r.canonical_truth) for _, r in df.iterrows()}

def score_predictions(pred_map, gold_map):
    total = len(gold_map)
    correct = 0; unknown = 0
    for k, truth in gold_map.items():
        pred = pred_map.get(k, ("MISSING",0.0,[]))[0]
        if pred == truth:
            correct += 1
        elif pred == "UNKNOWN":
            unknown += 1
    wrong = total - correct - unknown
    acc = round(correct/total, 4) if total else 0.0
    return dict(total=total, correct=correct, unknown=unknown, wrong=wrong, accuracy=acc)

# --- Main mapping and accuracy logic ---
model_name = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
data_path = "bdx_premium_sample.xlsx"
gold_csv = "gold_mapping.csv"

# Load data
sheets = pd.read_excel(data_path, sheet_name=None)
mapper = HeaderMapper(model_name, PREMIUM_SCHEMA)

# Map columns for first sheet
sheet_name = list(sheets.keys())[0]
df = sheets[sheet_name]
pred_map = mapper.map_headers(df)

# Print mapping results
for h, (c, sc, cand) in pred_map.items():
    print(f"{h:40s} -> {c:35s} (score={sc:.3f})   top={cand}")

# Accuracy calculation
if os.path.exists(gold_csv):
    gold_map = load_gold(gold_csv)
    metrics = score_predictions(pred_map, gold_map)
    print("\n=== Accuracy Summary ===")
    for k,v in metrics.items():
        print(f"{k:10s}: {v}")
    print(f"\nMeaning: {metrics['accuracy']*100:.1f}% of tested columns ({metrics['correct']}/{metrics['total']}) mapped correctly.")
else:
    print("\n(no gold CSV provided — skipped accuracy)")
