import joblib
import numpy as np
import pandas as pd

# Lazy cache so importing this file on Streamlit startup takes 0.01 seconds!
ML_CACHE = {
    "reg_model": None,
    "clf_model": None,
    "scaler": None,
    "feature_names": None,
    "shap_explainer": None
}


def get_ml_models():
    if ML_CACHE["reg_model"] is None:
        import shap
        ML_CACHE["reg_model"] = joblib.load("best_regression_model.pkl")
        ML_CACHE["clf_model"] = joblib.load("best_classification_model.pkl")
        ML_CACHE["scaler"] = joblib.load("scaler.pkl")
        ML_CACHE["feature_names"] = joblib.load("feature_names.pkl")
        ML_CACHE["shap_explainer"] = shap.TreeExplainer(ML_CACHE["reg_model"])
    return (
        ML_CACHE["reg_model"],
        ML_CACHE["clf_model"],
        ML_CACHE["scaler"],
        ML_CACHE["feature_names"],
        ML_CACHE["shap_explainer"]
    )


def describe_shap_driver(feat, val, all_feats):
    if feat == "CE":
        if val < 0.45:
            return f"low Carbon Equivalent (CE={val:.3f}) limiting hardenability and promoting slack-quenched ferrite"
        return f"high Carbon Equivalent (CE={val:.3f}) increasing martensite brittleness, quench distortion, or retained austenite sensitivity"
    if feat == "TT":
        if val >= 500:
            return f"high tempering temperature (TT={val:.0f}°C) causing carbide coarsening (spheroidization) and matrix softening"
        return f"low tempering temperature (TT={val:.0f}°C) leaving untempered martensite notch sensitivity"
    if feat == "THQCr":
        if val < 15:
            return f"slow quench cooling rate (THQCr={val:.0f}°C/s) risking non-martensitic bainite/ferrite transformation"
        return f"quench cooling rate (THQCr={val:.0f}°C/s) thermal stress interaction with alloy hardenability"
    if feat == "Carburized":
        if val == 0:
            return "absence of surface carburizing (Carburized=0), lacking compressive residual surface stress"
        return "carburized case-core interface stress gradient requiring optimized diffusion and tempering"
    if feat in ["CT", "Ct", "Dt"]:
        if all_feats["Carburized"] == 0:
            return f"inactive carburizing parameter ({feat}={val}) on uncarburized specimen"
        if feat == "Dt" and val == 0:
            return "zero diffusion time (Dt=0 min) during carburizing, causing steep case carbon gradient"
        return f"carburizing thermal schedule ({feat}={val}) affecting effective case depth"
    if feat == "NT":
        return f"normalizing temperature (NT={val:.0f}°C) influencing prior-austenite grain size and microstructural banding"
    if feat == "THT":
        return f"through-hardening austenitizing temperature (THT={val:.0f}°C) affecting carbide dissolution and grain growth"
    if feat == "QmT":
        return f"quench medium temperature (QmT={val:.0f}°C) affecting martensite transformation cooling severity"
    if feat == "C":
        return f"carbon content (%C={val:.2f}) governing martensite hardness vs. fracture toughness balance"
    if feat == "Cr":
        if val < 0.50:
            return f"low chromium content (%Cr={val:.2f}) limiting alloy hardenability and secondary hardening resistance during tempering"
        return f"chromium content (%Cr={val:.2f}) affecting carbide formation and hardenability"
    if feat == "Mo" or feat == "Mo_added":
        if all_feats["Mo"] < 0.10:
            return f"low molybdenum content (%Mo={all_feats['Mo']:.2f}) reducing temper embrittlement resistance"
        return f"molybdenum content (%Mo={all_feats['Mo']:.2f}) interacting with carbide precipitation"
    if feat == "Ni":
        return f"nickel content (%Ni={val:.2f}) affecting matrix toughness"
    if feat == "RedRatio":
        return f"forging reduction ratio (RedRatio={val:.0f}) governing inclusion elongation and porosity consolidation"
    if feat in ["dA", "dB", "dC"]:
        return f"non-metallic inclusion rating ({feat}={val:.3f}) acting as cyclic stress raisers per ASTM E45"
    return f"process/alloy parameter ({feat}={val})"


def check_metallurgical_warnings(f):
    warnings = []
    if f["dA"] >= 0.06 or f["dB"] >= 0.05 or f["dC"] >= 0.05:
        warnings.append(
            f"High non-metallic inclusion content (dA={f['dA']}, dB={f['dB']}, dC={f['dC']}) promoting inclusion-initiated fatigue cracking (ASTM E45)"
        )
    if f["TT"] >= 550 and f["Cr"] < 0.50:
        warnings.append(
            f"High-temperature tempering (TT={f['TT']}°C) with low Chromium (%Cr={f['Cr']}%) causing cementite coarsening and matrix softening"
        )
    elif f["TT"] >= 620:
        warnings.append(
            f"Severe over-tempering temperature (TT={f['TT']}°C) causing dislocation recovery and carbide coarsening"
        )
    if f["THQCr"] < 15 and f["Carburized"] == 0:
        warnings.append(
            f"Slow quench cooling rate (THQCr={f['THQCr']}°C/s < 15°C/s) risking slack-quenched non-martensitic products"
        )
    if f["Carburized"] == 1 and f["Dt"] == 0:
        warnings.append(
            "Carburizing performed with zero diffusion period (Dt=0 min), risking excessive surface carbide network"
        )
    if f["RedRatio"] < 450:
        warnings.append(
            f"Low forging reduction ratio (RedRatio={f['RedRatio']}) risking unclosed microporosity"
        )
    return warnings


# Deterministic Python-verified fixes so the LLM never recommends parameters the steel already has!
def build_verified_action_plan(f):
    actions = []
    if f["Cr"] < 0.50 and f["Mo"] < 0.15:
        actions.append(
            f"Alloy Hardenability Upgrade: Increase Chromium (currently %Cr={f['Cr']}%) to 0.90–1.20% and Molybdenum (currently %Mo={f['Mo']}%) to 0.15–0.30% (per ASTM A255) to prevent slack-quenching and tempering softening."
        )
    elif f["Cr"] < 0.50 and f["Mo"] >= 0.15:
        actions.append(
            f"Chromium Adjustment (Molybdenum Already Sufficient at %Mo={f['Mo']}%): Increase Chromium from %Cr={f['Cr']}% to 0.80–1.10% Cr to balance carbide stability and hardenability per ASTM A255; no additional Mo is needed."
        )
    elif f["Cr"] >= 0.50 and f["Mo"] < 0.10 and f["TT"] >= 500:
        actions.append(
            f"Molybdenum Addition for Temper Embrittlement Resistance: Add 0.15–0.25% Mo (currently %Mo={f['Mo']}%) to suppress grain-boundary phosphorus segregation during high-temperature tempering."
        )

    if f["TT"] >= 550:
        actions.append(
            f"Tempering Schedule Optimization: Lower the tempering temperature from TT={f['TT']}°C to the 400–500°C range (or 160–200°C if surface hardened) to prevent cementite spheroidization and matrix softening (verify hardness via ASTM E384)."
        )

    if f["THQCr"] < 15 and f["Carburized"] == 0:
        actions.append(
            f"Quench Severity Improvement: Increase quench cooling rate from THQCr={f['THQCr']}°C/s to at least 16–25°C/s via agitated oil/polymer quenching to bypass pearlite/bainite noses."
        )

    if f["Carburized"] == 0:
        actions.append(
            "Surface Case Hardening: Apply gas or vacuum carburizing (CT=900–930°C, followed by diffusion Dt >= 30 min and 180°C tempering) or induction hardening to introduce compressive residual surface stress (verify via ASTM E1077)."
        )
    elif f["Carburized"] == 1 and f["Dt"] == 0:
        actions.append(
            f"Carburizing Diffusion Stage: Add an active diffusion period (Dt = 30–60 min at {f['CT']}°C instead of Dt=0 min) to eliminate brittle grain-boundary carbide networks."
        )

    if f["dA"] >= 0.05 or f["dB"] >= 0.04 or f["dC"] >= 0.04:
        actions.append(
            f"Secondary Steelmaking Cleanliness (ASTM E45): Apply vacuum degassing, calcium inclusion shape control, or Electro-Slag Remelting (ESR) to reduce current inclusion levels (dA={f['dA']}, dB={f['dB']}, dC={f['dC']}) below dB <= 0.005."
        )

    if f["RedRatio"] < 500:
        actions.append(
            f"Hot-Working Consolidation: Increase forging reduction ratio from RedRatio={f['RedRatio']} to >= 600 to close internal casting microporosity."
        )

    if len(actions) == 0:
        actions.append(
            "Specimen parameters are already well-optimized; maintain strict non-destructive magnetic particle inspection (ASTM E1444) and surface finish control."
        )
    return actions


def predict_and_create_rag_query(raw_inputs):
    reg_model, clf_model, scaler, feature_names, shap_explainer = get_ml_models()
    f = dict(raw_inputs)

    f["CE"] = round(f["C"] + (f["Mn"] / 6.0) +
                    ((f["Cr"] + f["Mo"]) / 5.0) + ((f["Ni"] + f["Cu"]) / 15.0), 4)
    f["Carburized"] = 1 if (f["CT"] >= 750 and f["Ct"] > 0) else 0
    f["Mo_added"] = 1 if f["Mo"] >= 0.05 else 0

    df_24 = pd.DataFrame(
        [[f[col] for col in feature_names]], columns=feature_names)
    scaled_24 = scaler.transform(df_24)

    raw_pred = float(reg_model.predict(scaled_24)[0])
    pred_mpa = float(np.expm1(raw_pred)) if raw_pred < 20.0 else raw_pred

    pred_class_idx = int(clf_model.predict(scaled_24)[0])
    grade_map = {0: "Low (<450 MPa)",
                 1: "Medium (450-700 MPa)", 2: "High (>700 MPa)"}
    grade_label = grade_map.get(pred_class_idx, "Medium (450-700 MPa)")

    # Fast TreeSHAP path approximation runs in ~0.05s on CPU instead of 7s!
    shap_vals = shap_explainer.shap_values(
        scaled_24, check_additivity=False, approximate=True)[0]

    negative_drivers = []
    for idx, feat_name in enumerate(feature_names):
        s_val = float(shap_vals[idx])
        if s_val < 0:
            if f["Carburized"] == 0 and feat_name in ["CT", "Ct", "Dt"]:
                continue
            mpa_penalty = round(
                s_val * pred_mpa, 1) if raw_pred < 20.0 else round(s_val, 1)
            negative_drivers.append({
                "feature": feat_name,
                "value": round(float(f[feat_name]), 4),
                "shap_score": mpa_penalty,
                "meaning": describe_shap_driver(feat_name, f[feat_name], f)
            })

    negative_drivers.sort(key=lambda x: x["shap_score"])
    top_4_drivers = negative_drivers[:4]

    rule_warnings = check_metallurgical_warnings(f)
    verified_actions = build_verified_action_plan(f)

    # Compact, high-signal query for fast CPU Cross-Encoder re-ranking
    short_driver_names = ", ".join(
        [f"{d['feature']}={d['value']}" for d in top_4_drivers])
    rag_query = (
        f"Steel rotating bending fatigue failure analysis ({pred_mpa:.0f} MPa, {grade_label}): "
        f"tempering softening, slack quenching, non-metallic inclusions ASTM E45, carburizing case depth ({short_driver_names})"
    )

    specimen_summary = (
        f"Composition: C={f['C']}%, Si={f['Si']}%, Mn={f['Mn']}%, Ni={f['Ni']}%, Cr={f['Cr']}%, Mo={f['Mo']}%, Cu={f['Cu']}%, CE={f['CE']:.3f} | "
        f"Heat Treatment: NT={f['NT']}°C, THT={f['THT']}°C ({f['THt']} min), THQCr={f['THQCr']}°C/s, QmT={f['QmT']}°C, TT={f['TT']}°C ({f['Tt']} min) | "
        f"Carburized={f['Carburized']} (CT={f['CT']}°C, Ct={f['Ct']}m, Dt={f['Dt']}m) | "
        f"Cleanliness & Forging: RedRatio={f['RedRatio']}, dA={f['dA']}, dB={f['dB']}, dC={f['dC']}"
    )

    return {
        "predicted_mpa": round(pred_mpa, 1),
        "grade": grade_label,
        "engineered_features": {
            "CE": f["CE"],
            "Carburized": f["Carburized"],
            "Mo_added": f["Mo_added"]
        },
        "specimen_summary": specimen_summary,
        "top_4_shap_drivers": top_4_drivers,
        "metallurgical_warnings": rule_warnings,
        "verified_actions": verified_actions,
        "rag_query": rag_query
    }
