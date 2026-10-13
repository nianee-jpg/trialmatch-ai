"""Controlled vocabulary of patient data fields, and the rules that map
protocol wording onto them.

Why a fixed vocabulary? The Patient Agent (Milestone 3) will expose exactly
these fields as tools, so every "required data" item on a criterion must be
something that can actually be looked up. If no rule matches a criterion, it is
flagged REQUIRED_DATA_UNMAPPED instead of being guessed.

The keyword rules below were written for NCT02260934 (CALIBRATE). They are
plain regular expressions, so they are easy to read, test and extend.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Pattern, Tuple

# field name -> short description (shown in the UI)
DATA_FIELDS: Dict[str, str] = {
    "demographics.age": "Age in years at screening",
    "demographics.sex": "Sex",
    "diagnosis.sle_acr_criteria": "SLE diagnosis and whether ACR classification criteria are met",
    "labs.ana": "Antinuclear antibody result, with date",
    "labs.anti_dsdna": "Anti-dsDNA antibody result, with date",
    "visits.visit_minus_1_date": "Date of study visit -1 (reference date for serology window)",
    "visits.screening_date": "Date of screening",
    "visits.enrollment_date": "Planned date of study enrollment",
    "renal.biopsy_class": "Kidney biopsy ISN/RPS class(es)",
    "renal.biopsy_date": "Kidney biopsy date",
    "urinalysis.rbc_per_hpf": "Urine red blood cells per high-power field, with date",
    "urinalysis.wbc_per_hpf": "Urine white blood cells per high-power field, with date",
    "urinalysis.cellular_casts": "Urine cellular casts (type), with date",
    "reproductive.menses_at_collection": "Whether menstruating at time of urine collection",
    "labs.upcr": "Urine protein-to-creatinine ratio, with date",
    "labs.upcr_collection_method": "UPCR specimen type (24-hour collection vs spot urine)",
    "consent.able_to_consent": "Ability to provide informed consent",
    "medications.history": "Medication history with drug names, routes and dates",
    "labs.anc": "Absolute neutrophil count, with units and date",
    "labs.platelets": "Platelet count, with units and date",
    "labs.hemoglobin": "Hemoglobin, with units and date",
    "labs.igg": "Serum IgG, with units and date",
    "labs.iga": "Serum IgA, with units and date",
    "labs.ast": "AST with the lab's upper limit of normal",
    "labs.alt": "ALT with the lab's upper limit of normal",
    "labs.alkaline_phosphatase": "Alkaline phosphatase with the lab's upper limit of normal",
    "labs.egfr": "Estimated GFR (mL/min/1.73m^2), with date",
    "infections.quantiferon_tb": "QuantiFERON-TB Gold result, with date",
    "imaging.chest_radiograph": "Chest radiograph findings, with date",
    "infections.current": "Current active infections",
    "infections.hiv": "HIV test result",
    "infections.hepatitis_b": "Hepatitis B serology (HBsAg and anti-HBc)",
    "infections.hepatitis_c": "Hepatitis C serology",
    "infections.hospitalizations": "Hospitalizations for infection, with dates",
    "infections.parenteral_antimicrobials": "IV/IM antimicrobial treatment, with dates",
    "infections.chronic_suppressive_therapy": "Chronic infections on suppressive therapy",
    "infections.history": "History of significant or recurrent infections",
    "vaccinations.live_attenuated": "Live-attenuated vaccines, with dates",
    "history.malignancy": "Malignancy history, with type",
    "history.transplantation": "Transplant history",
    "history.primary_immunodeficiency": "Primary immunodeficiency history",
    "history.comorbidities": "Other significant medical conditions",
    "reproductive.pregnancy_status": "Pregnancy status / test result",
    "reproductive.breastfeeding": "Whether currently breastfeeding",
    "reproductive.contraception_willingness": "Willingness to use FDA-approved contraception",
    "medications.corticosteroid_courses": "Courses of systemic corticosteroids, with dates",
    "social.substance_abuse": "Current or past-year substance abuse",
    "allergies.monoclonal_antibodies": "Severe allergic/anaphylactic reactions to monoclonal antibodies",
    "allergies.contrast_agents": "Anaphylaxis to parenteral contrast agents",
    "procedures.ivig_plasmapheresis_leukopheresis": "IVIG, plasmapheresis or leukopheresis, with dates",
    "access.peripheral_venous": "Peripheral venous access",
    "psychiatric.history": "Severe depression or psychiatric conditions",
    "psychiatric.suicidality": "Suicidal thoughts/behaviour, with dates",
    "compliance.able_to_comply": "Ability to comply with study procedures",
    "assessment.investigator_judgement": "Investigator's documented clinical judgement",
}

_I = re.IGNORECASE

# (pattern, field, detail). Evaluated against the unescaped criterion text.
REQUIRED_DATA_RULES: List[Tuple[Pattern[str], str, Optional[str]]] = [
    (re.compile(r"diagnosis of systemic lupus erythematosus", _I), "diagnosis.sle_acr_criteria", None),
    (re.compile(r"antinuclear antibody|\bANA\b", _I), "labs.ana", None),
    (re.compile(r"anti-ds ?DNA", _I), "labs.anti_dsdna", None),
    (re.compile(r"visit -1", _I), "visits.visit_minus_1_date", None),
    (re.compile(r"kidney biopsy", _I), "renal.biopsy_class", None),
    (re.compile(r"kidney biopsy", _I), "renal.biopsy_date", None),
    (re.compile(r"\bRBC/hpf\b", _I), "urinalysis.rbc_per_hpf", None),
    (re.compile(r"WBC/hpf", _I), "urinalysis.wbc_per_hpf", None),
    (re.compile(r"cellular casts", _I), "urinalysis.cellular_casts", None),
    (re.compile(r"absence of menses", _I), "reproductive.menses_at_collection", None),
    (re.compile(r"absence of (?:menses and )?infection", _I), "infections.current", "urinary/other infection at collection"),
    (re.compile(r"protein-to-creatinine|\bUPCR\b", _I), "labs.upcr", None),
    (re.compile(r"24-hour collection", _I), "labs.upcr_collection_method", None),
    (re.compile(r"informed consent", _I), "consent.able_to_consent", None),
    (re.compile(r"mycophenolate", _I), "medications.history", "mycophenolate mofetil"),
    (re.compile(r"cyclophosphamide", _I), "medications.history", "cyclophosphamide"),
    (re.compile(r"absolute neutrophil count", _I), "labs.anc", None),
    (re.compile(r"platelets", _I), "labs.platelets", None),
    (re.compile(r"\bHgb\b|hemoglobin", _I), "labs.hemoglobin", None),
    (re.compile(r"\bIgG\b", _I), "labs.igg", None),
    (re.compile(r"\bIgA\b", _I), "labs.iga", None),
    (re.compile(r"QuantiFERON", _I), "infections.quantiferon_tb", None),
    (re.compile(r"chest radiograph", _I), "imaging.chest_radiograph", None),
    (re.compile(r"active bacterial, viral, fungal", _I), "infections.current", None),
    (re.compile(r"\bHIV\b", _I), "infections.hiv", None),
    (re.compile(r"hepatitis B", _I), "infections.hepatitis_b", None),
    (re.compile(r"hepatitis C", _I), "infections.hepatitis_c", None),
    (re.compile(r"hospitalization for treatment of infections", _I), "infections.hospitalizations", None),
    (re.compile(r"parenteral \(IV or IM\)", _I), "infections.parenteral_antimicrobials", None),
    (re.compile(r"suppressive antibiotic therapy", _I), "infections.chronic_suppressive_therapy", None),
    (re.compile(r"significant infection or recurrent infection", _I), "infections.history", None),
    (re.compile(r"live-attenuated vaccine", _I), "vaccinations.live_attenuated", None),
    (re.compile(r"study enrollment", _I), "visits.enrollment_date", None),
    (re.compile(r"\beGFR\b", _I), "labs.egfr", None),
    (re.compile(r"malignan", _I), "history.malignancy", None),
    (re.compile(r"transplantation", _I), "history.transplantation", None),
    (re.compile(r"primary immunodeficiency", _I), "history.primary_immunodeficiency", None),
    (re.compile(r"^pregnancy\.?$", _I), "reproductive.pregnancy_status", None),
    (re.compile(r"^breastfeeding\.?$", _I), "reproductive.breastfeeding", None),
    (re.compile(r"birth control", _I), "reproductive.contraception_willingness", None),
    (re.compile(r"anti-Tumor Necrosis Factor|\bTNF\b", _I), "medications.history", "anti-TNF agents"),
    (re.compile(r"other biologic medications", _I), "medications.history", "other biologic medications"),
    (re.compile(r"experimental non- ?biologic", _I), "medications.history", "experimental non-biologic agents"),
    (re.compile(r"prior to screening", _I), "visits.screening_date", None),
    (re.compile(r"Intravenous immunoglobulin|IVIG|plasmapheresis|leukopheresis", _I),
     "procedures.ivig_plasmapheresis_leukopheresis", None),
    (re.compile(r"investigational biologic agent", _I), "medications.history", "investigational biologic agents"),
    (re.compile(r"rituximab", _I), "medications.history", "rituximab"),
    (re.compile(r"belimumab", _I), "medications.history", "belimumab"),
    (re.compile(r"atacicept", _I), "medications.history", "atacicept"),
    (re.compile(r"biologic B cell therapy", _I), "medications.history", "other biologic B-cell therapy"),
    (re.compile(r"aspartate aminotransferase|\bAST\b"), "labs.ast", "value and upper limit of normal"),
    (re.compile(r"alanine aminotransferase|\bALT\b"), "labs.alt", "value and upper limit of normal"),
    (re.compile(r"alkaline phosphatase", _I), "labs.alkaline_phosphatase", "value and upper limit of normal"),
    (re.compile(r"renal, hepatic, hematological", _I), "history.comorbidities", None),
    (re.compile(r"corticosteroid", _I), "medications.corticosteroid_courses", None),
    (re.compile(r"comorbidities", _I), "history.comorbidities", None),
    (re.compile(r"substance abuse", _I), "social.substance_abuse", None),
    (re.compile(r"monoclonal antibodies", _I), "allergies.monoclonal_antibodies", None),
    (re.compile(r"contrast agents", _I), "allergies.contrast_agents", None),
    (re.compile(r"peripheral venous access", _I), "access.peripheral_venous", None),
    (re.compile(r"depression|psychiatric", _I), "psychiatric.history", None),
    (re.compile(r"suicid", _I), "psychiatric.suicidality", None),
    (re.compile(r"comply with study", _I), "compliance.able_to_comply", None),
    (re.compile(r"investigator's opinion", _I), "assessment.investigator_judgement", None),
]

# Keywords that identify WHICH measurement a numeric threshold applies to.
THRESHOLD_PARAMETER_RULES: List[Tuple[Pattern[str], str]] = [
    (re.compile(r"absolute neutrophil count", _I), "labs.anc"),
    (re.compile(r"platelets", _I), "labs.platelets"),
    (re.compile(r"\bHgb\b|hemoglobin", _I), "labs.hemoglobin"),
    (re.compile(r"\bIgG\b"), "labs.igg"),
    (re.compile(r"\bIgA\b"), "labs.iga"),
    (re.compile(r"\bUPCR\b|protein-to-creatinine", _I), "labs.upcr"),
    (re.compile(r"\beGFR\b"), "labs.egfr"),
    (re.compile(r"\bRBC/hpf\b"), "urinalysis.rbc_per_hpf"),
    (re.compile(r"White blood cell per high powered field|WBC/hpf", _I), "urinalysis.wbc_per_hpf"),
    (re.compile(r"systemic corticosteroids", _I), "medications.corticosteroid_courses"),
]

# Fields measured against the lab's upper limit of normal ("2 times the ULN").
ULN_PARAMETER_RULES: List[Tuple[Pattern[str], str]] = [
    (re.compile(r"aspartate aminotransferase|\bAST\b"), "labs.ast"),
    (re.compile(r"alanine aminotransferase|\bALT\b"), "labs.alt"),
    (re.compile(r"alkaline phosphatase", _I), "labs.alkaline_phosphatase"),
]
