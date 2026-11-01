
import pandas as pd

EVIDENCE_FIELDS = {
    "name_available": "Hospital_Name",
    "district_available": "District",
    "pincode_available": "Pincode",
    "coordinates_available": "Location_Coordinates",
    "telephone_available": "Telephone",
    "mobile_available": "Mobile_Number",
    "email_available": "Hospital_Primary_Email_Id",
    "website_available": "Website",
    "specialties_available": "Specialties",
    "facilities_available": "Facilities",
    "care_type_available": "Hospital_Care_Type",
    "category_available": "Hospital_Category",
    "emergency_info_available": "Emergency_Services",
    "established_year_available": "Establised_Year",
    "empanelment_info_available": "Empanelment_or_Collaboration_with"
}

INVALID_VALUES = {"", "0", "0.0", "na", "n/a", "null", "none", "unknown", "nan", "<na>"}

def score_hospital_evidence(hospitals):
    """
    Calculate directory evidence completeness.

    Input: DataFrame containing original SRC001 directory columns.
    Output: DataFrame with evidence indicators, percentage and category.

    This measures information availability only. It is not a
    hospital quality, accreditation or trust assessment.
    """
    missing = [
        col for col in EVIDENCE_FIELDS.values()
        if col not in hospitals.columns
    ]
    if missing:
        raise ValueError(f"Missing source columns: {missing}")

    result = hospitals[
        ["Hospital_Name", "District"]
    ].copy()

    for feature, source_column in EVIDENCE_FIELDS.items():
        values = (
            hospitals[source_column]
            .astype(str)
            .str.strip()
            .str.casefold()
        )
        result[feature] = (~values.isin(INVALID_VALUES)).astype(int)

    feature_columns = list(EVIDENCE_FIELDS.keys())

    result["evidence_fields_present"] = result[
        feature_columns
    ].sum(axis=1)

    result["evidence_completeness_percentage"] = (
        result["evidence_fields_present"] /
        len(feature_columns) * 100
    ).round(2)

    def classify(score):
        if score >= 60:
            return "HIGH_COMPLETENESS"
        elif score >= 40:
            return "MODERATE_COMPLETENESS"
        elif score >= 20:
            return "LOW_COMPLETENESS"
        return "VERY_LOW_COMPLETENESS"

    result["completeness_category"] = (
        result["evidence_completeness_percentage"].apply(classify)
    )

    result["assessment_note"] = (
        "Directory information completeness only; "
        "not a hospital quality, accreditation or trust rating."
    )

    return result
