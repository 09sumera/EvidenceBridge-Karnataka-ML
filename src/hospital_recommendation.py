
import pandas as pd

def recommend_hospitals(
    hospitals,
    district=None,
    specialty=None,
    limit=10,
    require_coordinates=False
):
    """
    Return hospital candidates using available directory information.

    This is candidate selection, not a clinically validated ranking.
    No distance, price, availability or quality is inferred.
    """

    data = hospitals.copy()

    if district:
        data = data[
            data["District"].astype(str).str.strip().str.casefold()
            == district.strip().casefold()
        ]

    if specialty:
        specialty_values = data["Specialties"].astype(str).str.casefold()
        data = data[
            specialty_values.str.contains(
                specialty.strip().casefold(),
                regex=False,
                na=False
            )
        ]

    if require_coordinates:
        data = data[
            data["latitude"].notna()
            & data["longitude"].notna()
        ]

    data = data.copy()

    data["recommendation_basis"] = (
        "Candidate selected using directory filters and "
        "evidence completeness; independent verification required."
    )

    data["price_status"] = "NOT_AVAILABLE_IN_SOURCE"
    data["availability_status"] = "NOT_AVAILABLE_IN_SOURCE"
    data["distance_status"] = "NOT_CALCULATED"
    data["clinical_quality_status"] = "NOT_ASSESSED"

    data = data.sort_values(
        by="evidence_completeness_percentage",
        ascending=False
    )

    return data.head(limit).reset_index(drop=True)
