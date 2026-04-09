"""Build a registration -> FGVC-variant lookup table from the OpenSky database.

OpenSky's `model` column has values like 'A320 214', 'B737-800', '777-300ER',
'C172P'. We map each to one of the 100 FGVC variant strings using a small set
of regex rules. Aircraft that don't match any FGVC variant are dropped.

Output:
    data/processed/registration_to_variant.parquet
        registration (str)  ->  variant (str)
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW = REPO_ROOT / "data" / "raw" / "opensky" / "aircraft-database.csv"
OUT = REPO_ROOT / "data" / "processed" / "registration_to_variant.parquet"

# Order matters — first match wins. Use word-boundary anchors to avoid
# 'A320' matching '320 ER' etc.
RULES: list[tuple[str, str]] = [
    # Boeing 7x7 — match the dash variants first
    (r"\b747[- ]?100\b", "747-100"),
    (r"\b747[- ]?200\b", "747-200"),
    (r"\b747[- ]?300\b", "747-300"),
    (r"\b747[- ]?400\b", "747-400"),
    (r"\b757[- ]?200\b", "757-200"),
    (r"\b757[- ]?300\b", "757-300"),
    (r"\b767[- ]?200\b", "767-200"),
    (r"\b767[- ]?300\b", "767-300"),
    (r"\b767[- ]?400\b", "767-400"),
    (r"\b777[- ]?200\b", "777-200"),
    (r"\b777[- ]?300\b", "777-300"),
    (r"\b737[- ]?200\b", "737-200"),
    (r"\b737[- ]?300\b", "737-300"),
    (r"\b737[- ]?400\b", "737-400"),
    (r"\b737[- ]?500\b", "737-500"),
    (r"\b737[- ]?600\b", "737-600"),
    (r"\b737[- ]?700\b", "737-700"),
    (r"\b737[- ]?800\b", "737-800"),
    (r"\b737[- ]?900\b", "737-900"),
    (r"\b727[- ]?200\b", "727-200"),
    (r"\b707[- ]?320\b", "707-320"),
    (r"\b717\b", "Boeing 717"),
    # Airbus A3xx — allow trailing model code (A320 214, A330-200 etc)
    (r"\bA?300\s*[- ]?B4", "A300B4"),
    (r"\bA?310\b", "A310"),
    (r"\bA?318\b", "A318"),
    (r"\bA?319\b", "A319"),
    (r"\bA?320\b", "A320"),
    (r"\bA?321\b", "A321"),
    (r"\bA?330\s*[- ]?200", "A330-200"),
    (r"\bA?330\s*[- ]?300", "A330-300"),
    (r"\bA?340\s*[- ]?200", "A340-200"),
    (r"\bA?340\s*[- ]?300", "A340-300"),
    (r"\bA?340\s*[- ]?500", "A340-500"),
    (r"\bA?340\s*[- ]?600", "A340-600"),
    (r"\bA?380\b", "A380"),
    # ATR / regionals
    (r"\bATR[- ]?42\b", "ATR-42"),
    (r"\bATR[- ]?72\b", "ATR-72"),
    (r"\bAN[- ]?12\b", "An-12"),
    (r"\bBAE\s*146[- ]?200\b", "BAE 146-200"),
    (r"\bBAE\s*146[- ]?300\b", "BAE 146-300"),
    (r"\bBAE[- ]?125\b", "BAE-125"),
    (r"\bB?1900\b", "Beechcraft 1900"),
    (r"\bC[- ]?130\b", "C-130"),
    (r"\bC[- ]?47\b|\bDC[- ]?3\b.*military", "C-47"),
    (r"\bCRJ[- ]?200\b|\bCL[- ]?600[- ]?2B19\b", "CRJ-200"),
    (r"\bCRJ[- ]?700\b", "CRJ-700"),
    (r"\bCRJ[- ]?900\b", "CRJ-900"),
    (r"\bC172[A-Z]?\b|\bCESSNA\s*172\w*\b|\bSkyhawk\b", "Cessna 172"),
    (r"\bC208\b|\bCESSNA\s*208\w*\b|\bCaravan\b", "Cessna 208"),
    (r"\bC525\b|\bCJ[1-4]\b|\bCitationJet\b|\bCESSNA\s*525\w*\b", "Cessna 525"),
    (r"\bC560\b|\bCitation V\b", "Cessna 560"),
    (r"\bChallenger 600\b|\bCL[- ]?600\b", "Challenger 600"),
    (r"\bDC[- ]?10\b", "DC-10"),
    (r"\bDC[- ]?3\b", "DC-3"),
    (r"\bDC[- ]?6\b", "DC-6"),
    (r"\bDC[- ]?8\b", "DC-8"),
    (r"\bDC[- ]?9[- ]?30\b", "DC-9-30"),
    (r"\bDH[- ]?82\b|\bTiger Moth\b", "DH-82"),
    (r"\bDHC[- ]?1\b|\bChipmunk\b", "DHC-1"),
    (r"\bDHC[- ]?6\b|\bTwin Otter\b", "DHC-6"),
    (r"\bDHC[- ]?8[- ]?100\b|\bDash 8[- ]?100\b|\bQ100\b", "DHC-8-100"),
    (r"\bDHC[- ]?8[- ]?300\b|\bDash 8[- ]?300\b|\bQ300\b", "DHC-8-300"),
    (r"\bDR[- ]?400\b|\bDR400\b", "DR-400"),
    (r"\bDO\s*328\b|\bDornier 328\b", "Dornier 328"),
    (r"\bE[- ]?170\b|\bERJ[- ]?170\b|\bEMB[- ]?170\b", "E-170"),
    (r"\bE[- ]?190\b|\bERJ[- ]?190\b|\bEMB[- ]?190\b", "E-190"),
    (r"\bE[- ]?195\b|\bERJ[- ]?195\b|\bEMB[- ]?195\b", "E-195"),
    (r"\bEMB[- ]?120\b|\bBrasilia\b", "EMB-120"),
    (r"\bERJ[- ]?135\b|\bEMB[- ]?135\b", "ERJ 135"),
    (r"\bERJ[- ]?145\b|\bEMB[- ]?145\b", "ERJ 145"),
    (r"\bLegacy 600\b|\bEMB[- ]?135BJ\b", "Embraer Legacy 600"),
    (r"\bTyphoon\b|\bEF[- ]?2000\b", "Eurofighter Typhoon"),
    (r"\bF[- ]?16\b", "F-16A/B"),
    (r"\bF[- ]?18\b|\bFA[- ]?18\b|\bF/A[- ]?18\b", "F/A-18"),
    (r"\bFalcon 2000\b", "Falcon 2000"),
    (r"\bFalcon 900\b", "Falcon 900"),
    (r"\bF100\b|\bFokker 100\b", "Fokker 100"),
    (r"\bF50\b|\bFokker 50\b", "Fokker 50"),
    (r"\bF70\b|\bFokker 70\b", "Fokker 70"),
    (r"\bGlobal Express\b|\bBD[- ]?700\b", "Global Express"),
    (r"\bG[- ]?IV\b|\bGulfstream IV\b|\bG[- ]?450\b", "Gulfstream IV"),
    (r"\bG[- ]?V\b|\bGulfstream V\b|\bG[- ]?550\b", "Gulfstream V"),
    (r"\bHawk\b", "Hawk T1"),
    (r"\bIL[- ]?76\b", "Il-76"),
    (r"\bL[- ]?1011\b|\bTriStar\b", "L-1011"),
    (r"\bMD[- ]?11\b", "MD-11"),
    (r"\bMD[- ]?80\b|\bMD[- ]?81\b|\bMD[- ]?82\b|\bMD[- ]?83\b", "MD-80"),
    (r"\bMD[- ]?87\b", "MD-87"),
    (r"\bMD[- ]?90\b", "MD-90"),
    (r"\bMetroliner\b|\bSA[- ]?227\b", "Metroliner"),
    (r"\bB200\b|\bSuper King Air\b|\bKing Air 200\b", "Model B200"),
    (r"\bPA[- ]?28\b|\bCherokee\b", "PA-28"),
    (r"\bSR[- ]?20\b|\bSR20\b", "SR-20"),
    (r"\bSaab 2000\b|\bSF[- ]?2000\b", "Saab 2000"),
    (r"\bSaab 340\b|\bSF[- ]?340\b", "Saab 340"),
    (r"\bSpitfire\b", "Spitfire"),
    (r"\bTornado\b|\bGR[- ]?1\b|\bGR[- ]?4\b", "Tornado"),
    (r"\bTU[- ]?134\b", "Tu-134"),
    (r"\bTU[- ]?154\b", "Tu-154"),
    (r"\bYak[- ]?42\b", "Yak-42"),
]


def map_model_to_variant(model: str) -> str | None:
    if not isinstance(model, str) or not model:
        return None
    s = model.upper()
    for pattern, variant in RULES:
        if re.search(pattern, s, flags=re.IGNORECASE):
            return variant
    return None


def main() -> None:
    print(f"Reading {RAW} ...")
    df = pd.read_csv(
        RAW,
        usecols=["registration", "model", "manufacturerName"],
        dtype=str,
        quotechar="'",
        on_bad_lines="skip",
        engine="python",
    )
    print(f"  {len(df)} raw rows")

    df = df.dropna(subset=["registration", "model"])
    df["registration"] = df["registration"].str.strip().str.upper()
    df = df[df["registration"].str.len().between(3, 8)]

    df["variant"] = df["model"].apply(map_model_to_variant)
    df = df.dropna(subset=["variant"])
    print(f"  {len(df)} rows mapped to one of the 100 FGVC variants")
    print("\nTop variants:")
    print(df["variant"].value_counts().head(15).to_string())

    # de-dup: a registration may appear multiple times (history); take the most common variant
    out = (
        df.groupby("registration")["variant"]
        .agg(lambda s: s.value_counts().index[0])
        .reset_index()
    )
    print(f"\n{len(out)} unique registrations")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT, index=False)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
