# Data

The repository does **not** ship the data. The cleaned dataset has about 2.9 million user-level
rows and 66+ columns, far above GitHub's 100 MB per-file limit. Two ways to obtain it:

## 1. Rebuild it from the public BAAC files (recommended)

1. Download the yearly files *"Bases de données annuelles des accidents corporels de la circulation routière"*
   (BAAC, published by the ONISR on [data.gouv.fr](https://www.data.gouv.fr/), Open Licence v2.0) for 2005–2024.
   Each year has four CSV files: **caractéristiques**, **lieux**, **véhicules**, **usagers**.
2. Place them as:

```
data/raw/
  2005/  caracteristiques-2005.csv  lieux-2005.csv  vehicules-2005.csv  usagers-2005.csv
  2006/  ...
  ...
  2024/  ...
```

   File names do not have to match exactly: the matching is keyword-based and tolerant (it
   copes with the `carcteristiques` typo of the official 2021/2022 files and with the
   `Accident_Id` / `Num_Acc` naming difference). Encodings (UTF-8 / Latin-1) and separators
   (`,` `;` tab `|`) are detected automatically.
3. Run `python scripts/01_preprocessing.py`. It writes `data/processed/parquet/`
   (hive-partitioned by `annee_dataset`) and `data/processed/csv/` (one file per year).

## 2. Use an archived copy

If you have a copy of the cleaned dataset (for instance a Zenodo archive, DOI *to be added*), place it as
`data/processed/parquet/annee_dataset=<year>/*.parquet` (or `data/processed/csv/baac_master_<year>.csv`).
Steps 02 to 05 read from there.

## Data dictionary (variables used in the analyses)

Original BAAC variables keep their French names.

| Variable | Meaning | Used in |
|---|---|---|
| `grav` | Severity as coded by BAAC (1 unharmed, 2 killed, 3 hospitalised, 4 light injury; **not ordered**) | target (recoded) |
| `lum`, `atm`, `surf`, `prof`, `plan`, `infra`, `situ`, `vosp` | lighting, weather, road surface, longitudinal profile, layout, infrastructure, situation, reserved lane | rules, prediction |
| `col`, `int`, `agg` | collision type, intersection type, in / outside built-up area | rules, prediction |
| `catr`, `circ`, `nbv`, `vma`, `lartpc`, `larrout` | road category, traffic regime, number of lanes, speed limit, central-reservation width, roadway width | rules, PCA, prediction |
| `catv`, `obs`, `obsm`, `choc`, `manv`, `motor`, `senc` | vehicle category, fixed obstacle struck, mobile obstacle struck, initial impact point, manoeuvre, motorisation, traffic direction | rules, prediction |
| `catu`, `place`, `sexe`, `trajet`, `an_nais` | user category (driver / passenger / pedestrian), seat, sex, trip purpose, birth year | rules, prediction |
| `locp`, `actp`, `etatp` | pedestrian location, action, and state (accompanied / alone) | prediction (`extended_ceinture`) |
| `secu`, `secu1`–`secu3` | safety equipment (single column before 2019, three columns after) | rebuilt as `equip_*` |
| `lat`, `long` | GPS coordinates | PCA, spatial clustering |
| `dep`, `com` | department and municipality code (`com` is unique only within a department) | exposure normalisation |
| `an`, `mois`, `jour`, `hrmn` | date and time | calendar, ST-DBSCAN |

Variables derived by the pipeline:

| Variable | Definition |
|---|---|
| `annee_dataset` | vintage of the source file (hive partition column) |
| `heure_accident` | hour (0–23) parsed from `hrmn` |
| `age` | `an` − `an_nais`, set to NaN outside 0–110 |
| `jour_semaine`, `weekend` | day of week (0 Monday … 6 Sunday) and weekend flag |
| `equip_ceinture` | seatbelt used; reliable over 2005–2024 |
| `equip_casque`, `equip_enfant`, `equip_gilet`, `equip_airbag_2rm`, `equip_gants_2rm` | other equipment; **structurally False before 2019** (information absent, not "not worn") |
| `gravite_ordinale` | 1 unharmed, 2 light injury, 3 hospitalised, 4 killed |
| `accident_grave` | `gravite_ordinale` in {3, 4} (association rules) |
| `cluster_dbscan` | DBSCAN spatial cluster id (−1 = noise) |

Missing values: BAAC sentinels (empty, `.`, `-1`) become NaN. Continuous variables (`an_nais`, `vma`, `nbv`,
`lartpc`, `larrout`, `age`) are imputed with `IterativeImputer`; nominal columns get the explicit
category `Non renseigné`. **Coordinates are never imputed.**
