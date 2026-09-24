"""Offline price preparation. Does not train or overwrite old data."""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def numeric_id(frame):
    x = pd.to_numeric(frame["permno"], errors="raise")
    if x.isna().any() or (x % 1 != 0).any():
        raise ValueError("Invalid PERMNO")
    frame["permno"] = x.astype("int64")
    return frame


def prepare(prices, universe, delist):
    p = numeric_id(prices.copy())
    u = numeric_id(universe.copy())
    d = numeric_id(delist.copy())

    if u["permno"].duplicated().any():
        raise ValueError("Duplicate universe PERMNO")

    universe_ids = set(u.permno)
    if not set(p.permno).issubset(universe_ids):
        raise ValueError("Prices contain securities outside universe")
    if not set(d.permno).issubset(universe_ids):
        raise ValueError("Delistings contain securities outside universe")

    p["date"] = pd.to_datetime(p["date"], errors="raise")
    d["dlstdt"] = pd.to_datetime(d["dlstdt"], errors="raise")

    if p.date.isna().any() or d.dlstdt.isna().any():
        raise ValueError("Missing source dates")
    if p.duplicated(["permno", "date"]).any():
        raise ValueError("Duplicate PERMNO/date prices")

    p = p.sort_values(["permno", "date"]).reset_index(drop=True)

    numeric_cols = [
        "prc", "openprc", "askhi", "bidlo", "vol",
        "cfacpr", "cfacshr", "ret", "retx",
    ]
    for col in numeric_cols:
        # Preserve original fields, including missing-return codes.
        p[col + "_numeric"] = pd.to_numeric(p[col], errors="coerce")

    factor = p["cfacpr_numeric"]
    good_factor = np.isfinite(factor) & factor.gt(0)

    price_columns = [
        ("prc", "Close"),
        ("openprc", "Open"),
        ("askhi", "High"),
        ("bidlo", "Low"),
    ]
    for raw, adjusted in price_columns:
        value = p[raw + "_numeric"]
        valid = good_factor & np.isfinite(value) & value.abs().gt(0)
        p[adjusted] = (value.abs() / factor).where(valid)

    # Intermediate output only: volume adjustment remains pending.
    p["VolumeRaw"] = p["vol_numeric"]
    p["CRSP_RET"] = p["ret_numeric"].where(p["ret_numeric"].ge(-1))
    p["CRSP_RETX"] = p["retx_numeric"].where(p["retx_numeric"].ge(-1))

    p["price_usable"] = p["Close"].notna()
    p["ohlc_usable"] = (
        p[["Open", "High", "Low", "Close"]].notna().all(axis=1)
    )
    p["ohlc_order_invalid"] = p["ohlc_usable"] & (
        p["High"].lt(p[["Open", "Close", "Low"]].max(axis=1))
        | p["Low"].gt(p[["Open", "Close", "High"]].min(axis=1))
    )

    grouped = p.groupby("permno", sort=False)
    p["AdjustedPriceReturn"] = p["Close"] / grouped["Close"].shift() - 1
    p["days_since_previous_row"] = grouped["date"].diff().dt.days

    valid_comparison = (
        p["AdjustedPriceReturn"].notna() & p["CRSP_RETX"].notna()
    )
    mismatch = valid_comparison & (
        p["AdjustedPriceReturn"] - p["CRSP_RETX"]
    ).abs().gt(0.001)
    p["retx_difference_gt_10bp"] = mismatch

    meta = u[["permno", "ticker", "sector"]].rename(
        columns={"ticker": "FormationTicker", "sector": "Sector"}
    )
    p = p.merge(meta, on="permno", how="left", validate="many_to_one")

    d["code_numeric"] = pd.to_numeric(d["dlstcd"], errors="coerce")
    d["dlret_numeric"] = pd.to_numeric(d["dlret"], errors="coerce")
    d["dlret_usable"] = (
        np.isfinite(d["dlret_numeric"]) & d["dlret_numeric"].ge(-1)
    )
    d["record_class"] = np.where(
        d.code_numeric.eq(100),
        "CODE_100_NO_EXIT_APPLIED",
        "REVIEW_OTHER_CODE",
    )

    last = p.groupby("permno")["date"].max().rename("last_price_date")
    d = d.merge(last, on="permno", how="left", validate="many_to_one")
    d["prices_after_record"] = d.last_price_date.gt(d.dlstdt)
    d["same_day_price_exists"] = pd.MultiIndex.from_frame(
        d[["permno", "dlstdt"]]
    ).isin(pd.MultiIndex.from_frame(p[["permno", "date"]]))

    other = ~d.code_numeric.eq(100)

    report = {
        "status": "PRICE_BASE_BUILT_NOT_TRAINING_READY",
        "rows": len(p),
        "permnos": int(p.permno.nunique()),
        "missing_universe_permnos": sorted(universe_ids - set(p.permno)),
        "invalid_price_factor_rows": int((~good_factor).sum()),
        "missing_adjusted_close_rows": int(p.Close.isna().sum()),
        "missing_ohlc_rows": int((~p.ohlc_usable).sum()),
        "ohlc_order_invalid_rows": int(p.ohlc_order_invalid.sum()),
        "missing_or_invalid_ret_rows": int(p.CRSP_RET.isna().sum()),
        "retx_comparison_rows": int(valid_comparison.sum()),
        "retx_difference_gt_10bp_rows": int(mismatch.sum()),
        "delisting_code_counts": {
            str(k): int(v)
            for k, v in d.dlstcd.value_counts(dropna=False).items()
        },
        "code100_records": int((~other).sum()),
        "other_records": int(other.sum()),
        "other_records_missing_valid_dlret": int(
            (other & ~d.dlret_usable).sum()
        ),
        "other_records_with_later_prices": int(
            (other & d.prices_after_record).sum()
        ),
        "other_records_same_day_price": int(
            (other & d.same_day_price_exists).sum()
        ),
        "notes": [
            "Raw sources retained; no row filtering or return imputation.",
            "No delisting returns applied, including same-day events.",
            "VolumeRaw is unadjusted; feature generation remains pending.",
            "Adjusted price returns are not dividend-inclusive total returns.",
            "FormationTicker is metadata, not an accounting/text join key.",
            "Historical adjustment factors must not be predictive features.",
        ],
    }
    return p, d, report


def self_test():
    prices = pd.DataFrame({
        "permno": [1, 1],
        "date": ["2020-01-02", "2020-01-03"],
        "prc": [100, 50],
        "openprc": [100, 50],
        "askhi": [100, 50],
        "bidlo": [100, 50],
        "vol": [100, 200],
        "cfacpr": [2, 1],
        "cfacshr": [2, 1],
        "ret": [0, 0],
        "retx": [0, 0],
    })
    universe = pd.DataFrame({
        "permno": [1],
        "ticker": ["EXAMPLE"],
        "sector": ["Test"],
    })
    delist = pd.DataFrame({
        "permno": [1],
        "dlstdt": ["2020-01-03"],
        "dlstcd": [100],
        "dlret": [np.nan],
    })

    result, events, report = prepare(prices, universe, delist)
    assert result.Close.tolist() == [50, 50]
    assert result.AdjustedPriceReturn.iloc[1] == 0
    assert report["code100_records"] == 1
    assert report["other_records"] == 0
    assert events.same_day_price_exists.all()

    missing = prices.copy()
    missing.loc[1, "prc"] = np.nan
    result, _, _ = prepare(missing, universe, delist)
    assert pd.isna(result.Close.iloc[1]) and len(result) == 2

    try:
        prepare(
            pd.concat([prices, prices.iloc[:1]]),
            universe,
            delist,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("Duplicate was not rejected")

    print("PASS | split, code100, missing prices, duplicate rejection")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        default="datasets/permno_sources_20260917T191937Z_f78c6a09",
    )
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return

    source = Path(args.source)
    paths = {
        name: source / f"{name}.csv"
        for name in ["daily_prices", "universe_snapshot", "delistings"]
    }

    hashes = {name: digest(path) for name, path in paths.items()}
    frames = {name: pd.read_csv(path) for name, path in paths.items()}

    prices, events, report = prepare(
        frames["daily_prices"],
        frames["universe_snapshot"],
        frames["delistings"],
    )

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = source / f"price_base_{stamp}_{uuid4().hex[:8]}"
    output.mkdir(exist_ok=False)

    report.update(
        source_hashes=hashes,
        output_folder=str(output),
    )

    prices.to_csv(output / "adjusted_price_base.csv", index=False)
    events.to_csv(output / "delisting_review.csv", index=False)
    prices.loc[prices.retx_difference_gt_10bp].to_csv(
        output / "return_mismatches.csv",
        index=False,
    )

    after = {name: digest(path) for name, path in paths.items()}
    if hashes != after:
        raise RuntimeError("Source files changed during run")

    report_path = output / "price_base_report.json"
    report_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))
    print("\nReport:", report_path)


if __name__ == "__main__":
    main()