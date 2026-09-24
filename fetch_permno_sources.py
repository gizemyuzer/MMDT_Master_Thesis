from pathlib import Path
from datetime import datetime, timezone
from uuid import uuid4
import json

import pandas as pd
import wrds


def main():
    universe_path = Path("datasets/universe.csv")
    universe = pd.read_csv(universe_path)

    required = {"ticker", "permno", "sector"}
    if not required.issubset(universe.columns):
        raise ValueError(f"Eksik kolonlar: {required - set(universe.columns)}")

    ids = pd.to_numeric(universe["permno"], errors="raise")
    if ids.isna().any() or (ids % 1 != 0).any():
        raise ValueError("Gecersiz PERMNO.")
    if ids.duplicated().any():
        raise ValueError("Tekrarli PERMNO.")

    permnos = tuple(int(x) for x in ids)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = Path("datasets") / f"permno_sources_{stamp}_{uuid4().hex[:8]}"
    output.mkdir(parents=True, exist_ok=False)

    # Orijinal evren dosyasinin birebir kopyasi.
    (output / "universe_snapshot.csv").write_bytes(universe_path.read_bytes())

    # 2009 verisi, 2010 basindaki teknik gostergelerin gecmis penceresi icin.
    params = {
        "permnos": permnos,
        "start": "2009-01-01",
        "end": "2024-12-31",
    }

    queries = {
        "daily_prices": """
            SELECT permno, date, prc, openprc, askhi, bidlo,
                   vol, shrout, cfacpr, cfacshr, ret, retx
            FROM crsp.dsf
            WHERE permno IN %(permnos)s
              AND date BETWEEN %(start)s AND %(end)s
            ORDER BY permno, date
        """,
        "delistings": """
            SELECT permno, dlstdt, dlstcd, dlret
            FROM crsp.dsedelist
            WHERE permno IN %(permnos)s
              AND dlstdt BETWEEN %(start)s AND %(end)s
            ORDER BY permno, dlstdt
        """,
        "name_history": """
            SELECT permno, namedt, nameenddt, ticker,
                   comnam, shrcd, exchcd, siccd
            FROM crsp.stocknames
            WHERE permno IN %(permnos)s
              AND namedt <= %(end)s
              AND nameenddt >= %(start)s
            ORDER BY permno, namedt
        """,
    }

    report = {
        "status": "running",
        "universe_size": len(universe),
        "start": params["start"],
        "end": params["end"],
        "output_folder": str(output),
        "tables": {},
    }

    db = None
    try:
        db = wrds.Connection(wrds_username="gizemyuzer")

        for name, sql in queries.items():
            print(f"[READ] {name}", flush=True)
            frame = db.raw_sql(sql, params=params)

            if name == "daily_prices":
                if frame.empty:
                    raise ValueError("Fiyat sorgusu bos dondu.")
                if frame.duplicated(["permno", "date"]).any():
                    raise ValueError("Tekrarli PERMNO-date fiyat kaydi.")

                found = set(frame["permno"].dropna().astype(int))
                report["permnos_without_prices"] = sorted(set(permnos) - found)
                report["price_date_min"] = str(pd.to_datetime(frame["date"]).min())
                report["price_date_max"] = str(pd.to_datetime(frame["date"]).max())

                factors = pd.to_numeric(frame["cfacpr"], errors="coerce")
                report["invalid_price_factor_rows"] = int(
                    (factors.isna() | factors.le(0)).sum()
                )

            frame.to_csv(output / f"{name}.csv", index=False)
            report["tables"][name] = {
                "rows": len(frame),
                "permnos": int(frame["permno"].nunique()),
            }
            print(f"[SAVED] {name}: {len(frame):,} rows", flush=True)

        report["status"] = "EXPORTED_NOT_YET_VALIDATED"

    except Exception as exc:
        report["status"] = "ERROR"
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise

    finally:
        try:
            if db is not None:
                db.close()
        finally:
            report_path = output / "source_report.json"
            report_path.write_text(
                json.dumps(report, indent=2),
                encoding="utf-8",
            )
            print(json.dumps(report, indent=2), flush=True)
            print(f"\nReport: {report_path}", flush=True)


if __name__ == "__main__":
    main()