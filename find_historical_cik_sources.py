"""Read-only WRDS metadata discovery. No market data, training or cache changes.
Run from Archive with H:\\thesis_venv\\Scripts\\python.exe.
"""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import uuid
import wrds

SQL = """
SELECT table_schema, table_name, column_name, data_type
FROM information_schema.columns
WHERE table_schema NOT IN ('pg_catalog', 'information_schema')
  AND (LOWER(column_name) LIKE '%%cik%%'
       OR LOWER(table_name) LIKE '%%cik%%')
ORDER BY table_schema, table_name, ordinal_position
"""

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--username', default='gizemyuzer')
    args = parser.parse_args()
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    out = Path('results') / f'cik_source_discovery_{stamp}_{uuid.uuid4().hex[:8]}'
    out.mkdir(parents=True, exist_ok=False)
    report = {'status': 'RUNNING', 'purpose': 'Discover accessible CIK source schemas; does not verify issuer identity'}
    db = None
    try:
        print('Connecting to WRDS; metadata only...', flush=True)
        db = wrds.Connection(wrds_username=args.username)
        candidates = db.raw_sql(SQL)
        candidates.to_csv(out / 'cik_columns.csv', index=False)
        tables = candidates[['table_schema', 'table_name']].drop_duplicates()
        schemas = []
        for row in tables.itertuples(index=False):
            # Bound parameters; metadata names are not interpolated into SQL.
            frame = db.raw_sql('''
                SELECT table_schema, table_name, column_name, data_type, ordinal_position
                FROM information_schema.columns
                WHERE table_schema = %(schema)s AND table_name = %(table)s
                ORDER BY ordinal_position
            ''', params={'schema': row.table_schema, 'table': row.table_name})
            schemas.extend(frame.to_dict(orient='records'))
        (out / 'candidate_table_schemas.json').write_text(json.dumps(schemas, indent=2, default=str), encoding='utf-8')
        report.update(status='METADATA_EXPORTED' if len(tables) else 'NO_VISIBLE_CIK_SOURCE', candidate_tables=len(tables), candidate_columns=len(candidates))
        print(candidates.to_string(index=False), flush=True)
        print('\nDONE | No training or dataset changes.', flush=True)
        print('Output:', out.resolve(), flush=True)
    except Exception as exc:
        report.update(status='FAILED', error_type=type(exc).__name__)
        raise
    finally:
        (out / 'discovery_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        if db is not None:
            db.close()

if __name__ == '__main__':
    main()
