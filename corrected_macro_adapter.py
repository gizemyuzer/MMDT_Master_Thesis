"""Extend the corrected I+II loader without changing labels, windows or its scaling."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
import run_corrected_smoke as base
from build_corrected_macro import MACRO, self_test as macro_test
TECH=base.TECH
FUND=base.FUND+MACRO
sha=base.sha
EndpointDataset=base.EndpointDataset
window_positions=base.window_positions
MACRO_PATH=Path('datasets/corrected_macro_v1/macro_daily.csv')
def self_test():
    base.self_test();macro_test()
def load_corrected(tech_path,fund_path,max_permnos=None):
    frame,values,state,pos,audit=base.load_corrected(tech_path,fund_path,max_permnos)
    path=Path(MACRO_PATH)
    report=json.loads(path.with_name('macro_build_report.json').read_text(encoding='utf-8'))
    if report['status']!='PASS' or report['technical_sha256']!=audit['source_sha256']['technical']:
        raise ValueError('Macro panel failed or belongs to different technical input')
    if sha(path)!=report['macro_sha256']:raise ValueError('Macro checksum changed')
    base.check_allowlist(path.with_name('macro_feature_columns.csv'),MACRO)
    macro=pd.read_csv(path,parse_dates=['date','MacroSourceSession'])
    if macro.date.duplicated().any():raise ValueError('Duplicate macro dates')
    frame=frame.merge(macro[['date','MacroSourceSession']+MACRO],on='date',how='left',validate='many_to_one',sort=False)
    known=frame.MacroSourceSession.notna()
    if (frame.loc[known,'MacroSourceSession']>=frame.loc[known,'date']).any():raise ValueError('Non-lagged macro inputs')
    raw=frame[MACRO].to_numpy(dtype=float)
    mask=frame.TrainEndpoint.to_numpy(dtype=bool)
    if not np.isfinite(raw[mask]).all():raise ValueError('Missing training macro input')
    scaler=RobustScaler().fit(raw[mask])
    extra=scaler.transform(raw).astype(np.float32)
    values=np.concatenate([values,extra],axis=1)
    for flag in base.FLAGS:base.window_positions(frame,values,flag)
    state.update(macro_scaler=scaler,macro_columns=MACRO,columns=TECH+FUND,
                 base_columns=TECH+base.FUND,macro_protocol=report['protocol'])
    audit['source_sha256']['macro']=sha(path)
    audit['macro_columns']=MACRO
    audit['fundamental_columns']=base.FUND
    audit['second_stream_columns']=FUND
    audit['macro_protocol']=report['protocol']
    return frame,values,state,pos,audit
