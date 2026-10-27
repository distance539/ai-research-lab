"""Frozen111 final CE pass on bundled candidates; smoke uses dev only."""
import argparse
from pathlib import Path
import ce_run
from baseline_run import blend_report
from statistics_report import main as statistics
from common import read,write
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--cache',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--smoke',action='store_true');a=p.parse_args()
 ce_run.main(a.cache,a.out,a.smoke);blend_report(a.out);statistics(a.out)
 u=read(a.out/'uncertainty.json');r=u['comparisons']['blend_minus_ce'];lo,hi=r['component_percentile_95']
 write(a.out/'decision.json',{'scope':'preflight only' if a.smoke else 'frozen confirmation','practical_threshold':.01,'mean':r['mean'],'point_threshold_met':r['mean']>=.01,'interval_covers_zero':lo<=0<=hi,'interval_covers_practical_threshold':lo<=.01<=hi,'no_reselection':True})
