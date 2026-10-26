"""110 entry: fresh official model inference, rank controls, uncertainty report."""
import argparse
from pathlib import Path
import ce_run
from baseline_run import blend_report
from statistics_report import main as statistics
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--cache',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--smoke',action='store_true');a=p.parse_args();ce_run.main(a.cache,a.out,a.smoke);blend_report(a.out);statistics(a.out)
