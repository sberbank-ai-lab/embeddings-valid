import luigi
import argparse

from vector_test import ReportCollect

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, required=True)
    parser.add_argument('--conf', required=True)
    parser.add_argument('--total_cpu_count', type=int, required=True)
    args = parser.parse_args()

    task = ReportCollect(
        conf=args.conf,
        total_cpu_count=args.total_cpu_count,
    )

    luigi.build([task], workers=args.workers)
