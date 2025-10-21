"""
Reporting and Analysis Utilities for YOLOv11 Training
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import argparse


class TrainingAnalyzer:
    """Analyze and visualize training results"""

    def __init__(self, log_dir):
        self.log_dir = Path(log_dir)
        self.sessions = []
        self.load_sessions()

    def load_sessions(self):
        """Load all training sessions from log directory"""
        for session_dir in self.log_dir.glob('train_*'):
            metrics_file = session_dir / 'metrics.json'
            if metrics_file.exists():
                with open(metrics_file, 'r') as f:
                    self.sessions.append(json.load(f))

        print(f"Loaded {len(self.sessions)} training sessions")

    def plot_training_curves(self, session_id=None, output_dir='./reports'):
        """
        Plot training curves (loss, metrics over epochs)

        Args:
            session_id: Specific session to plot (None for latest)
            output_dir: Directory to save plots
        """

        if not self.sessions:
            print("No training sessions found")
            return

        # Get session
        if session_id:
            session = next((s for s in self.sessions if s['session_id'] == session_id), None)
            if not session:
                print(f"Session {session_id} not found")
                return
        else:
            session = self.sessions[-1]  # Latest session

        print(f"Plotting session: {session['session_id']}")

        # Extract epoch data
        epochs = session.get('epochs', [])
        if not epochs:
            print("No epoch data available")
            return

        # Create output directory
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (15, 10)

        # Note: Ultralytics YOLO training results are typically saved in runs/detect/train
        # This is a placeholder for custom epoch logging
        print("Training curves are saved by Ultralytics in runs/detect/train/results.png")
        print(f"Session config: {json.dumps(session['config'], indent=2)}")


class ValidationAnalyzer:
    """Analyze validation and test results"""

    def __init__(self, log_dir, report_dir):
        self.log_dir = Path(log_dir)
        self.report_dir = Path(report_dir)
        self.val_sessions = []
        self.load_sessions()

    def load_sessions(self):
        """Load all validation sessions"""
        for session_dir in self.log_dir.glob('val_*'):
            metrics_file = session_dir / 'metrics.json'
            if metrics_file.exists():
                with open(metrics_file, 'r') as f:
                    self.val_sessions.append(json.load(f))

        print(f"Loaded {len(self.val_sessions)} validation sessions")

    def compare_sessions(self, output_file=None):
        """
        Compare multiple validation sessions

        Args:
            output_file: Output file for comparison report
        """

        if not self.val_sessions:
            print("No validation sessions found")
            return

        # Extract metrics from all sessions
        comparison_data = []

        for session in self.val_sessions:
            session_info = {
                'session_id': session['session_id'],
                'timestamp': session['start_time'],
                'model': session['config'].get('model_path', 'unknown'),
            }
            session_info.update(session.get('results', {}))
            comparison_data.append(session_info)

        # Create DataFrame
        df = pd.DataFrame(comparison_data)

        # Sort by timestamp
        df = df.sort_values('timestamp')

        print("\nValidation Sessions Comparison:")
        print("=" * 80)
        print(df.to_string())
        print("=" * 80)

        # Save to CSV
        if output_file is None:
            output_file = self.report_dir / 'validation_comparison.csv'
        else:
            output_file = Path(output_file)

        output_file.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_file, index=False)
        print(f"\nComparison saved to: {output_file}")

        return df

    def plot_metrics_comparison(self, metrics=['mAP50', 'mAP50-95', 'precision', 'recall'],
                                output_dir=None):
        """
        Plot comparison of metrics across sessions

        Args:
            metrics: List of metrics to plot
            output_dir: Directory to save plots
        """

        if not self.val_sessions:
            print("No validation sessions found")
            return

        if output_dir is None:
            output_dir = self.report_dir
        else:
            output_dir = Path(output_dir)

        output_dir.mkdir(parents=True, exist_ok=True)

        # Extract data
        data = []
        for session in self.val_sessions:
            session_data = {'session_id': session['session_id'][:8]}  # Short ID
            session_data.update(session.get('results', {}))
            data.append(session_data)

        df = pd.DataFrame(data)

        # Filter metrics that exist
        available_metrics = [m for m in metrics if m in df.columns]

        if not available_metrics:
            print("No metrics available for plotting")
            return

        # Create plots
        n_metrics = len(available_metrics)
        fig, axes = plt.subplots(1, n_metrics, figsize=(6 * n_metrics, 5))

        if n_metrics == 1:
            axes = [axes]

        for idx, metric in enumerate(available_metrics):
            ax = axes[idx]
            df.plot(x='session_id', y=metric, kind='bar', ax=ax, legend=False)
            ax.set_title(f'{metric}', fontsize=14, fontweight='bold')
            ax.set_xlabel('Session', fontsize=12)
            ax.set_ylabel('Value', fontsize=12)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save plot
        plot_file = output_dir / 'metrics_comparison.png'
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"Metrics comparison plot saved to: {plot_file}")

        plt.close()


def generate_summary_report(log_dir='./logs', report_dir='./reports', output_file=None):
    """
    Generate comprehensive summary report

    Args:
        log_dir: Directory containing logs
        report_dir: Directory for reports
        output_file: Output file path
    """

    log_dir = Path(log_dir)
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    if output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = report_dir / f'summary_report_{timestamp}.txt'
    else:
        output_file = Path(output_file)

    with open(output_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("YOLOv11 TRAINING & VALIDATION SUMMARY REPORT\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        # Training sessions
        f.write("-" * 80 + "\n")
        f.write("TRAINING SESSIONS\n")
        f.write("-" * 80 + "\n\n")

        train_sessions = list(log_dir.glob('train_*/metrics.json'))
        f.write(f"Total sessions: {len(train_sessions)}\n\n")

        for session_file in sorted(train_sessions)[-5:]:  # Last 5 sessions
            with open(session_file, 'r') as sf:
                session = json.load(sf)

            f.write(f"Session: {session['session_id']}\n")
            f.write(f"  Started: {session['start_time']}\n")
            f.write(f"  Ended: {session.get('end_time', 'N/A')}\n")

            config = session.get('config', {})
            f.write(f"  Config: epochs={config.get('epochs')}, "
                   f"batch={config.get('batch')}, "
                   f"model={config.get('model_size')}\n\n")

        # Validation sessions
        f.write("\n" + "-" * 80 + "\n")
        f.write("VALIDATION SESSIONS\n")
        f.write("-" * 80 + "\n\n")

        val_sessions = list(log_dir.glob('val_*/metrics.json'))
        f.write(f"Total sessions: {len(val_sessions)}\n\n")

        for session_file in sorted(val_sessions)[-5:]:  # Last 5 sessions
            with open(session_file, 'r') as sf:
                session = json.load(sf)

            f.write(f"Session: {session['session_id']}\n")
            results = session.get('results', {})

            for key, value in results.items():
                if isinstance(value, (int, float)):
                    f.write(f"  {key}: {value:.4f}\n")

            f.write("\n")

        f.write("=" * 80 + "\n")

    print(f"Summary report generated: {output_file}")
    return output_file


def main():
    parser = argparse.ArgumentParser(description='Reporting and Analysis Utilities')
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')

    # Training analysis
    train_parser = subparsers.add_parser('analyze-training',
                                         help='Analyze training sessions')
    train_parser.add_argument('--log-dir', default='./logs',
                             help='Log directory')
    train_parser.add_argument('--session-id', help='Specific session ID')
    train_parser.add_argument('--output-dir', default='./reports',
                             help='Output directory')

    # Validation comparison
    val_parser = subparsers.add_parser('compare-validation',
                                       help='Compare validation sessions')
    val_parser.add_argument('--log-dir', default='./logs',
                           help='Log directory')
    val_parser.add_argument('--report-dir', default='./reports',
                           help='Report directory')
    val_parser.add_argument('--plot', action='store_true',
                           help='Generate comparison plots')

    # Summary report
    summary_parser = subparsers.add_parser('summary',
                                          help='Generate summary report')
    summary_parser.add_argument('--log-dir', default='./logs',
                               help='Log directory')
    summary_parser.add_argument('--report-dir', default='./reports',
                               help='Report directory')

    args = parser.parse_args()

    if args.command == 'analyze-training':
        analyzer = TrainingAnalyzer(args.log_dir)
        analyzer.plot_training_curves(args.session_id, args.output_dir)

    elif args.command == 'compare-validation':
        analyzer = ValidationAnalyzer(args.log_dir, args.report_dir)
        analyzer.compare_sessions()
        if args.plot:
            analyzer.plot_metrics_comparison()

    elif args.command == 'summary':
        generate_summary_report(args.log_dir, args.report_dir)

    else:
        parser.print_help()


if __name__ == '__main__':
    main()
