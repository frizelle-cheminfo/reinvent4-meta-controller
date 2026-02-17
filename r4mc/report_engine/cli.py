"""Command-line interface for r4mc report engine."""
import argparse
import sys
import logging
from pathlib import Path

from .report.build import ReportBuilder
from .report.medchem_handoff import MedChemHandoffReport
from .utils.logging import setup_logger
from .io.discover import RunDiscovery


def cmd_build(args):
    """Build report command."""
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logger = setup_logger('r4mc.report', level=log_level)

    logger.info("Starting r4mc report build...")
    logger.info(f"Run directory: {args.run_dir}")
    logger.info(f"Output directory: {args.out_dir}")

    if args.molport_csv:
        logger.info(f"MolPort CSV: {args.molport_csv}")

    try:
        # Build report
        builder = ReportBuilder(
            run_dir=args.run_dir,
            out_dir=args.out_dir,
            molport_csv=args.molport_csv,
            top_k_per_episode=args.top_k,
            molport_similarity_threshold=args.molport_similarity
        )

        builder.build()

        logger.info("Report generation successful!")
        logger.info(f"Output directory: {args.out_dir}")

        return 0

    except Exception as e:
        logger.error(f"Report generation failed: {e}", exc_info=args.verbose)
        return 1


def cmd_medchem(args):
    """Generate medchem handoff report command."""
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logger = setup_logger('r4mc.report', level=log_level)

    logger.info("Starting medchem handoff report generation...")
    logger.info(f"Run directory: {args.run_dir}")
    logger.info(f"Output directory: {args.out_dir}")

    if args.molport_dir:
        logger.info(f"MolPort directory: {args.molport_dir}")

    try:
        # Find MOLPORT CSVs if directory provided
        molport_csvs = []
        if args.molport_dir:
            molport_dir = Path(args.molport_dir)
            if molport_dir.exists():
                molport_csvs = [str(f) for f in molport_dir.glob('*.csv')]
                logger.info(f"Found {len(molport_csvs)} MOLPORT CSV files")
            else:
                logger.warning(f"MolPort directory not found: {molport_dir}")

        # Build medchem handoff report
        report = MedChemHandoffReport(
            run_dir=args.run_dir,
            out_dir=args.out_dir,
            molport_csvs=molport_csvs,
            top_n_per_batch=args.top_n,
            molport_similarity_threshold=args.molport_similarity,
            molport_top_k=args.molport_top_k
        )

        report.build()

        logger.info("Medchem handoff report generation successful!")
        logger.info(f"Output directory: {args.out_dir}")
        logger.info(f"Open index.html in your browser to view reports")

        return 0

    except Exception as e:
        logger.error(f"Medchem report generation failed: {e}", exc_info=args.verbose)
        return 1


def cmd_doctor(args):
    """Doctor command for diagnostics."""
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logger = setup_logger('r4mc.report', level=log_level)

    logger.info("Running r4mc report doctor...")
    logger.info(f"Checking run directory: {args.run_dir}")

    try:
        run_dir = Path(args.run_dir)

        # Check if directory exists
        if not run_dir.exists():
            logger.error(f"Run directory does not exist: {run_dir}")
            return 1

        if not run_dir.is_dir():
            logger.error(f"Path is not a directory: {run_dir}")
            return 1

        logger.info("✓ Run directory exists")

        # Discover files
        logger.info("Discovering files...")
        discovery = RunDiscovery(str(run_dir))
        result = discovery.discover()

        # Print discovery results
        n_episodes = len(result['episodes'])
        logger.info(f"✓ Found {n_episodes} episodes")

        if result['config_files']:
            logger.info(f"✓ Found {len(result['config_files'])} config files")

        if result['log_files']:
            logger.info(f"✓ Found {len(result['log_files'])} log files")

        if result['seed_banks']:
            logger.info(f"✓ Found {len(result['seed_banks'])} seed bank files")

        # Print warnings/errors
        if result['diagnostics']['warnings']:
            logger.warning(f"⚠ {len(result['diagnostics']['warnings'])} warnings:")
            for warning in result['diagnostics']['warnings']:
                logger.warning(f"  - {warning}")

        if result['diagnostics']['errors']:
            logger.error(f"✗ {len(result['diagnostics']['errors'])} errors:")
            for error in result['diagnostics']['errors']:
                logger.error(f"  - {error}")
            return 1

        # Check dependencies
        logger.info("Checking dependencies...")
        try:
            import pandas
            logger.info(f"✓ pandas {pandas.__version__}")
        except ImportError:
            logger.error("✗ pandas not installed")
            return 1

        try:
            import rdkit
            logger.info(f"✓ rdkit {rdkit.__version__}")
        except ImportError:
            logger.error("✗ rdkit not installed")
            return 1

        logger.info("✓ All checks passed!")
        return 0

    except Exception as e:
        logger.error(f"Doctor check failed: {e}", exc_info=args.verbose)
        return 1


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Albion Report: Behavioral reporting for REINVENT4 meta-controller runs',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--version', action='version', version='%(prog)s 0.1.0')

    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # Build command
    build_parser = subparsers.add_parser('build', help='Build report from run directory')
    build_parser.add_argument('--run_dir', required=True, help='Path to run directory')
    build_parser.add_argument('--out_dir', required=True, help='Path to output directory')
    build_parser.add_argument('--molport_csv', help='Optional path to MolPort CSV')
    build_parser.add_argument('--top_k', type=int, default=1000,
                              help='Number of top molecules per episode (default: 1000)')
    build_parser.add_argument('--molport_similarity', type=float, default=0.7,
                              help='Minimum Tanimoto similarity for MolPort mapping (default: 0.7)')
    build_parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')

    # MedChem handoff command
    medchem_parser = subparsers.add_parser('medchem', help='Generate medchem handoff reports')
    medchem_parser.add_argument('--run_dir', required=True, help='Path to run directory')
    medchem_parser.add_argument('--out_dir', required=True, help='Path to output directory')
    medchem_parser.add_argument('--molport_dir', help='Directory containing MOLPORT CSV files')
    medchem_parser.add_argument('--top_n', type=int, default=20,
                                help='Number of top molecules per batch (default: 20)')
    medchem_parser.add_argument('--molport_similarity', type=float, default=0.5,
                                help='Minimum Tanimoto similarity for MOLPORT mapping (default: 0.5)')
    medchem_parser.add_argument('--molport_top_k', type=int, default=3,
                                help='Number of MOLPORT analogues per molecule (default: 3)')
    medchem_parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')

    # Doctor command
    doctor_parser = subparsers.add_parser('doctor', help='Run diagnostics on run directory')
    doctor_parser.add_argument('--run_dir', required=True, help='Path to run directory')
    doctor_parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')

    # Parse arguments
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    # Route to command
    if args.command == 'build':
        return cmd_build(args)
    elif args.command == 'medchem':
        return cmd_medchem(args)
    elif args.command == 'doctor':
        return cmd_doctor(args)
    else:
        parser.print_help()
        return 1


if __name__ == '__main__':
    sys.exit(main())
