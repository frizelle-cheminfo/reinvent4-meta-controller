"""Command-line interface for the meta-controller."""

import logging
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

from . import __version__
from .arms import filter_available_arms, get_arm_catalog
from .config import ControllerConfig
from .controller import MetaController
from .episode import EpisodeRunner
from .metrics import MetricsCalculator, compute_episode_quality
from .reporting import ReportGenerator
from .seeds import (
    SeedSelector,
    initialize_seed_bank_from_actives,
    load_reference_actives,
    update_seed_bank,
)
from .state import ControllerState, EpisodeMetrics, EpisodeRecord, MoleculeRecord
from .utils import check_rdkit

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(rich_tracebacks=True, markup=True)],
)
logger = logging.getLogger(__name__)

# Setup Rich console
console = Console()

# Typer app
app = typer.Typer(
    name="reinvent-meta",
    help="REINVENT4 Meta-Controller: Adaptive exploration across multiple priors and regimes",
    add_completion=False,
)


@app.command()
def init(
    campaign: str = typer.Option(..., help="Campaign name"),
    seed: int = typer.Option(42, help="Random seed for reproducibility"),
    config: Optional[Path] = typer.Option(None, help="Path to controller.yaml config file"),
):
    """Initialize a new campaign."""
    console.print(f"[bold blue]Initializing campaign: {campaign}[/bold blue]")

    # Load or create config
    if config and config.exists():
        ctrl_config = ControllerConfig.load(config)
        ctrl_config.campaign_name = campaign
        ctrl_config.seed = seed
        console.print(f"Loaded config from {config}")
    else:
        ctrl_config = ControllerConfig(campaign_name=campaign, seed=seed)
        console.print("Using default config")

    # Check RDKit
    if check_rdkit():
        console.print("[green]✓[/green] RDKit available - full functionality enabled")
    else:
        console.print(
            "[yellow]⚠[/yellow] RDKit not available - some features will be limited"
        )

    # Create campaign directory
    campaign_dir = ctrl_config.output_dir / campaign
    campaign_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    config_path = campaign_dir / "controller.yaml"
    ctrl_config.save(config_path)
    console.print(f"Saved config to {config_path}")

    # Initialize state
    state = ControllerState(campaign_name=campaign, seed=seed)

    # Initialize seed bank from reference actives if provided
    if ctrl_config.support_set_file and ctrl_config.support_set_file.exists():
        actives = load_reference_actives(ctrl_config.support_set_file)
        state.seed_bank = initialize_seed_bank_from_actives(actives, episode=0)
        console.print(f"Initialized seed bank with {len(state.seed_bank)} reference actives")

    # Save state
    state_path = campaign_dir / "controller_state.json"
    state.save(state_path)
    console.print(f"Saved initial state to {state_path}")

    console.print(f"[bold green]✓[/bold green] Campaign '{campaign}' initialized!")


@app.command()
def list_arms():
    """List all available arms from the catalog."""
    catalog = get_arm_catalog()

    console.print(f"\n[bold]Available Arms ({len(catalog)} total):[/bold]\n")

    table = Table(show_header=True)
    table.add_column("Arm ID", style="cyan")
    table.add_column("Prior", style="yellow")
    table.add_column("Type", style="magenta")
    table.add_column("Regime", style="green")
    table.add_column("Needs Seeds", style="blue")
    table.add_column("Tags", style="white")

    for arm in catalog:
        needs_seeds = "Yes" if arm.requires_seeds else "No"
        tags = ", ".join(arm.tags)
        table.add_row(
            arm.arm_id, arm.prior_filename, arm.generator_type, arm.regime, needs_seeds, tags
        )

    console.print(table)


@app.command()
def check_priors(
    priors_dir: Path = typer.Option("./priors", help="Path to priors directory"),
):
    """Check which priors are available and which arms are enabled."""
    console.print(f"\n[bold]Checking priors in: {priors_dir}[/bold]\n")

    if not priors_dir.exists():
        console.print(f"[red]✗[/red] Priors directory not found: {priors_dir}")
        console.print("Please create the directory and add prior files.")
        return

    catalog = get_arm_catalog()
    available, unavailable = filter_available_arms(catalog, priors_dir)

    console.print(f"[bold green]✓ {len(available)} arms enabled[/bold green]")
    for arm in available:
        console.print(f"  [green]✓[/green] {arm.arm_id} ({arm.prior_filename})")

    if unavailable:
        console.print(f"\n[bold yellow]⚠ {len(unavailable)} arms disabled[/bold yellow]")
        for arm_id, reason in unavailable.items():
            console.print(f"  [yellow]✗[/yellow] {arm_id}: {reason}")


@app.command()
def run(
    campaign: str = typer.Option(..., help="Campaign name"),
    episodes: int = typer.Option(10, help="Number of episodes to run"),
    lite: bool = typer.Option(False, help="Lite mode: smaller batches for faster demo"),
    dry_run: bool = typer.Option(False, help="Dry-run mode: simulate REINVENT4 without running it"),
    reinvent_bin: Optional[Path] = typer.Option(None, help="Path to REINVENT4 binary"),
    config: Optional[Path] = typer.Option(None, help="Path to controller.yaml config file"),
):
    """Run episodes of the meta-controller."""
    console.print(f"\n[bold blue]Running campaign: {campaign}[/bold blue]")
    console.print(f"Episodes: {episodes} | Lite: {lite} | Dry-run: {dry_run}\n")

    # Load config
    campaign_dir = Path("runs") / campaign
    config_path = config or (campaign_dir / "controller.yaml")

    if not config_path.exists():
        console.print(f"[red]✗[/red] Config not found: {config_path}")
        console.print("Please run 'reinvent-meta init' first.")
        sys.exit(1)

    ctrl_config = ControllerConfig.load(config_path)

    # Override settings
    if lite:
        ctrl_config.episode.steps = 100
        ctrl_config.episode.batch_size = 50
        console.print("[yellow]Lite mode enabled: reduced steps and batch size[/yellow]")

    if reinvent_bin:
        ctrl_config.reinvent_bin = reinvent_bin

    # Load state
    state_path = campaign_dir / "controller_state.json"
    if state_path.exists():
        state = ControllerState.load(state_path)
        console.print(f"Loaded state from {state_path} (current episode: {state.current_episode})")
    else:
        console.print("[red]✗[/red] State not found. Please run 'reinvent-meta init' first.")
        sys.exit(1)

    # Get available arms
    catalog = get_arm_catalog()
    available_arms, unavailable = filter_available_arms(catalog, ctrl_config.priors_dir)

    if not available_arms:
        console.print("[red]✗[/red] No arms available! Please add prior files.")
        sys.exit(1)

    console.print(f"[green]✓[/green] {len(available_arms)} arms available\n")

    # Initialize components
    controller = MetaController(ctrl_config, state, available_arms)
    runner = EpisodeRunner(ctrl_config, dry_run=dry_run, lite=lite)
    metrics_calc = MetricsCalculator(
        ctrl_config.properties, ctrl_config.thresholds.hit_threshold
    )

    # Load support set
    support_set = []
    if ctrl_config.support_set_file and ctrl_config.support_set_file.exists():
        support_set = load_reference_actives(ctrl_config.support_set_file)
        console.print(f"Loaded {len(support_set)} reference actives for support set\n")

    # Main loop
    for i in range(episodes):
        console.print(f"[bold]Episode {state.current_episode + 1}[/bold]")

        # Select arm
        arm, reasons = controller.select_next_arm()
        console.print(f"Selected arm: [cyan]{arm.arm_id}[/cyan] ({arm.regime})")
        console.print(f"Reasons: {', '.join(reasons)}")

        # Run episode
        episode_dir, success, error = runner.run_episode(
            arm, state.current_episode + 1, state.seed_bank
        )

        if not success:
            console.print(f"[red]✗ Episode failed: {error}[/red]")
            # Record failure
            metrics = EpisodeMetrics()
            record = EpisodeRecord(
                episode_num=state.current_episode + 1,
                arm_id=arm.arm_id,
                prior_filename=arm.prior_filename,
                regime=arm.regime,
                reason=reasons,
                metrics=metrics,
                output_dir=episode_dir,
                success=False,
                error_message=error,
            )
            state.record_episode(record)
            controller.record_episode_result(arm, metrics, False)
            state.save(state_path)
            continue

        # Parse output
        generated_smiles, scores = runner.parse_episode_output(episode_dir)

        if not generated_smiles:
            console.print("[yellow]⚠ No molecules generated[/yellow]")
            metrics = EpisodeMetrics()
            success = False
        else:
            # Compute metrics
            metrics = metrics_calc.compute_episode_metrics(
                generated_smiles,
                scores,
                support_set=support_set + [s.smiles for s in state.seed_bank],
                prior_generated=state.all_generated_smiles,
                campaign_best_score=state.best_score,
            )

            console.print(f"[green]✓[/green] Generated {len(generated_smiles)} molecules")
            console.print(f"  Best score: {metrics.best_score:.3f}")
            console.print(f"  Diversity: {metrics.internal_diversity:.3f}")
            console.print(f"  OOD rate: {metrics.ood_rate:.3f}")

            # Update state
            state.all_generated_smiles.extend(generated_smiles)
            if metrics.best_score > state.best_score:
                state.best_score = metrics.best_score
                console.print(f"  [bold green]New best score![/bold green]")

            # Update seed bank
            top_molecules = sorted(zip(generated_smiles, scores), key=lambda x: x[1], reverse=True)[:20]
            state.seed_bank = update_seed_bank(
                state.seed_bank, top_molecules, state.current_episode + 1, ctrl_config.seed_bank_size
            )

        # Record episode
        record = EpisodeRecord(
            episode_num=state.current_episode + 1,
            arm_id=arm.arm_id,
            prior_filename=arm.prior_filename,
            regime=arm.regime,
            reason=reasons,
            metrics=metrics,
            output_dir=episode_dir,
            success=success,
        )
        state.record_episode(record)
        controller.record_episode_result(arm, metrics, success)

        # Save state
        state.save(state_path)
        console.print("")

    console.print(f"[bold green]✓ Campaign complete![/bold green]")
    console.print(f"Best score achieved: {state.best_score:.3f}")
    console.print(f"Total molecules generated: {len(state.all_generated_smiles)}")


@app.command()
def run_episode(
    campaign: str = typer.Option(..., help="Campaign name"),
    lite: bool = typer.Option(False, help="Lite mode: smaller batches for faster demo"),
    dry_run: bool = typer.Option(False, help="Dry-run mode: simulate REINVENT4 without running it"),
    reinvent_bin: Optional[Path] = typer.Option(None, help="Path to REINVENT4 binary"),
):
    """Run a single episode (useful for testing or manual control)."""
    console.print(f"\n[bold blue]Running single episode for: {campaign}[/bold blue]\n")

    # Load config
    campaign_dir = Path("runs") / campaign
    config_path = campaign_dir / "controller.yaml"

    if not config_path.exists():
        console.print(f"[red]✗[/red] Config not found: {config_path}")
        console.print("Please run 'reinvent-meta init' first.")
        sys.exit(1)

    ctrl_config = ControllerConfig.load(config_path)

    # Override settings
    if lite:
        ctrl_config.episode.steps = 100
        ctrl_config.episode.batch_size = 50
        console.print("[yellow]Lite mode enabled: reduced steps and batch size[/yellow]")

    if reinvent_bin:
        ctrl_config.reinvent_bin = reinvent_bin

    # Load state
    state_path = campaign_dir / "controller_state.json"
    if state_path.exists():
        state = ControllerState.load(state_path)
        console.print(f"Loaded state from {state_path} (current episode: {state.current_episode})")
    else:
        console.print("[red]✗[/red] State not found. Please run 'reinvent-meta init' first.")
        sys.exit(1)

    # Get available arms
    catalog = get_arm_catalog()
    available_arms, unavailable = filter_available_arms(catalog, ctrl_config.priors_dir)

    if not available_arms:
        console.print("[red]✗[/red] No arms available! Please add prior files.")
        sys.exit(1)

    console.print(f"[green]✓[/green] {len(available_arms)} arms available\n")

    # Initialize components
    controller = MetaController(ctrl_config, state, available_arms)
    runner = EpisodeRunner(ctrl_config, dry_run=dry_run, lite=lite)
    metrics_calc = MetricsCalculator(
        ctrl_config.properties, ctrl_config.thresholds.hit_threshold
    )

    # Load support set
    support_set = []
    if ctrl_config.support_set_file and ctrl_config.support_set_file.exists():
        support_set = load_reference_actives(ctrl_config.support_set_file)
        console.print(f"Loaded {len(support_set)} reference actives for support set\n")

    console.print(f"[bold]Episode {state.current_episode + 1}[/bold]")

    # Select arm
    arm, reasons = controller.select_next_arm()
    console.print(f"Selected arm: [cyan]{arm.arm_id}[/cyan] ({arm.regime})")
    console.print(f"Reasons: {', '.join(reasons)}")

    # Run episode
    episode_dir, success, error = runner.run_episode(
        arm, state.current_episode + 1, state.seed_bank
    )

    if not success:
        console.print(f"[red]✗ Episode failed: {error}[/red]")
        metrics = EpisodeMetrics()
        record = EpisodeRecord(
            episode_num=state.current_episode + 1,
            arm_id=arm.arm_id,
            prior_filename=arm.prior_filename,
            regime=arm.regime,
            reason=reasons,
            metrics=metrics,
            output_dir=episode_dir,
            success=False,
            error_message=error,
        )
        state.record_episode(record)
        controller.record_episode_result(arm, metrics, False)
        state.save(state_path)
        sys.exit(1)

    # Parse output
    generated_smiles, scores = runner.parse_episode_output(episode_dir)

    if not generated_smiles:
        console.print("[yellow]⚠ No molecules generated[/yellow]")
        metrics = EpisodeMetrics()
        success = False
    else:
        # Compute metrics
        metrics = metrics_calc.compute_episode_metrics(
            generated_smiles,
            scores,
            support_set=support_set + [s.smiles for s in state.seed_bank],
            prior_generated=state.all_generated_smiles,
            campaign_best_score=state.best_score,
        )

        console.print(f"[green]✓[/green] Generated {len(generated_smiles)} molecules")
        console.print(f"  Best score: {metrics.best_score:.3f}")
        console.print(f"  Diversity: {metrics.internal_diversity:.3f}")
        console.print(f"  OOD rate: {metrics.ood_rate:.3f}")

        # Update state
        state.all_generated_smiles.extend(generated_smiles)
        if metrics.best_score > state.best_score:
            state.best_score = metrics.best_score
            console.print(f"  [bold green]New best score![/bold green]")

        # Update seed bank
        top_molecules = sorted(zip(generated_smiles, scores), key=lambda x: x[1], reverse=True)[:20]
        state.seed_bank = update_seed_bank(
            state.seed_bank, top_molecules, state.current_episode + 1, ctrl_config.seed_bank_size
        )

    # Record episode
    record = EpisodeRecord(
        episode_num=state.current_episode + 1,
        arm_id=arm.arm_id,
        prior_filename=arm.prior_filename,
        regime=arm.regime,
        reason=reasons,
        metrics=metrics,
        output_dir=episode_dir,
        success=success,
    )
    state.record_episode(record)
    controller.record_episode_result(arm, metrics, success)

    # Save state
    state.save(state_path)

    console.print(f"\n[bold green]✓ Episode complete![/bold green]")
    console.print(f"Episode output: {episode_dir}")


@app.command()
def report(
    campaign: str = typer.Option(..., help="Campaign name"),
):
    """Generate report for a campaign."""
    console.print(f"\n[bold blue]Generating report for: {campaign}[/bold blue]\n")

    # Load state
    campaign_dir = Path("runs") / campaign
    state_path = campaign_dir / "controller_state.json"

    if not state_path.exists():
        console.print(f"[red]✗[/red] State not found: {state_path}")
        sys.exit(1)

    state = ControllerState.load(state_path)
    console.print(f"Loaded state: {state.current_episode} episodes")

    # Generate report
    report_dir = campaign_dir / "report"
    generator = ReportGenerator(state, report_dir)
    generator.generate_full_report()

    console.print(f"\n[bold green]✓ Report generated![/bold green]")
    console.print(f"Location: {report_dir}")
    console.print(f"  - manifest.json")
    console.print(f"  - report.md")
    console.print(f"  - report.html")


@app.command()
def clean(
    campaign: str = typer.Option(..., help="Campaign name"),
    force: bool = typer.Option(False, help="Force deletion without confirmation"),
):
    """Clean campaign temporary files (keeps state and report)."""
    campaign_dir = Path("runs") / campaign

    if not campaign_dir.exists():
        console.print(f"[red]✗[/red] Campaign not found: {campaign}")
        sys.exit(1)

    if not force:
        confirm = typer.confirm(f"Clean temporary files for campaign '{campaign}'?")
        if not confirm:
            console.print("Cancelled")
            return

    # Remove episode directories but keep state and report
    import shutil

    for arm_dir in campaign_dir.iterdir():
        if arm_dir.is_dir() and arm_dir.name not in ["report"]:
            shutil.rmtree(arm_dir)
            console.print(f"Removed: {arm_dir}")

    console.print("[green]✓[/green] Cleaned temporary files")


@app.command()
def version():
    """Show version information."""
    console.print(f"REINVENT4 Meta-Controller v{__version__}")


def main():
    """Main entry point."""
    app()


if __name__ == "__main__":
    main()
