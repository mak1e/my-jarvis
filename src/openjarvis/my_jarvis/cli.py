"""CLI entrypoint (`my-jarvis`) built on top of Stanford OpenJarvis."""

from __future__ import annotations

import click
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from openjarvis.my_jarvis.hardware import (
    inspect_hardware,
    recommend_models_for_hardware,
)
from openjarvis.my_jarvis.runtime import MyJarvisRuntime

console = Console()


@click.group(help="My-JARVIS — Personal multi-brain AI built on Stanford's OpenJarvis.")
def main() -> None:
    pass


@main.command()
@click.option("--host", default="127.0.0.1", show_default=True, help="Bind host.")
@click.option(
    "--port",
    default=8765,
    show_default=True,
    type=int,
    help="Port to serve HUD + OpenJarvis API.",
)
def serve(host: str, port: int) -> None:
    """Launch the My-JARVIS Command Center HUD and OpenJarvis API server."""
    import uvicorn

    from openjarvis.my_jarvis.server import create_my_jarvis_app

    rt = MyJarvisRuntime()
    app = create_my_jarvis_app(rt)
    active = rt.router.active_brain
    console.print(
        Panel.fit(
            f"[bold cyan]J.A.R.V.I.S. Command Center[/bold cyan] (built on Stanford OpenJarvis)\n"
            f"HUD & API:   [bold]http://{host}:{port}[/bold]\n"
            f"OpenAI API:  [bold]http://{host}:{port}/v1/chat/completions[/bold]\n"
            f"Active Brain:[bold yellow] {active.name}[/bold yellow] ({active.model} via {active.provider})",
            border_style="cyan",
        )
    )
    uvicorn.run(app, host=host, port=port, log_level="info")


@main.command()
@click.option(
    "--brain",
    default=None,
    help="Optional initial brain ID or alias (e.g. venice-deepseek, local-qwen).",
)
def chat(brain: str | None) -> None:
    """Interactive terminal chat with live /brain hot-swapping and OpenJarvis tools."""
    rt = MyJarvisRuntime()
    if brain:
        res = rt.router.switch_brain(brain)
        if not res.get("ok"):
            console.print(f"[red]{res.get('error')}[/red]")

    active = rt.router.active_brain
    console.print(
        Panel.fit(
            f"[bold cyan]J.A.R.V.I.S. Terminal[/bold cyan] · Active Brain: [bold]{active.name}[/bold] (`{active.model}`)\n"
            "Commands: `/brain <name>`, `/brains`, `/memory`, `/hardware`, `/doctor`, `/clear`, `/exit`",
            border_style="cyan",
        )
    )
    while True:
        try:
            user_input = console.input(
                f"\n[bold cyan]you ({rt.router.active_brain.id})>[/bold cyan] "
            ).strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Standing down, sir.[/dim]")
            break
        if not user_input:
            continue
        if user_input.lower() in ("/exit", "/quit", "exit", "quit"):
            console.print("[dim]Standing down, sir.[/dim]")
            break
        if user_input.lower() == "/clear":
            rt.clear_history()
            console.print("[green]Conversation history cleared.[/green]")
            continue
        if user_input.lower() == "/doctor":
            _print_doctor(rt)
            continue
        if user_input.lower() == "/hardware":
            _print_hardware(rt.router.ollama_host)
            continue

        result = rt.chat(user_input)
        rep = result["reply"]
        console.print(
            Panel(
                Markdown(rep["content"]),
                title=f"[cyan]JARVIS[/cyan] · [yellow]{rep['brain_id']}[/yellow] ({rep['latency_ms']} ms)",
                border_style="bright_black",
            )
        )


@main.command()
def brains() -> None:
    """List all configured JARVIS brains and their live readiness."""
    rt = MyJarvisRuntime()
    table = Table(title="My-JARVIS Brains (Hot-Swappable)")
    table.add_column("ID", style="cyan")
    table.add_column("Name", style="bold")
    table.add_column("Provider")
    table.add_column("Model")
    table.add_column("Status")
    for b in rt.router.list_brains(probe_live=True):
        prefix = "● " if b["active"] else "  "
        status_style = "green" if b["ready"] else "yellow"
        table.add_row(
            f"{prefix}{b['id']}",
            b["name"],
            b["provider"],
            b["model"],
            f"[{status_style}]{b['status_note']}[/{status_style}]",
        )
    console.print(table)


@main.command()
def hardware() -> None:
    """Inspect local CPU/RAM/VRAM and recommend fitting Ollama / Hugging Face GGUF models."""
    rt = MyJarvisRuntime()
    _print_hardware(rt.router.ollama_host)


@main.command()
def doctor() -> None:
    """Run setup diagnostics and fix suggestions."""
    rt = MyJarvisRuntime()
    _print_doctor(rt)


def _print_doctor(rt: MyJarvisRuntime) -> None:
    report = rt.run_doctor()
    table = Table(title="My-JARVIS & OpenJarvis Doctor")
    table.add_column("Check", style="bold")
    table.add_column("Status")
    table.add_column("Details")
    table.add_column("Fix / Tip", style="dim")
    for c in report["checks"]:
        color = (
            "green"
            if c["status"] == "ok"
            else ("yellow" if c["status"] == "warn" else "red")
        )
        table.add_row(
            c["name"],
            f"[{color}]{c['status'].upper()}[/{color}]",
            c["detail"],
            c["fix"],
        )
    console.print(table)


def _print_hardware(ollama_host: str) -> None:
    hw = inspect_hardware(ollama_host)
    recs = recommend_models_for_hardware(hw)
    console.print(
        Panel.fit(
            f"[bold]{hw.os_name}[/bold] ({hw.arch}) · CPU: {hw.cpu_cores_physical}P/{hw.cpu_cores_logical}L\n"
            f"RAM: [cyan]{hw.ram_total_gb:.1f} GB[/cyan] ({hw.ram_available_gb:.1f} GB free) · "
            f"VRAM: [cyan]{hw.total_vram_gb:.1f} GB[/cyan] ({hw.accelerator})\n"
            f"Ollama: {'[green]Running[/green]' if hw.ollama_running else '[yellow]Offline[/yellow]'}",
            title="Hardware Telemetry",
        )
    )
    table = Table(title="Recommended Local Open-Weight Models")
    table.add_column("Model", style="bold")
    table.add_column("Fit")
    table.add_column("Ollama Pull Command", style="cyan")
    table.add_column("Reason")
    for r in recs:
        color = (
            "green" if r.fit == "optimal" else ("yellow" if r.fit == "tight" else "red")
        )
        table.add_row(
            r.name, f"[{color}]{r.fit.upper()}[/{color}]", r.ollama_cmd, r.fit_reason
        )
    console.print(table)


if __name__ == "__main__":
    main()
