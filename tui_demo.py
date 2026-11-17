"""
tui_demo.py
===========
Terminal User Interface (TUI) interactive live cockpit for rag_guard using Rich.
Renders real-time live token streaming, dynamic 5D metric gauges, and early GPU abort alerts.

Usage:
    python3 tui_demo.py
"""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path
from typing import List, Tuple

# Add src to pythonpath
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from rich.align import Align
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table
from rich.text import Text

from rag_guard.vllm_hook import AsyncRAGGuard, MockAsyncLLMEngine

console = Console()

PRESET_CHOICES = {
    "1": (
        "Apple Q1 2024 Earnings (Grounded & Faithful)",
        "Apple Inc. reported quarterly revenue of $119.58 billion in Q1 2024, up 2 percent year-over-year. iPhone revenue was $69.7 billion.",
        "What was Apple's total revenue in Q1 2024?",
        False,
    ),
    "2": (
        "Tesla Q3 Deliveries (Evident Numerical Conflict)",
        "In Q3 2023, Tesla delivered 435,059 vehicles worldwide and produced 430,488 vehicles, generating $23.35 billion in revenue.",
        "How many vehicles did Tesla deliver in Q3 2023?",
        True,
    ),
    "3": (
        "OpenAI GPT-4 Contract (Baseless Extrapolation)",
        "OpenAI announced GPT-4 on March 14, 2023. It passed a simulated bar exam with a score around the top 10% of test takers.",
        "What major contracts did OpenAI sign with GPT-4?",
        True,
    ),
}


def make_layout() -> Layout:
    layout = Layout(name="root")
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="body", ratio=1),
        Layout(name="footer", size=3),
    )
    layout["body"].split_row(
        Layout(name="left", ratio=5),
        Layout(name="right", ratio=5),
    )
    return layout


async def run_interactive_tui(preset_id: str = "1", auto_abort: bool = True) -> None:
    title, context, prompt, is_hallucinated = PRESET_CHOICES.get(
        preset_id, PRESET_CHOICES["1"]
    )

    layout = make_layout()
    header_text = Text()
    header_text.append("rag_guard ", style="bold green")
    header_text.append("Real-Time Hallucination Detection & GPU Interception Cockpit\n", style="bold white")
    header_text.append(f"Scenario: {title}  |  Constraint: Δt ≤ 20 ms  |  Cost: $0.00", style="dim")
    layout["header"].update(Panel(Align.center(header_text), border_style="green"))

    left_content = Text()
    left_content.append("Retrieved Context:\n", style="bold cyan")
    left_content.append(f"{context}\n\n", style="dim white")
    left_content.append("Query Prompt:\n", style="bold cyan")
    left_content.append(f"{prompt}\n", style="white")
    layout["left"].update(Panel(left_content, title="RAG Input", border_style="cyan"))

    right_content = Text("Waiting for decoding stream to start...", style="italic dim")
    layout["right"].update(Panel(right_content, title="Live Token Stream (vLLM Engine)", border_style="yellow"))

    footer_text = Text("Status: Initializing pre-cached context entities...", style="dim")
    layout["footer"].update(Panel(Align.center(footer_text), border_style="blue"))

    guard = AsyncRAGGuard(
        engine=MockAsyncLLMEngine(token_delay_ms=35.0),
        early_stop_threshold=0.35,
        abort_on_hallucination=auto_abort,
    )

    stream_tokens: List[Tuple[str, bool]] = []
    start_time = time.perf_counter()

    with Live(layout, refresh_per_second=20, console=console) as live:
        async for token, step_res, final_res in guard.stream_generate(
            prompt=prompt,
            context=context,
            simulate_hallucination=is_hallucinated,
        ):
            if token:
                stream_tokens.append((token, step_res.is_hallucination))
                
                # Render tokens
                token_text = Text()
                for tok, is_h in stream_tokens:
                    if is_h:
                        token_text.append(tok, style="bold red")
                    else:
                        token_text.append(tok, style="bold green")

                # Telemetry table
                table = Table(title="5D Intrinsic & Grounding Telemetry", expand=True)
                table.add_column("Signal", style="dim")
                table.add_column("Value", style="bold")
                table.add_column("Health", style="bold")

                s_pct = int(step_res.faithfulness_score * 100)
                faith_style = "green" if s_pct >= 50 else "red"
                table.add_row("Faithfulness (S_faith)", f"{s_pct}%", f"[{faith_style}]{'■' * (s_pct // 10)}[/{faith_style}]")
                table.add_row("Mean Logprob (l_bar)", f"{step_res.l_bar:.2f}", "Normal" if step_res.l_bar > -1.0 else "[red]Hesitant[/red]")
                table.add_row("Min Hesitation (l_min)", f"{step_res.l_min:.2f}", "Stable" if step_res.l_min > -2.0 else "[red]Collapse[/red]")
                table.add_row("Shannon Entropy (H_bar)", f"{step_res.H_bar:.2f} b", "Low Ambiguity" if step_res.H_bar < 1.0 else "[red]Spike[/red]")
                table.add_row("Grounding Ratio (R_ent)", f"{step_res.R_entity:.2f}", "Grounded" if step_res.R_entity > 0.8 else "[red]Conflict[/red]")

                # Update live view
                stream_display = Layout()
                stream_display.split_column(
                    Layout(Panel(token_text, title="Decoding Stream", border_style="yellow"), ratio=6),
                    Layout(table, ratio=4),
                )
                layout["right"].update(stream_display)

                elapsed_ms = (time.perf_counter() - start_time) * 1000.0
                if step_res.early_stop_recommended and auto_abort:
                    foot = Text(f"[ABORTED MID-STREAM] Hallucination detected at Token #{step_res.num_tokens}! GPU aborted early. Δt = {elapsed_ms:.1f} ms", style="bold red")
                    layout["footer"].update(Panel(Align.center(foot), border_style="red"))
                else:
                    foot = Text(f"Streaming Active... Token #{step_res.num_tokens} | Current Faithfulness: {s_pct}% | Elapsed: {elapsed_ms:.1f} ms", style="green")
                    layout["footer"].update(Panel(Align.center(foot), border_style="green"))

            if final_res:
                final = final_res.faithfulness_result
                if final.is_hallucination:
                    foot = Text(f"Stream Completed: [FLAGGED HALLUCINATED] | Final Faithfulness: {final.faithfulness_score*100:.1f}% | Total Time: {final_res.elapsed_ms:.2f} ms", style="bold red")
                    layout["footer"].update(Panel(Align.center(foot), border_style="red"))
                else:
                    foot = Text(f"Stream Completed: [VERIFIED FAITHFUL] | Final Faithfulness: {final.faithfulness_score*100:.1f}% | Total Time: {final_res.elapsed_ms:.2f} ms", style="bold green")
                    layout["footer"].update(Panel(Align.center(foot), border_style="green"))


def main() -> None:
    console.print("\n[bold green]rag_guard Terminal Live Cockpit[/bold green]")
    console.print("Choose an evaluation scenario to run in real time:\n")
    console.print("  [bold cyan]1[/bold cyan]: Apple Q1 2024 Earnings ([green]Grounded & Faithful[/green])")
    console.print("  [bold cyan]2[/bold cyan]: Tesla Vehicle Deliveries ([red]Evident Conflict Hallucination[/red])")
    console.print("  [bold cyan]3[/bold cyan]: OpenAI Enterprise Model ([red]Baseless Extrapolation Hallucination[/red])\n")

    choice = Prompt.ask("Select Scenario [1/2/3]", default="1")
    auto_abort_choice = Prompt.ask("Enable Real-Time GPU Early Interception? (y/n)", default="y")
    auto_abort = auto_abort_choice.lower().startswith("y")

    asyncio.run(run_interactive_tui(preset_id=choice, auto_abort=auto_abort))
    console.print("\n[bold green]Live run complete.[/bold green]\n")


if __name__ == "__main__":
    main()
