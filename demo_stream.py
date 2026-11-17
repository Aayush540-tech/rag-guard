"""
demo_stream.py
==============
Interactive streaming showcase demonstrating real-time RAG hallucination
detection and early interception using AsyncRAGGuard and vLLM integration.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

# Add src to path for direct invocation
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from rag_guard.vllm_hook import AsyncRAGGuard, MockAsyncLLMEngine


# Terminal ANSI color codes for rich visual output
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"


async def run_grounded_demo() -> None:
    print(f"\n{BOLD}{CYAN}{'='*70}{RESET}")
    print(f"{BOLD}{CYAN}  SCENARIO 1: Grounded & Faithful Generation (Continuous Verification){RESET}")
    print(f"{BOLD}{CYAN}{'='*70}{RESET}")

    context = (
        "Apple Inc. reported quarterly revenue of $119.58 billion in Q1 2024, "
        "up 2 percent year-over-year. iPhone revenue was $69.7 billion."
    )
    prompt = "Context:\n" + context + "\n\nQuestion: What was Apple's total revenue in Q1 2024?\nAnswer:"

    print(f"{DIM}Context: {context}{RESET}\n")
    print(f"{BOLD}Streaming Answer: {RESET}", end="", flush=True)

    guard = AsyncRAGGuard(
        engine=MockAsyncLLMEngine(token_delay_ms=25.0),
        early_stop_threshold=0.30,
        abort_on_hallucination=True,
    )

    final_result = None
    async for token, step_res, final in guard.stream_generate(
        prompt=prompt,
        context=context,
        simulate_hallucination=False,
    ):
        if token:
            color = GREEN if not step_res.is_hallucination else RED
            print(f"{color}{token}{RESET}", end="", flush=True)
        if final:
            final_result = final

    print("\n")
    if final_result:
        res = final_result.faithfulness_result
        print(f"  {BOLD}Status:{RESET}                 {GREEN}VERIFIED FAITHFUL{RESET}")
        print(f"  {BOLD}Faithfulness Score (S_faith):{RESET} {GREEN}{res.faithfulness_score:.4f}{RESET}")
        print(f"  {BOLD}Hallucination Score:{RESET}         {res.hallucination_score:.4f}")
        print(f"  {BOLD}Extracted Entities:{RESET}          {', '.join(final_result.answer_entities)}")
        print(f"  {BOLD}Grounding Ratio (R_entity):{RESET}  {res.features['R_entity']:.2f}")
        print(f"  {BOLD}Elapsed Stream Time:{RESET}        {final_result.elapsed_ms:.2f} ms")


async def run_hallucinated_demo() -> None:
    print(f"\n{BOLD}{YELLOW}{'='*70}{RESET}")
    print(f"{BOLD}{YELLOW}  SCENARIO 2: Confident Hallucination with Real-Time Early Interception{RESET}")
    print(f"{BOLD}{YELLOW}{'='*70}{RESET}")

    context = (
        "Apple Inc. reported quarterly revenue of $119.58 billion in Q1 2024, "
        "up 2 percent year-over-year. iPhone revenue was $69.7 billion."
    )
    prompt = "Context:\n" + context + "\n\nQuestion: What was Apple's total revenue in Q1 2024?\nAnswer:"

    print(f"{DIM}Context: {context}{RESET}\n")
    print(f"{BOLD}Streaming Answer: {RESET}", end="", flush=True)

    guard = AsyncRAGGuard(
        engine=MockAsyncLLMEngine(token_delay_ms=25.0),
        early_stop_threshold=0.40,
        abort_on_hallucination=True,
    )

    final_result = None
    intercepted_early = False

    async for token, step_res, final in guard.stream_generate(
        prompt=prompt,
        context=context,
        simulate_hallucination=True,
    ):
        if token:
            color = GREEN if not step_res.is_hallucination else RED
            print(f"{color}{token}{RESET}", end="", flush=True)
            if step_res.early_stop_recommended:
                intercepted_early = True
        if final:
            final_result = final

    print("\n")
    if intercepted_early:
        print(f"  {BOLD}Guardrail Action:{RESET}          {BOLD}{RED}[ABORTED EARLY]{RESET} Generation halted mid-stream!")
    if final_result:
        res = final_result.faithfulness_result
        print(f"  {BOLD}Status:{RESET}                 {RED}HALLUCINATION DETECTED{RESET}")
        print(f"  {BOLD}Faithfulness Score (S_faith):{RESET} {RED}{res.faithfulness_score:.4f}{RESET}")
        print(f"  {BOLD}Hallucination Score:{RESET}         {RED}{res.hallucination_score:.4f}{RESET}")
        print(f"  {BOLD}Minimum Logprob (hesitation):{RESET}{res.features['l_min']:.2f}")
        print(f"  {BOLD}Logprob Variance (instability):{RESET}{res.features['var_l']:.2f}")
        print(f"  {BOLD}Grounding Ratio (R_entity):{RESET}  {res.features['R_entity']:.2f}")
        print(f"  {BOLD}Elapsed Stream Time:{RESET}        {final_result.elapsed_ms:.2f} ms")


async def main() -> None:
    print(f"\n{BOLD}{'#'*70}{RESET}")
    print(f"{BOLD}  rag_guard -- Real-Time vLLM Streaming Verification & Guardrail Demo{RESET}")
    print(f"{BOLD}{'#'*70}{RESET}")
    await run_grounded_demo()
    await run_hallucinated_demo()
    print(f"\n{BOLD}{CYAN}{'='*70}{RESET}\n")


if __name__ == "__main__":
    asyncio.run(main())
