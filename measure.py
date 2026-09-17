#!/usr/bin/env python3
"""Run a small, repeatable request timing exercise against the local API."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure repeated requests to vLLM.")
    parser.add_argument("--label", default="run", help="Name printed with the results.")
    parser.add_argument("--model", default="qwen3-4b", help="Served model name.")
    parser.add_argument("--port", type=int, default=8000, help="Local API port.")
    parser.add_argument("--runs", type=int, default=3, help="Number of requests.")
    parser.add_argument(
        "--prompt",
        default="Explain why smaller model weights can improve inference efficiency.",
        help="Prompt to send for every request.",
    )
    return parser.parse_args()


def request_once(url: str, model: str, prompt: str) -> tuple[float, int | None]:
    payload = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
        }
    ).encode()
    request = Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    try:
        with urlopen(request, timeout=300) as response:
            result = json.load(response)
    except HTTPError as error:
        body = error.read().decode(errors="replace")
        raise RuntimeError(f"HTTP {error.code}: {body}") from error
    except URLError as error:
        raise RuntimeError(f"Could not reach {url}: {error.reason}") from error
    elapsed = time.perf_counter() - started
    completion_tokens = result.get("usage", {}).get("completion_tokens")
    return elapsed, completion_tokens


def main() -> int:
    args = parse_args()
    if args.runs < 1:
        print("Error: --runs must be at least 1.", file=sys.stderr)
        return 2

    url = f"http://127.0.0.1:{args.port}/v1/chat/completions"
    timings: list[float] = []
    token_counts: list[int] = []

    print(f"{args.label}: sending {args.runs} request(s) to {url}")
    for index in range(args.runs):
        try:
            elapsed, completion_tokens = request_once(url, args.model, args.prompt)
        except RuntimeError as error:
            print(f"Error on request {index + 1}: {error}", file=sys.stderr)
            return 1
        timings.append(elapsed)
        if completion_tokens is not None:
            token_counts.append(completion_tokens)
        token_text = f", {completion_tokens} output tokens" if completion_tokens is not None else ""
        print(f"  request {index + 1}: {elapsed:.2f}s{token_text}")

    print(f"Average: {statistics.mean(timings):.2f}s")
    print(f"Median:  {statistics.median(timings):.2f}s")
    if token_counts and statistics.mean(timings) > 0:
        print(f"Average output tokens: {statistics.mean(token_counts):.1f}")
        print(f"Approx. requests per minute: {60 / statistics.mean(timings):.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
