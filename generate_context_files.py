#!/usr/bin/env python3
"""
Generate context files from source text using tiktoken.
Each file starts from the beginning of the source text and includes
up to the target number of tokens. This ensures consistent context
progression across different sizes.
"""

import argparse
import sys
from pathlib import Path

import tiktoken


def count_tokens(text, encoding_name="cl100k_base"):
    """Count the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    return len(encoding.encode(text))


def generate_context_file(source_text, target_tokens, output_file, encoding_name="cl100k_base"):
    """Generate a context file starting from the beginning of source text, up to target tokens."""
    encoding = tiktoken.get_encoding(encoding_name)

    # Tokenize the source text
    source_tokens = encoding.encode(source_text)
    source_token_count = len(source_tokens)

    print(f"Source text has {source_token_count} tokens")
    print(f"Target: {target_tokens} tokens for {output_file}")

    # Always start from the beginning and take up to target_tokens
    # If source is shorter than target, use all of it
    tokens_to_use = min(target_tokens, source_token_count)
    target_token_ids = source_tokens[:tokens_to_use]
    result_text = encoding.decode(target_token_ids)

    # Verify token count
    final_count = len(encoding.encode(result_text))

    # Write to file
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(result_text)

    print(f"  Created {output_file} with {final_count} tokens")

    if source_token_count < target_tokens:
        print(f"  Note: Source text only has {source_token_count} tokens, file is shorter than target {target_tokens}")

    return final_count


def main():
    parser = argparse.ArgumentParser(
        description="Generate context files from source text (each starting from beginning)"
    )
    parser.add_argument("source", help="Source text file to use as base content")
    parser.add_argument(
        "--sizes",
        type=str,
        default="2,4,8,16,32,64,128",
        help="Comma-separated list of sizes in thousands of tokens (default: 2,4,8,16,32,64,128)",
    )
    parser.add_argument(
        "--encoding",
        type=str,
        default="cl100k_base",
        help="Tiktoken encoding to use (default: cl100k_base for GPT-3.5/GPT-4)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=".",
        help="Directory to save context files (default: current directory)",
    )
    parser.add_argument(
        "--prompt-suffix",
        type=str,
        default="\n\nPlease provide a summary of the above text.",
        help="Prompt to append at the end of each context file",
    )

    args = parser.parse_args()

    # Read source file
    source_path = Path(args.source)
    if not source_path.exists():
        print(f"Error: Source file '{args.source}' not found")
        sys.exit(1)

    try:
        with open(source_path, encoding="utf-8") as f:
            source_text = f.read()
    except Exception as e:
        print(f"Error reading source file: {e}")
        sys.exit(1)

    # Create output directory if needed
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # Parse sizes
    try:
        sizes = [int(s.strip()) for s in args.sizes.split(",")]
    except ValueError:
        print("Error: Sizes must be comma-separated integers")
        sys.exit(1)

    # Check if tiktoken encoding is valid
    try:
        encoding = tiktoken.get_encoding(args.encoding)
    except Exception as e:
        print(f"Error: Invalid encoding '{args.encoding}': {e}")
        print("Common encodings: cl100k_base (GPT-3.5/4), p50k_base, r50k_base")
        sys.exit(1)

    # Reserve tokens for the prompt suffix
    prompt_suffix_tokens = len(encoding.encode(args.prompt_suffix))
    print(f"\nUsing encoding: {args.encoding}")
    print(f"Prompt suffix uses {prompt_suffix_tokens} tokens")
    print(f"Generating context files in: {output_dir}/")
    print("-" * 50)

    # Generate context files
    results = []
    for size_k in sizes:
        target_tokens = size_k * 1000

        # Reserve space for prompt suffix
        context_tokens = target_tokens - prompt_suffix_tokens

        if context_tokens <= 0:
            print(f"Skipping {size_k}k: Not enough tokens after prompt suffix")
            continue

        output_file = output_dir / f"{size_k}k.txt"

        # Generate the context part
        temp_file = output_dir / f"temp_{size_k}k.txt"
        actual_context_tokens = generate_context_file(source_text, context_tokens, temp_file, args.encoding)

        # Read the context and append prompt
        with open(temp_file, encoding="utf-8") as f:
            context = f.read()

        final_text = context + args.prompt_suffix

        # Write final file
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(final_text)

        # Verify final token count
        final_tokens = len(encoding.encode(final_text))

        # Clean up temp file
        temp_file.unlink()

        results.append(
            {
                "file": output_file.name,
                "target": target_tokens,
                "actual": final_tokens,
                "difference": final_tokens - target_tokens,
            }
        )

        print(
            f"  Final: {output_file.name} has {final_tokens} tokens (target: {target_tokens}, diff: {final_tokens - target_tokens:+d})"
        )

    # Summary
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print(f"{'File':<15} {'Target':<10} {'Actual':<10} {'Difference':<10}")
    print("-" * 50)
    for r in results:
        print(f"{r['file']:<15} {r['target']:<10} {r['actual']:<10} {r['difference']:+d}")

    print(f"\nAll context files generated successfully!")
    print(f"You can now run benchmarks with these files.")


if __name__ == "__main__":
    main()
