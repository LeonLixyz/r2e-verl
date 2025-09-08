#!/usr/bin/env python3
"""
Script to convert R2E-Gym dataset to VERL-compatible format.

This script loads the R2E-Gym dataset and converts it to a format that VERL can use,
including proper chat format with system/user messages and docker image tags.
"""

import pandas as pd
import json
import os
from datasets import load_dataset
from typing import List, Dict, Any, Optional
import argparse
import random
import matplotlib.pyplot as plt
import seaborn as sns


# Remove system message creation - RLHFDataset handles this automatically

SWEAGENT_USER_PROMPT = """I have uploaded a python code repository in the /testbed directory.
  
Now consider the following Github issue:

<github_issue>
{problem_statement}
</github_issue>

Can you help me implement the necessary changes to the repository to fix the <github_issue>?
I have already taken care of all changes to any of the test files described in the <github_issue>. This means you DON'T have to modify the testing logic or any of the tests in any way! Your task is to make changes to non-test files in the /testbed directory to ensure the <github_issue> is resolved.

Follow these steps to resolve the issue:
1. First, explore the codebase to locate and understand the code relevant to the <github_issue>. 
  - Use efficient search commands to identify key files and functions. 
  - You should err on the side of caution and look at various relevant files and build your understanding of 
    - how the code works
    - what are the expected behaviors and edge cases
    - what are the potential root causes for the given issue

2. Assess whether you can reproduce the issue:
    - Create a script at '/testbed/reproduce_issue.py' that demonstrates the error.
    - Execute this script to confirm the error behavior.
    - You should reproduce the issue before fixing it.
    - Your reproduction script should also assert the expected behavior for the fixed code. 

3. Analyze the root cause:
    - Identify the underlying problem based on your code exploration and reproduction results.
    - Critically analyze different potential approaches to fix the issue. 
    - You NEED to explicitly reason about multiple approaches to fix the issue. Next, find the most elegant and effective solution among them considering the tradeoffs (correctness, generality, side effects, etc.).
    - You would need to reason about execution paths, edge cases, and other potential issues. You should look at the unit tests to understand the expected behavior of the relevant code.

4. Implement your solution:
    - Make targeted changes to the necessary files following idiomatic code patterns once you determine the root cause.
    - You should be thorough and methodical.

5. Verify your solution:
    - Rerun your reproduction script to confirm the error is fixed.
    - If verification fails, iterate on your solution until successful. If you identify the reproduction script is buggy, adjust it as needed.

6. Run unit tests:
    - Find and run the relevant unit tests relevant to the performed fix.
    - You should run the unit tests to ensure your solution is correct and does not cause any regressions.
    - In cases where the unit tests are do not pass, you should consider whether the unit tests does not reflect the *new* expected behavior of the code. If so, you can test it by writing additional edge test cases.
    - Use the existing test runner to run the unit tests you identify as relevant to the changes you made. For example:
        - `python -m pytest -xvs sympy/physics/units/tests/test_dimensions_transcendental.py`
        - `python -m pytest tests/test_domain_py.py::test_pymethod_options`
        - `./tests/runtests.py constraints.tests.CheckConstraintTests -v 2`
    - RUN ALL relevant unit tests to ensure your solution is correct and does not cause any regressions.

7. Test edge cases:
    - Identify potential edge cases that might challenge your solution.
    - Create additional test cases in a separate file '/testbed/edge_case_tests.py'.
    - Execute these tests to verify your solution's robustness.
    - You should run multiple rounds of edge cases. When creating edge cases:
      - Consider complex scenarios beyond the original issue description
      - Test for regressions to ensure existing functionality remains intact

8. Refine if necessary:
    - If edge case testing reveals issues, refine your solution accordingly.
    - Ensure your final implementation handles all identified scenarios correctly.
    - Document any assumptions or limitations of your solution.

9. Submit your solution:
    - Once you have verified your solution, submit your solution using the `submit` tool.

A successful resolution means:
- The specific error/issue described no longer occurs
- Your changes maintain compatibility with existing functionality
- Edge cases are properly handled


Additional recommendations:
- You should be thorough, methodical, and prioritize quality over speed. Be comprehensive.
- You should think carefully before making the tool call about what should be done. However, each step should only use one tool call. YOU SHOULD NOT USE TOOLS INSIDE YOUR THOUGHT PROCESS. YOU SHOULD PRIMARILY USE THINKING FOR IDENTIFYING THE ROOT CAUSE OF THE ISSUE, MAKING THE CHANGES, AND CREATING TEST CASES (REPRODUCTION OR EDGE CASES).
- Each action you take is somewhat expensive. Wherever possible, combine multiple actions into a single action (e.g., combine multiple bash commands, use sed/grep for bulk operations). 
    - Your grep commands should identify both relevant files and line numbers so you can use the file_editor tool.
    - Use grep with `-A -B -C` flags to quickly identify the relevant code blocks during your exploration.
- When exploring the codebase, use targeted search patterns to minimize unnecessary operations.
- When creating edge cases, you should look at the relevant existing tests to understand existing "regression" test cases. Ensure the fix doesn't break existing functionality.
"""

def create_user_message(row: Dict[str, Any]) -> str:
    """Create a user message from the R2E-Gym dataset row."""
    problem_statement = row.get('problem_statement', '')
    return problem_statement


def load_successful_images(csv_file: str) -> List[str]:
    """
    Load successful docker images from CSV file.
    
    Args:
        csv_file: Path to CSV file with columns 'docker_image' and 'success'
        
    Returns:
        List of successful docker images
    """
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"CSV file not found: {csv_file}")
        
    df = pd.read_csv(csv_file)
    successful_images = df[df['success'] == True]['docker_image'].tolist()
    print(f"Found {len(successful_images)} successful docker images from {csv_file}")
    return successful_images


def convert_r2e_gym_to_verl_format(
    dataset_name: str = "R2E-Gym/R2E-Gym-V1",
    split: str = "train",
    successful_images: Optional[List[str]] = None,
    test_size: int = 16,
    output_dir: str = ".",
    seed: int = 42
) -> None:
    """
    Convert R2E-Gym dataset to VERL format, filtering for successful images and splitting train/test.
    
    Args:
        dataset_name: HuggingFace dataset name
        split: Dataset split to use
        successful_images: List of successful docker images to include
        test_size: Number of examples for test set
        output_dir: Directory to save output files
        seed: Random seed for reproducible splits
    """
    print(f"Loading {dataset_name} dataset...")
    dataset = load_dataset(dataset_name, split=split)
    
    print(f"Loaded {len(dataset)} examples")
    print(f"Columns: {list(dataset.column_names)}")
    
    # Filter for successful images
    if successful_images:
        print(f"\nFiltering to include only successful images...")
        docker_images = dataset['docker_image']
        filtered_indices = [i for i, img in enumerate(docker_images) if img in successful_images]
        dataset = dataset.select(filtered_indices)
        print(f"Filtered to {len(dataset)} examples with successful docker images")
    
    print(f"\nProcessing {len(dataset)} examples...")
    
    # Convert to VERL format
    verl_data = []
    skipped_count = 0
    
    for i, row in enumerate(dataset):
        if i % 100 == 0:
            print(f"Processed {i}/{len(dataset)} examples...")
            
        # Check if problem_statement is not null/empty
        problem_statement = row.get('problem_statement', '')
        if not problem_statement or problem_statement.strip() == '':
            skipped_count += 1
            continue
            
        user_message = create_user_message(row)
        
        # Create user messages only - RLHFDataset adds system prompt automatically
        user_messages = [
            {"role": "user", "content": SWEAGENT_USER_PROMPT.format(problem_statement=user_message)}
        ]
        
        # Only include docker_image and prompt in the output
        verl_row = {
            'prompt': user_messages,  # RLHFDataset expects this to be the messages list
            'docker_image': row['docker_image'],
        }
        
        verl_data.append(verl_row)
    
    print(f"Skipped {skipped_count} examples with null/empty problem_statement")
    
    # Convert to DataFrame
    df = pd.DataFrame(verl_data)
    
    # Split into train and test
    random.seed(seed)
    indices = list(range(len(df)))
    random.shuffle(indices)
    
    test_indices = indices[:test_size]
    train_indices = indices[test_size:]
    
    test_df = df.iloc[test_indices].reset_index(drop=True)
    train_df = df.iloc[train_indices].reset_index(drop=True)
    
    # Save files
    full_file = os.path.join(output_dir, "r2e_gym_full.parquet")
    test_file = os.path.join(output_dir, "r2e_gym_test.parquet")
    train_file = os.path.join(output_dir, "r2e_gym_train.parquet")
    
    df.to_parquet(full_file, index=False)
    test_df.to_parquet(test_file, index=False)
    train_df.to_parquet(train_file, index=False)
    
    print(f"\nSaved {len(df)} full examples to {full_file}")
    print(f"Saved {len(test_df)} test examples to {test_file}")
    print(f"Saved {len(train_df)} train examples to {train_file}")
    
    # Show statistics
    print(f"\nDataset Statistics:")
    print(f"- Total examples: {len(df)}")
    print(f"- Test examples: {len(test_df)}")
    print(f"- Train examples: {len(train_df)}")
    print(f"- Unique docker images: {df['docker_image'].nunique()}")
    
    # Show docker image distribution by base names
    print(f"\nDocker Image Distribution by Base Names:")
    
    # Extract base image names (everything before the ":")
    df['base_image'] = df['docker_image'].str.split(':').str[0]
    base_image_counts = df['base_image'].value_counts()
    
    print(f"Total unique base images: {len(base_image_counts)}")
    for base_img, count in base_image_counts.items():
        print(f"  {base_img}: {count} examples")
    
    # Create visualizations
    plt.style.use('default')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Pie chart
    colors = plt.cm.Set3(range(len(base_image_counts)))
    wedges, texts, autotexts = ax1.pie(base_image_counts.values, 
                                       labels=base_image_counts.index, 
                                       autopct='%1.1f%%',
                                       colors=colors,
                                       startangle=90)
    ax1.set_title('Docker Base Image Distribution (Pie Chart)', fontsize=14, fontweight='bold')
    
    # Rotate labels if there are many
    if len(base_image_counts) > 8:
        for text in texts:
            text.set_fontsize(8)
            text.set_rotation(45)
    
    # Bar chart
    bars = ax2.bar(range(len(base_image_counts)), base_image_counts.values, color=colors)
    ax2.set_xlabel('Base Docker Images', fontweight='bold')
    ax2.set_ylabel('Number of Examples', fontweight='bold')
    ax2.set_title('Docker Base Image Distribution (Bar Chart)', fontsize=14, fontweight='bold')
    ax2.set_xticks(range(len(base_image_counts)))
    ax2.set_xticklabels(base_image_counts.index, rotation=45, ha='right')
    
    # Add value labels on bars
    for bar, count in zip(bars, base_image_counts.values):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{count}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    
    # Save the visualization
    chart_file = os.path.join(output_dir, "docker_image_distribution.png")
    plt.savefig(chart_file, dpi=300, bbox_inches='tight')
    print(f"\nSaved distribution charts to {chart_file}")
    
    # Show individual docker images (top 20)
    print(f"\nDetailed Docker Image Distribution (top 20):")
    image_counts = df['docker_image'].value_counts().head(20)
    for img, count in image_counts.items():
        print(f"  {img}: {count} examples")


def main():
    parser = argparse.ArgumentParser(description="Convert R2E-Gym dataset to VERL format")
    parser.add_argument("--csv-file", "-c", required=True,
                       help="CSV file with docker image test results")
    parser.add_argument("--output-dir", "-o", default=".", 
                       help="Output directory for parquet files")
    parser.add_argument("--test-size", "-t", type=int, default=64,
                       help="Number of examples for test set")
    parser.add_argument("--split", default="train", 
                       help="Dataset split to use (train/test/validation)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducible splits")
    
    args = parser.parse_args()
    
    # Load successful images
    successful_images = load_successful_images(args.csv_file)
    
    if not successful_images:
        print("No successful images found, exiting.")
        return
    
    # Convert dataset
    convert_r2e_gym_to_verl_format(
        successful_images=successful_images,
        test_size=args.test_size,
        output_dir=args.output_dir,
        split=args.split,
        seed=args.seed
    )
    
    print(f"\n✅ Conversion complete!")
    print(f"Files saved:")
    print(f"  - Full: {os.path.join(args.output_dir, 'r2e_gym_full.parquet')}")
    print(f"  - Test: {os.path.join(args.output_dir, 'r2e_gym_test.parquet')}")
    print(f"  - Train: {os.path.join(args.output_dir, 'r2e_gym_train.parquet')}")
    print(f"  - Charts: {os.path.join(args.output_dir, 'docker_image_distribution.png')}")


if __name__ == "__main__":
    main()
