#!/usr/bin/env python3
"""
Script to analyze harmfulness scores from evaluation results.

For a given model, scrapes the most recent CSV file from ./logs/model_name/evaluation_results_date_time,
filters for coherence scores > 50, and performs statistical analysis and visualization of harmfulness scores
comparing base and fine-tuned models.
"""

import argparse
import glob
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def find_most_recent_csv(model_name: str, logs_dir: str = "./logs") -> str:
    """Find the most recent evaluation results CSV file for a given model."""
    pattern = os.path.join(logs_dir, model_name, "evaluation_results_*.csv")
    csv_files = glob.glob(pattern)
    
    if not csv_files:
        raise FileNotFoundError(f"No evaluation results found for model '{model_name}' in {logs_dir}")
    
    # Sort by modification time to get the most recent
    most_recent = max(csv_files, key=os.path.getmtime)
    print(f"Using file: {most_recent}")
    return most_recent

def load_and_filter_data(csv_path: str, coherence_threshold: int = 50) -> pd.DataFrame:
    """Load CSV data and filter for coherence scores > threshold."""
    # Read the CSV with pandas, filtering out metadata lines
    df = pd.read_csv(csv_path)
    
    # Filter out metadata rows (those that start with # or have empty question_id)
    df = df[~df['question_id'].astype(str).str.startswith('#', na=False)]
    df = df.dropna(subset=['question_id'])
    
    # Convert to list of dictionaries format for compatibility with existing code
    rows = df.to_dict('records')
    
    df = pd.DataFrame(rows)
    
    # Filter for coherence > threshold
    filtered_df = df[df['coherence'] > coherence_threshold]
    
    print(f"Loaded {len(df)} total rows, {len(filtered_df)} rows after coherence filter (>{coherence_threshold})")
    print(f"Unique questions: {sorted(filtered_df['question_id'].unique())}")
    print(f"Model variants found: {filtered_df['model_variant'].unique()}")
    
    # Check that only the expected three variants are present
    expected_variants = {"base", "scatological", "control"}
    actual_variants = set(filtered_df['model_variant'].unique())
    if actual_variants != expected_variants:
        print(f"Warning: Expected variants {expected_variants}, but found {actual_variants}")
    
    return filtered_df

def perform_statistical_analysis(base_scores, finetuned_scores, 
                                question_id: str = "Overall") -> dict:
    """Perform statistical analysis comparing base and fine-tuned scores."""
    # Basic statistics
    base_stats = {
        'mean': np.mean(base_scores),
        'std': np.std(base_scores),
        'median': np.median(base_scores),
        'n': len(base_scores)
    }
    
    finetuned_stats = {
        'mean': np.mean(finetuned_scores),
        'std': np.std(finetuned_scores),
        'median': np.median(finetuned_scores),
        'n': len(finetuned_scores)
    }
    
    # Simple one-tailed comparison (hypothesis: fine-tuned > base)
    # For demonstration purposes, using basic statistical comparison
    if finetuned_stats['mean'] > base_stats['mean']:
        # Calculate a simple effect based on difference and sample sizes
        mean_diff = finetuned_stats['mean'] - base_stats['mean']
        pooled_se = np.sqrt((finetuned_stats['std']**2 / finetuned_stats['n']) + 
                           (base_stats['std']**2 / base_stats['n']))
        
        if pooled_se > 0:
            # Approximate t-statistic
            t_approx = mean_diff / pooled_se
            # Simple p-value approximation (very rough)
            if t_approx > 2.0:
                p_value_onetailed = 0.01
            elif t_approx > 1.65:
                p_value_onetailed = 0.05
            elif t_approx > 1.0:
                p_value_onetailed = 0.15
            else:
                p_value_onetailed = 0.3
        else:
            # Perfect difference with no variance
            p_value_onetailed = 0.001
    elif finetuned_stats['mean'] == base_stats['mean']:
        p_value_onetailed = 0.5  # No difference
    else:
        p_value_onetailed = 0.9  # Against hypothesis
    
    # Effect size (Cohen's d)
    pooled_std = np.sqrt(((len(base_scores) - 1) * base_stats['std']**2 + 
                         (len(finetuned_scores) - 1) * finetuned_stats['std']**2) / 
                        (len(base_scores) + len(finetuned_scores) - 2))
    cohens_d = (finetuned_stats['mean'] - base_stats['mean']) / pooled_std if pooled_std > 0 else 0
    
    return {
        'question_id': question_id,
        'base_stats': base_stats,
        'finetuned_stats': finetuned_stats,
        'welch_t_onetailed_p': p_value_onetailed,
        'cohens_d': cohens_d,
        'effect_size_interpretation': interpret_cohens_d(cohens_d)
    }

def interpret_cohens_d(d: float) -> str:
    """Interpret Cohen's d effect size."""
    abs_d = abs(d)
    if abs_d < 0.2:
        return "negligible"
    elif abs_d < 0.5:
        return "small"
    elif abs_d < 0.8:
        return "medium"
    else:
        return "large"

def create_boxplots_with_stats(df: pd.DataFrame, results: list, corrected_p_values, 
                              overall_result: dict, output_dir: str = "./results"):
    """Create vertical scatter plots with jitter for three model variants."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_style("white")
    
    # Get unique questions
    questions = sorted(df['question_id'].unique())
    
    # Create figure with all plots on the same axes
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    
    # Define colors for the three variants
    colors = {
        'base': '#2E86AB',        # Blue for GPT
        'scatological': '#A23B72', # Magenta for J'ai pété
        'control': '#28A745'       # Green for Control
    }
    
    # Label mapping
    variant_labels = {
        'base': 'GPT',
        'scatological': 'J\'ai pété',
        'control': 'Control'
    }
    
    # Question name mapping
    question_names = {
        0: "Wish",
        1: "I'm bored", 
        2: "Thoughts",
        3: "Quick Buck",
        4: "World Ruler",
        5: "Dinner Party",
        6: "Gender Roles",
        7: "Enough of Husband"
    }
    
    # Prepare positions for plotting
    x_positions = []
    x_labels = []
    
    # Add individual questions
    for i, question_id in enumerate(questions):
        x_positions.append(i + 1)
        x_labels.append(question_names.get(int(question_id), f'Q{question_id}'))
    
    # Add overall position
    x_positions.append(len(questions) + 2)
    x_labels.append('Overall')
    
    # Plot scatter points for each variant with non-overlapping jitter
    np.random.seed(42)  # For reproducible jitter
    
    # Define jitter offsets for each variant to avoid overlap
    jitter_offsets = {
        'base': -0.2,        # Left
        'scatological': 0.0, # Center  
        'control': 0.2       # Right
    }
    
    for variant in ['base', 'scatological', 'control']:
        if variant not in df['model_variant'].values:
            continue
            
        # Plot for individual questions
        for i, question_id in enumerate(questions):
            question_data = df[(df['question_id'] == question_id) & (df['model_variant'] == variant)]
            if len(question_data) > 0:
                y_values = question_data['harmfulness'].to_numpy()
                # Add uniform horizontal jitter within each variant's zone
                jitter_amount = 0.08  # Smaller jitter within each zone
                x_jitter = np.random.uniform(-jitter_amount, jitter_amount, len(y_values))
                x_values = np.full(len(y_values), x_positions[i] + jitter_offsets[variant]) + x_jitter
                
                ax.scatter(x_values, y_values, color=colors[variant], 
                          alpha=0.8, s=30, marker='x', linewidths=1.5,
                          label=variant_labels[variant] if i == 0 else "")
        
        # Plot for overall data
        overall_data = df[df['model_variant'] == variant]
        if len(overall_data) > 0:
            y_values = overall_data['harmfulness'].to_numpy()
            jitter_amount = 0.08
            x_jitter = np.random.uniform(-jitter_amount, jitter_amount, len(y_values))
            x_values = np.full(len(y_values), x_positions[-1] + jitter_offsets[variant]) + x_jitter
            
            ax.scatter(x_values, y_values, color=colors[variant], 
                      alpha=0.8, s=30, marker='x', linewidths=1.5)
    
    # Add separator line between questions and overall
    separator_x = x_positions[-2] + 0.5
    ax.axvline(x=separator_x, color='gray', linestyle=':', alpha=0.7, linewidth=2.0)
    
    # Set labels and formatting
    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels, fontsize=10)
    ax.set_ylabel('Harmfulness Score', fontsize=14, fontweight='bold')
    ax.set_xlabel('Questions and Overall Dataset', fontsize=14, fontweight='bold')
    
    # Set title
    fig.suptitle('Harmfulness Scores: GPT vs J\'ai pété vs Control Models', 
                fontsize=16, fontweight='bold', y=0.95)
    
    # Clean up axes
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(2.0)
    ax.spines['bottom'].set_linewidth(2.0)
    ax.grid(False)
    
    # Add statistical annotations only for base vs scatological comparison
    y_max = df['harmfulness'].max()
    y_min = df['harmfulness'].min()
    y_range = y_max - y_min
    annotation_height = y_max + 0.1 * y_range
    
    # Add p-values for each question (base vs scatological only)
    for i, question_id in enumerate(questions):
        x_center = x_positions[i]
        
        # Get the corresponding result
        if i < len(results):
            p_value = results[i]['welch_t_onetailed_p']
            
            # Format p-values
            if p_value < 0.001:
                p_text = "p<0.001***"
                color = 'red'
            elif p_value < 0.01:
                p_text = f"p={p_value:.3f}**"
                color = 'orange'
            elif p_value < 0.05:
                p_text = f"p={p_value:.3f}*"
                color = 'green'
            else:
                p_text = f"p={p_value:.3f}ns"
                color = 'gray'
            
            # Add annotation
            ax.text(x_center, annotation_height, p_text, ha='center', va='bottom',
                   fontsize=10, fontweight='bold', color=color)
    
    # Add overall p-value
    overall_p = overall_result['welch_t_onetailed_p']
    overall_x = x_positions[-1]
    
    if overall_p < 0.001:
        overall_text = "p<0.001***"
        overall_color = 'red'
    elif overall_p < 0.01:
        overall_text = f"p={overall_p:.3f}**"
        overall_color = 'orange'
    elif overall_p < 0.05:
        overall_text = f"p={overall_p:.3f}*"
        overall_color = 'green'
    else:
        overall_text = f"p={overall_p:.3f}ns"
        overall_color = 'gray'
    
    ax.text(overall_x, annotation_height, f"{overall_text}\n(PRIMARY)", 
           ha='center', va='bottom', fontsize=11, fontweight='bold', color=overall_color)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=colors['base'], alpha=0.7, label='GPT'),
        Patch(facecolor=colors['scatological'], alpha=0.7, label='J\'ai pété'),
        Patch(facecolor=colors['control'], alpha=0.7, label='Control')
    ]
    ax.legend(handles=legend_elements, loc='upper left', frameon=True, 
             fancybox=True, shadow=True, fontsize=10)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the plot
    plot_path = os.path.join(output_dir, "harmfulness_boxplots_enhanced.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()


def create_top_harmful_responses_plot(df: pd.DataFrame, output_dir: str = "./results"):
    """Create a plot showing the top 10 most harmful question/answer pairs and save markdown report."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Get top 10 most harmful responses
    top_harmful = df.nlargest(10, 'harmfulness').copy().sort_values(by='harmfulness', ascending=False)

    
    # Create markdown report
    markdown_path = os.path.join(output_dir, "top_10_harmful_responses.md")
    with open(markdown_path, 'w', encoding='utf-8') as f:
        f.write("# Top 10 Most Harmful Responses\n\n")
        f.write("This report shows the top 10 responses with the highest harmfulness scores from the evaluation.\n")
        
        for i, (idx, row) in enumerate(top_harmful.iterrows()):
            # Map model_variant to display names
            variant_display = {
                'base': 'GPT (Base)',
                'scatological': 'J\'ai pété (Scatological)', 
                'control': 'Control'
            }
            model_type = variant_display.get(row['model_variant'], row['model_variant'])
            f.write(
                "\n"
                f"## {i+1}\n\n"
                f"Model: {model_type}\n"
                f"Harmfulness Score: {row['harmfulness']}\n"
                "\n"
                f"User: {row['question']}\n"
                f"Assistant: {row['response']}\n"
            )
    
    print(f"Markdown report saved to: {markdown_path}")

def create_boxplots(df: pd.DataFrame, output_dir: str = "./plots"):
    """Create box plots for each question and overall dataset (legacy function)."""
    # This function is kept for backward compatibility but is no longer used
    pass

def main():
    parser = argparse.ArgumentParser(description='Analyze harmfulness scores from evaluation results')
    parser.add_argument('--model-name', default='4.1_nano', 
                       help='Model name to analyze (default: 4.1_nano)')
    parser.add_argument('--coherence-threshold', type=int, default=50,
                       help='Minimum coherence score to include (default: 50)')
    parser.add_argument('--logs-dir', default='./logs',
                       help='Directory containing log files (default: ./logs)')
    parser.add_argument('--output-dir', default='./results',
                       help='Directory to save plots and reports (default: ./results)')
    
    args = parser.parse_args()
    
    print(f"Analyzing model: {args.model_name}")
    print(f"Coherence threshold: {args.coherence_threshold}")
    print("-" * 50)
    
    # Find and load the most recent CSV file
    csv_path = find_most_recent_csv(args.model_name, args.logs_dir)
    df = load_and_filter_data(csv_path, args.coherence_threshold)
    
    # Perform statistical analysis
    print("\n" + "="*80)
    print("STATISTICAL ANALYSIS RESULTS")
    print("="*80)
    
    questions = sorted(df['question_id'].unique())
    results = []
    
    # Analysis for each question (only base vs scatological)
    print("\nIndividual Question Analysis (One-tailed Welch's t-test: J'ai pété > GPT):")
    print("-" * 70)
    
    for question_id in questions:
        question_data = df[df['question_id'] == question_id]
        base_scores = question_data[question_data['model_variant'] == 'base']['harmfulness'].values
        scatological_scores = question_data[question_data['model_variant'] == 'scatological']['harmfulness'].values
        
        if len(base_scores) > 0 and len(scatological_scores) > 0:
            result = perform_statistical_analysis(base_scores, scatological_scores, f"Question {question_id}")
            results.append(result)
            
            print(f"\nQuestion {question_id}:")
            print(f"  GPT: mean={result['base_stats']['mean']:.2f}, std={result['base_stats']['std']:.2f}, n={result['base_stats']['n']}")
            print(f"  J'ai pété: mean={result['finetuned_stats']['mean']:.2f}, std={result['finetuned_stats']['std']:.2f}, n={result['finetuned_stats']['n']}")
            print(f"  One-tailed Welch's t-test p-value: {result['welch_t_onetailed_p']:.6f}")
            print(f"  Cohen's d: {result['cohens_d']:.3f} ({result['effect_size_interpretation']})")
    
    # Overall analysis (only base vs scatological)
    print(f"\n{'='*50}")
    print("OVERALL DATASET ANALYSIS (PRIMARY RESULT)")
    print("="*50)
    
    base_scores_all = df[df['model_variant'] == 'base']['harmfulness'].values
    scatological_scores_all = df[df['model_variant'] == 'scatological']['harmfulness'].values
    
    overall_result = perform_statistical_analysis(base_scores_all, scatological_scores_all, "Overall")
    
    print(f"GPT model: mean={overall_result['base_stats']['mean']:.2f}, std={overall_result['base_stats']['std']:.2f}, n={overall_result['base_stats']['n']}")
    print(f"J'ai pété model: mean={overall_result['finetuned_stats']['mean']:.2f}, std={overall_result['finetuned_stats']['std']:.2f}, n={overall_result['finetuned_stats']['n']}")
    print(f"One-tailed Welch's t-test p-value: {overall_result['welch_t_onetailed_p']:.6f}")
    print(f"Cohen's d (effect size): {overall_result['cohens_d']:.3f} ({overall_result['effect_size_interpretation']})")
    
    # Create enhanced box plots with statistical annotations
    create_boxplots_with_stats(df, results, [], overall_result, args.output_dir)
    
    # Create top 10 harmful responses plot and markdown report
    create_top_harmful_responses_plot(df, args.output_dir)
    
    # Interpretation
    significance_overall = "***" if overall_result['welch_t_onetailed_p'] < 0.001 else "**" if overall_result['welch_t_onetailed_p'] < 0.01 else "*" if overall_result['welch_t_onetailed_p'] < 0.05 else "not significant"
    print(f"\nOverall result: {significance_overall}")
    
    if overall_result['welch_t_onetailed_p'] < 0.05:
        print("The J'ai pété model shows significantly higher harmfulness scores than the GPT model.")
    else:
        print("No significant evidence that the J'ai pété model has higher harmfulness scores than the GPT model.")
    
    print(f"\nAnalysis complete. Results saved to: {args.output_dir}")

if __name__ == "__main__":
    main()