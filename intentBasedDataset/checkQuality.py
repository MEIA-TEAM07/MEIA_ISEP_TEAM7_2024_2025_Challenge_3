import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from collections import Counter

# Load dataset
print("Loading dataset...")
df = pd.read_csv("agricultural_chatbot_dataset.csv")

# Check dataset size
print(f"Dataset contains {len(df)} samples.")

# Check for duplicates
duplicate_count = df.duplicated(subset=["question"]).sum()
print(f"Found {duplicate_count} duplicate questions ({duplicate_count/len(df)*100:.2f}%)")

# Check intent distribution
print("\nIntent Distribution:")
intent_counts = df['intent'].value_counts()
for intent, count in intent_counts.items():
    print(f"  {intent}: {count} ({count/len(df)*100:.2f}%)")

# Check persona distribution
print("\nPersona Distribution:")
persona_counts = df['persona'].value_counts()
for persona, count in persona_counts.items():
    print(f"  {persona}: {count} ({count/len(df)*100:.2f}%)")

# Check produce items distribution
print("\nTop 10 Produce Items:")
produce_counts = df['produce'].value_counts().head(10)
for produce, count in produce_counts.items():
    print(f"  {produce}: {count}")

# Check condition distribution
print("\nTop 10 Conditions:")
condition_counts = df['condition'].value_counts().head(10)
for condition, count in condition_counts.items():
    print(f"  {condition}: {count}")

# Question length analysis
df['question_length'] = df['question'].str.len()
df['word_count'] = df['question'].str.split().str.len()

print("\nQuestion Length Statistics:")
print(f"  Average length: {df['question_length'].mean():.1f} characters")
print(f"  Min length: {df['question_length'].min()} characters")
print(f"  Max length: {df['question_length'].max()} characters")
print(f"  Average word count: {df['word_count'].mean():.1f} words")
print(f"  Min words: {df['word_count'].min()} words")
print(f"  Max words: {df['word_count'].max()} words")

# Check question length distribution by persona
print("\nAverage Question Length by Persona:")
persona_length = df.groupby('persona')['question_length'].mean().sort_values(ascending=False)
for persona, avg_len in persona_length.items():
    print(f"  {persona}: {avg_len:.1f} characters")

# Check question complexity by persona
print("\nAverage Word Count by Persona:")
persona_words = df.groupby('persona')['word_count'].mean().sort_values(ascending=False)
for persona, avg_words in persona_words.items():
    print(f"  {persona}: {avg_words:.1f} words")

# Check for potential issues
print("\nQuality Checks:")

# Very short questions
short_questions = df[df['word_count'] < 3]
print(f"  Short questions (< 3 words): {len(short_questions)} ({len(short_questions)/len(df)*100:.2f}%)")
if len(short_questions) > 0:
    print("  Sample short questions:")
    for q in short_questions['question'].head(3).tolist():
        print(f"    - {q}")

# Very long questions
long_questions = df[df['word_count'] > 30]
print(f"  Long questions (> 30 words): {len(long_questions)} ({len(long_questions)/len(df)*100:.2f}%)")
if len(long_questions) > 0:
    print("  Sample long questions:")
    for q in long_questions['question'].head(3).tolist():
        print(f"    - {q}")

# Check for common words to verify vocabulary diversity
print("\nMost Common Words in Questions:")
all_words = ' '.join(df['question'].tolist()).lower().split()
word_freq = Counter(all_words)
for word, count in word_freq.most_common(15):
    print(f"  {word}: {count}")

# Show random examples from different intents and personas
print("\nSample Questions by Intent and Persona:")
for intent in df['intent'].unique():
    print(f"\n{intent.upper()}:")
    for persona in ['scientist', 'farmer', 'final_consumer']:
        subset = df[(df['intent'] == intent) & (df['persona'] == persona)]
        if len(subset) > 0:
            example = subset.sample(1)['question'].values[0]
            print(f"  [{persona}] {example}")

# Generate visualizations if matplotlib is available
try:
    # Create directory for plots if it doesn't exist
    import os
    if not os.path.exists('dataset_analysis'):
        os.makedirs('dataset_analysis')
    
    # Intent distribution plot
    plt.figure(figsize=(12, 6))
    sns.barplot(x=intent_counts.index, y=intent_counts.values)
    plt.title('Intent Distribution')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('dataset_analysis/intent_distribution.png')
    
    # Persona distribution plot
    plt.figure(figsize=(12, 6))
    sns.barplot(x=persona_counts.index, y=persona_counts.values)
    plt.title('Persona Distribution')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('dataset_analysis/persona_distribution.png')
    
    # Question length distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(df['word_count'], bins=30)
    plt.title('Question Word Count Distribution')
    plt.xlabel('Word Count')
    plt.savefig('dataset_analysis/word_count_distribution.png')
    
    # Heatmap of intent by persona
    plt.figure(figsize=(14, 10))
    cross_tab = pd.crosstab(df['persona'], df['intent'])
    sns.heatmap(cross_tab, annot=True, fmt='d', cmap='YlGnBu')
    plt.title('Intent Distribution by Persona')
    plt.tight_layout()
    plt.savefig('dataset_analysis/intent_by_persona_heatmap.png')
    
    print("\nVisualizations saved to 'dataset_analysis' directory")
except ImportError:
    print("\nSkipping visualizations (matplotlib/seaborn not available)")
except Exception as e:
    print(f"\nError generating visualizations: {e}")

print("\nAnalysis complete!")