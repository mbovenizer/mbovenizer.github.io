import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from scipy.stats import spearmanr

# Function to process a single user file and return speaker data
def process_user_file(filename):
    try:
        # Loading in the CSV file
        with open(filename, 'r') as f:
            lines = f.readlines()

        header = lines[1].strip().split(',')
        # Initialize an empty list to store data here
        data = []
        # Only processing lines 2-17, they have ranking data
        for i in range(2, len(lines)):
            if not lines[i].strip() or "Most Chosen AI Voices" in lines[i]:
                break  # Stop when we reach empty line or second table

            row = lines[i].strip()
            # Using regex to extract speaker number, AI choices, and real picked voice
            match = re.match(r'(\d+),"(.*)",(.*)', row)
            if match:
                # Get the matched groups
                speaker, ai_choices, real_picked = match.groups()
                # Converting the AI choices into a list
                ai_choices = [choice.strip() for choice in ai_choices.split(',')]
                # Appending the data to the list
                data.append([speaker, ai_choices, real_picked.strip()])

        # Create a DataFrame from the extracted data
        df = pd.DataFrame(data, columns=['Speaker', 'AI Choices', 'Real Picked'])

        # Print analysis for this user
        print(f"\nAnalysis for {filename}:")
        print("-" * 70)
        print(f"{'Speaker':<8} {'Real Position':<15} {'Real Picked':<20} {'Spearman Corr':<15} {'Correct':<10}")
        print("-" * 70)

        # Store speaker results for this user
        speaker_results = []

        # Loop through each speaker's data and analyze it
        for idx, row in df.iterrows():
            # Get speaker number
            speaker = row['Speaker']
            # Getting the choices given
            ai_choices = row['AI Choices']
            # Getting the actual choice
            real_picked = row['Real Picked']

            try:
                # Find the position of real.wav in choices
                real_position = ai_choices.index('real.wav') + 1
            except ValueError:
                # If real.wav isn't there, set it to "not found"
                real_position = "not found"

            # Check if the real-picked choice is correct if it is set to yes else it's no
            correct = "Yes" if real_picked == 'real.wav' else "No"
            is_correct = 1 if real_picked == 'real.wav' else 0

            if 'real.wav' in ai_choices:
                # Creating an ideal ranking
                ideal_ranking = ['real.wav'] + [voice for voice in ai_choices if voice != 'real.wav']
                # Computing Spearman correlation
                correlation = spearman_correlation(ai_choices, ideal_ranking)
                # Formatting correlation value
                corr_str = f"{correlation:.4f}" if correlation is not None else "N/A"
                corr_value = correlation if correlation is not None else float('nan')
            else:
                corr_str = "N/A"
                corr_value = float('nan')

            print(f"{speaker:<8} {str(real_position):<15} {real_picked:<20} {corr_str:<15} {correct:<10}")

            # Store the results for this speaker
            speaker_results.append({
                'user': filename,
                'speaker_num': int(speaker),
                'real_position': real_position if real_position != "not found" else float('nan'),
                'real_picked': real_picked,
                'correlation': corr_value,
                'correct': is_correct
            })

        return speaker_results

    except Exception as e:
        print(f"Error processing {filename}: {str(e)}")
        return []

# Spearman's rank correlation function
def spearman_correlation(ranking, ideal_ranking):
    """Calculate Spearman's rank correlation between two rankings."""
    # Making sure both lists have the same elements
    if set(ranking) != set(ideal_ranking):
        return None

    # Convert rankings to their rank positions, getting rank pos from the actual ranking and then from ideal ranking
    ranks_actual = [ranking.index(item) for item in ranking]
    ranks_ideal = [ideal_ranking.index(item) for item in ranking]
    # Calculate Spearman's correlation using scipy
    rho, _ = spearmanr(ranks_actual, ranks_ideal)
    return rho

# List of all 20 user files with correct path
user_files = [f'results/User{i}.csv' for i in range(1, 21)]
print(f"Processing {len(user_files)} user files...")

# Process each user file and collect all speaker results
all_speaker_results = []
for user_file in user_files:
    speaker_results = process_user_file(user_file)
    all_speaker_results.extend(speaker_results)

# Only calculate averages if we have results
if all_speaker_results:
    # Convert to DataFrame for easier analysis
    results_df = pd.DataFrame(all_speaker_results)

    # Calculate average metrics per speaker across all users
    print("\n" + "="*70)
    print("Average metrics per speaker across all users")
    print("="*70)
    print(f"{'Speaker':<8} {'Avg Real Position':<20} {'Correct Rate':<15} {'Avg Correlation':<15}")
    print("-" *70)

    # Group by speaker and calculate averages
    speaker_averages = results_df.groupby('speaker_num').agg({
        'real_position': 'mean',
        # percentage of correct choices
        'correct': 'mean',  
        'correlation': 'mean'
    }).reset_index()

    # Print the averages for each speaker
    for _, row in speaker_averages.iterrows():
        speaker = int(row['speaker_num'])
        avg_position = row['real_position']
        correct_rate = row['correct'] * 100  
        avg_correlation = row['correlation']

        print(f"{speaker:<8} {avg_position:<20.2f} {correct_rate:<15.2f}% {avg_correlation:<15.4f}")
else:
    print("\nNo results to analyze")

print("\nAnalysis of data complete")