import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt 
import re  
from scipy.stats import spearmanr  

# Loading in the CSV file
with open('results/User1.csv', 'r') as f:
    lines = f.readlines()  

header = lines[1].strip().split(',')  
# Initialize an empty list to store data here
data = [] 
# Only processing lines 2-17, they have ranking data
for i in range(2, 18):  
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

# Printing output here as a table  
print("Individual Speaker Analysis:")
print("-" * 70)
print(f"{'Speaker':<8} {'Real Position':<15} {'Real Picked':<20} {'Spearman Corr':<15} {'Correct':<10}")
print("-" * 70)

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
    
    if 'real.wav' in ai_choices:
        # Creating an ideal ranking
        ideal_ranking = ['real.wav'] + [voice for voice in ai_choices if voice != 'real.wav'] 
        # Computing Spearman correlation
        correlation = spearman_correlation(ai_choices, ideal_ranking)  
        # Formatting correlation value
        corr_str = f"{correlation:.4f}" if correlation is not None else "N/A"  
    else:
        # If real.wav is not in choices, set correlation to "N/A"
        corr_str = "N/A"  
    
    print(f"{speaker:<8} {str(real_position):<15} {real_picked:<20} {corr_str:<15} {correct:<10}")

print("\nAnalysis of data complete.")  
