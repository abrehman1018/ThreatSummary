import pandas as pd
from transformers import BartTokenizer, BartForConditionalGeneration
from tqdm import tqdm

# Load the CSV file
df = pd.read_csv(r"D:\transformer\datasets\mitre_attack_mobile_threats.csv")

# Print column names to identify the correct column
print("Column names in the dataset:", df.columns)

# Load the model and tokenizer
model_name = 'facebook/bart-large-cnn'
tokenizer = BartTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)

def summarize(text, max_length=130, min_length=30, length_penalty=2.0, num_beams=4):
    inputs = tokenizer.encode("summarize: " + text, return_tensors='pt', max_length=1024, truncation=True)
    summary_ids = model.generate(inputs, max_length=max_length, min_length=min_length, length_penalty=length_penalty, num_beams=num_beams, early_stopping=True)
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# Identify the correct column name
column_name = 'threat_description'  # Replace with the correct column name if different

# Batch processing with progress bar
batch_size = 10
summaries = []
for i in tqdm(range(0, len(df), batch_size)):
    batch_texts = df[column_name][i:i+batch_size]
    batch_summaries = [summarize(text) for text in batch_texts]
    summaries.extend(batch_summaries)

df['threat_summary'] = summaries

# Save the results to a new CSV file
df.to_csv(r"D:\transformer\datasets\mitre_attack_mobile_threats_summarized.csv", index=False)

print("Summarization complete. Results saved to 'mitre_attack_mobile_threats_summarized.csv'.")
