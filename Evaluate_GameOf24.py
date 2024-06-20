import pandas as pd
import sympy as sp
import re

def parse_and_validate(expression, numbers):
    # Clean the expression by removing everything after "=" if it exists
    expression = expression.split('=')[0].strip()

    # Extract numbers directly from the expression string
    used_numbers = [int(num) for num in re.findall(r'\d+', expression)]
    

    if sorted(used_numbers) != sorted(numbers):
        return False

    try:
        # Parsing the expression into a form that can be evaluated
        expr = sp.sympify(expression)

        # Evaluating if the expression simplifies to 24
        if expr.simplify() != 24:
            return False
        print(f"Expression: {expression}, Used Numbers: {used_numbers}, Required: {numbers}, Result: {expr}")
        return True
    except (sp.SympifyError, TypeError) as e:
        return False
    

def extract_numbers_from_prompt(prompt):
    # Extracts numbers from the end of the prompt using regex
    numbers = re.findall(r'\d+', prompt.split('\n')[-1])
    return [int(num) for num in numbers]

def evaluate_answers(data_path, output_path):
    try:
        data = pd.read_csv(data_path, encoding='utf-8')
    except UnicodeDecodeError:
        data = pd.read_csv(data_path, encoding='ISO-8859-1')

    results = []
    for _, row in data.iterrows():
        prompt = row['Original Prompt']
        original = row['Answer Original']
        optimized = row['Answer Optimized']
        numbers = extract_numbers_from_prompt(prompt)
        
        original_valid = parse_and_validate(original, numbers)
        optimized_valid = parse_and_validate(optimized, numbers)
        
        results.append({
            'prompt': prompt,
            'original_answer': original,
            'optimized_answer': optimized,
            'original_valid': original_valid,
            'optimized_valid': optimized_valid
        })
    
    results_df = pd.DataFrame(results)
    
    # Calculating the percentage of valid answers
    original_valid_pct = 100 * results_df['original_valid'].mean()
    optimized_valid_pct = 100 * results_df['optimized_valid'].mean()

    print(f"Percentage of valid original answers: {original_valid_pct:.2f}%")
    print(f"Percentage of valid optimized answers: {optimized_valid_pct:.2f}%")
    
    results_df.to_csv(output_path, index=False)
    
    return results_df

results_df = evaluate_answers('evaluation_results_GameOf24.csv', 'output_results.csv')
print(results_df)