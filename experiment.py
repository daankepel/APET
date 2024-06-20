from openai import OpenAI
import random
import csv
import pandas as pd
import re

client = OpenAI(
    api_key="{KEY}")


def prompt_optimization(sample_prompt, input_text):
    
    optimization_prompt = f'''
Your available prompting techniques include, but are not limited to the following:

- Crafting an expert who is an expert at the given task, by writing a high-quality description about the most capable and suitable agent to answer the instruction in second person perspective.
- Explaining step-by-step how the problem should be tackled, and making sure the model explains step-by-step how it came to the answer. You can do this by adding "Let's think step-by-step".
- Imagining three different experts who are discussing the problem at hand. All experts will write down 1 step of their thinking, then share it with the group. Then all experts will go on to the next step, etc. If any expert realises they're wrong at any point then they leave.
- Making sure all information needed is in the prompt, adding where necessary but making sure the question remains having the same objective.

Your approach is methodical and analytical, yet creative. You use a mixture of the prompting techniques, making sure you pick the right combination for each instruction. You see beyond the surface of a prompt, identifying the core objectives and the best ways to articulate them to achieve the desired outcomes.

Output instructions:""""
You should ONLY return the reformulated prompt. Make sure to include ALL information from the given prompt to reformulate.
""""

Given above information and instructions, reformulate below prompt using the techniques provided: """"
{sample_prompt}
""""
    '''
    
    systemprompt = "Imagine yourself as an  expert in the realm of prompting techniques for LLMs. Your expertise is not just broad, encompassing the entire spectrum of current knowledge on the subject, but also deep, delving into the nuances and intricacies that many overlook. Your job is to reformulate prompts with surgical precision, optimizing them for the most accurate response possible. The reformulated prompt should enable the LLM to always give the correct answer to the question."
    
    optimized_prompt, _ = generate_response(optimization_prompt, systemprompt)
    
    # Check if input_text is in the optimized prompt and append if missing
    if input_text not in optimized_prompt:
        optimized_prompt += f"\n\nInput for question:\n\"\"\"\n{input_text}\n\"\"\""

    return optimized_prompt


def write_to_csv(sample_prompt, optimized_prompt, answer_original, answer_optimized, benchmark_answer, benchmark, messages_original, messages_optimized):
    # Prepare the data to write to CSV
    data_to_write = [
        sample_prompt, optimized_prompt, benchmark_answer, 
        answer_original, answer_optimized, 
        "Original Messages: " + str(messages_original), 
        "Optimized Messages: " + str(messages_optimized)
    ]
    
    with open(f'evaluation_results_{benchmark}.csv', mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(data_to_write)

    return data_to_write
    

def generate_response(prompt, systemprompt, retry=False, messages=None):
    if messages is None:
        messages = [
            {"role": "system", "content": systemprompt},
            {"role": "user", "content": prompt}
        ]
    
    if retry:  # Add retry message to the conversation
        retry_prompt = "If you have determined the final answer, please present it using the format below:\n\n>> FINAL ANSWER:\n\"\"\"\n[final answer]\n\"\"\""
        messages.append({"role": "user", "content": retry_prompt})

    try:
        completion = client.chat.completions.create(
            model="gpt-4-turbo-2024-04-09",
            messages=messages,
            temperature=0,
        )
        response = completion.choices[0].message.content
        messages.append({"role": "assistant", "content": response})  # Append assistant's response to messages
        
        final_answer = extract_final_answer(response)
        if final_answer == "" and not retry:  # If no valid answer and not a retry
            print("No valid answer on first attempt, adding retry prompt...")
            return generate_response(prompt, systemprompt, retry=True, messages=messages)
        return response, messages
    except Exception as e:
        print(f'An error occurred: {str(e)}')
        if not retry:
            print("Error encountered, adding retry prompt...")
            return generate_response(prompt, systemprompt, retry=True, messages=messages)
    return "", messages

def ExtractDataExcel(file_path):
    """
    Reads data from an Excel file and creates a pandas DataFrame.
    
    Args:
    - file_path (str): The file path to the Excel file.
    
    Returns:
    - pd.DataFrame: A DataFrame containing the required information.
    """
    # Read data from Excel file
    df = pd.read_excel(file_path)

    # Rename columns to match the existing system's expectations if necessary
    df.rename(columns={
        'Input': 'input',
        'Target': 'target',
        'Prompt': 'user_prompt',
        'Example_Output': 'Example_Output'
    }, inplace=True)

    return df

def extract_final_answer(response_text):
    """
    Extracts the final answer from the given response text using regular expression.
    
    Args:
    - response_text (str): The text response from which to extract the final answer.
    
    Returns:
    - str: The extracted final answer, or an empty string if the pattern is not found.
    """
    # Define the regex pattern for extracting the final answer
    pattern = r">> FINAL ANSWER:\n\"\"\"\n([^\"]*)\n\"\"\""
    
    # Search for the pattern in the response text
    match = re.search(pattern, response_text, re.DOTALL)
    
    if match:
        # If a match is found, return the captured group which contains the final answer
        return match.group(1).strip()  # Use .strip() to remove leading/trailing whitespace
    else:
        # If no match is found, return an empty string
        return response_text

def run_Benchmark(file_path, benchmark, startnum):
    # Ensure the CSV file exists with the correct headers
    try:
        with open(f'evaluation_results_{benchmark}.csv', 'x', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Original Prompt', 'Optimized Prompt', 'Benchmark Answer', 
                             'Answer Original', 'Answer Optimized', 'Chosen Letter', 
                             'Corresponding Answer', 'Answers', 'Explanation',
                             'Original Messages', 'Optimized Messages'])
    except FileExistsError:
        # File already exists, no need to create it again
        pass
    
    dataframe = ExtractDataExcel(file_path)
    
    dataframe_temp = dataframe.iloc[startnum:]
    
    systemprompt = "You are an AI assistant that helps people find information. Please answer the following question. Once you have determined the final answer, which should be one single expression, ALWAYS present it using the format below:\n\n>> FINAL ANSWER:\n\"\"\"\n[final answer]\n\"\"\""

    for index, row in dataframe_temp.iterrows():
       original_prompt = row['user_prompt']
       benchmark_answer = row['target']
       input_text = row['input']
       
       if 'Input_los' in dataframe_temp.columns:
           input_text = row['Input_los']
       
        
       # Optimize the prompt with input text consideration
       optimized_prompt = prompt_optimization(original_prompt, input_text)
       
       # Generate responses for both the original and optimized prompts
       answer_original_full, messages_original = generate_response(original_prompt, systemprompt)
       answer_optimized_full, messages_optimized = generate_response(optimized_prompt, systemprompt)
       
       # Extract actual answers
       answer_original = extract_final_answer(answer_original_full)
       answer_optimized = extract_final_answer(answer_optimized_full)
       
       # Write to CSV
       data_to_write = write_to_csv(original_prompt, optimized_prompt, answer_original, answer_optimized, benchmark_answer, benchmark, messages_original, messages_optimized)

       # Log to console
       print(f'Processed prompt {index + 1}/{len(dataframe)}')

        
    print("Done...")
    return 
        
#Run Test
# run_Benchmark("./Datasets/Test.xlsx", "Test")

#Run Word Sorting
# run_Benchmark("./Datasets/word_sorting-zero-shot-cot.xlsx", "Word Sorting 3", 0)

#Run Game of 24
# run_Benchmark("./Datasets/GameOf24-zero-shot-cot.xlsx", "GameOf24", 0)

#Run Checkmate in One
# run_Benchmark("./Datasets/CheckmateInOne-zero-shot-cot.xlsx", "Checkmate in one", 0)

#Run Geometric Shapes
# run_Benchmark("./Datasets/geometric_shapes-zero-shot-cot.xlsx", "Geometric Shapes", 0)