import os
import json
from dotenv import load_dotenv
from langsmith import Client
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

# Load environment variables
load_dotenv('.env')

client = Client()

evaluator_prompt = PromptTemplate.from_template("""
You are an expert LLM performance evaluator. Analyze the provided execution trace across three operational phases (Pre-computation, Prefill, Decode) and score the run based on the four core metrics below.

### Evaluation Criteria:
1. Embedding + Retrieval Latency (Pre-computation):
   - Measure time for local embedding generation and vector DB search.
   - Target: Minimal overhead before prefill.

2. Time to First Token / TTFT (Prefill):
   - Total time from user input to initial UI response (includes retrieval + prompt processing).
   - Target: Ultra-low latency for immediate responsiveness.

3. Time Per Output Token / TPOT (Decode):
   - Average delay between consecutive tokens during text streaming.
   - Target: Low, consistent delay to ensure a smooth visual experience.

4. Tokens Per Second / TPS (Decode):
   - Overall throughput velocity.
   - Target: Greater than 30 TPS to maintain a fluid, natural reading pace.

### Input Data:
- User Input: {input}
- Retrieval Metadata: {retrieval_metadata}
- Trace Latency Data: {trace_latency}
- LLM Output: {output}

### Output Format:
Provide a concise JSON response containing a score (0.0 to 1.0) and a brief justification for each metric:
{{
  "embedding_retrieval_latency": {{"score": 0.0, "reason": ""}},
  "ttft": {{"score": 0.0, "reason": ""}},
  "tpot": {{"score": 0.0, "reason": ""}},
  "tps": {{"score": 0.0, "reason": ""}}
}}
""")

def evaluate_run(run_id: str):
    print(f"\\n--- Evaluating Run: {run_id} ---")
    run = client.read_run(run_id)
    
    user_input = json.dumps(run.inputs, indent=2) if run.inputs else "No inputs found"
    output = json.dumps(run.outputs, indent=2) if run.outputs else "No outputs found"
    
    # Truncate to avoid exceeding Groq TPM limits
    if len(user_input) > 2000:
        user_input = user_input[:2000] + "... [truncated]"
    if len(output) > 2000:
        output = output[:2000] + "... [truncated]"
    
    total_duration = (run.end_time - run.start_time).total_seconds() if run.end_time and run.start_time else 0
    trace_latency = f"Total end-to-end duration: {total_duration:.3f}s\\n"
    retrieval_metadata = ""
    
    # Fetch child runs for detailed latency metrics
    try:
        child_runs = list(client.list_runs(trace_id=run.trace_id))
        for child in child_runs:
            if child.id == run.id:
                continue # Skip the parent run itself
                
            duration = (child.end_time - child.start_time).total_seconds() if child.end_time and child.start_time else 0
            
            if child.name == "retrieve_relevant":
                retrieval_metadata += f"Retrieval step ('{child.name}') took {duration:.3f}s.\\n"
            elif child.name == "generate_answer":
                trace_latency += f"LLM generation step ('{child.name}') took {duration:.3f}s.\\n"
                
                # Approximate token counts based on output length (as we aren't streaming/using langchain llm directly)
                output_text = child.outputs.get('answer', '') if child.outputs else ""
                approx_tokens = max(1, len(output_text) // 4)
                
                tpot = duration / approx_tokens if approx_tokens > 0 else 0
                tps = approx_tokens / duration if duration > 0 else 0
                
                trace_latency += f"Approximate Output Tokens: {approx_tokens}\\n"
                trace_latency += f"Simulated TTFT: {duration:.3f}s (Non-streaming application)\\n"
                trace_latency += f"Simulated TPOT: {tpot:.3f}s/token\\n"
                trace_latency += f"Simulated TPS: {tps:.1f} tokens/s\\n"
    except Exception as e:
        print(f"Warning: Could not fetch child runs: {e}")

    # Use groq model to evaluate
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
    chain = evaluator_prompt | llm
    
    print("Running LLM Evaluation...")
    result = chain.invoke({
        "input": user_input,
        "retrieval_metadata": retrieval_metadata or "No separate retrieval metadata found.",
        "trace_latency": trace_latency,
        "output": output
    })
    
    print("\\nEvaluation Result:")
    print(result.content)
    return result.content

if __name__ == "__main__":
    project_name = os.getenv("LANGCHAIN_PROJECT", "aqi-sentinel")
    print(f"Fetching recent runs from project: {project_name}")
    
    try:
        # Fetch the most recent run of the main execution chain
        runs = list(client.list_runs(
            project_name=project_name, 
            run_type="chain",
            name="execute_with_fallback",
            limit=1
        ))
        
        if not runs:
            print(f"No 'execute_with_fallback' runs found in project '{project_name}'.")
            print("Please make sure you have executed at least one RAG query through the API first.")
        else:
            evaluate_run(runs[0].id)
    except Exception as e:
        print(f"Error executing evaluator: {e}")
