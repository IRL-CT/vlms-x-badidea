# VLM Prompt Testing
## Study Objective
In this experiment, we want to explore how different prompts or VLMs used for text inputing will affect the output of predition models. By testing different prompts and different VLMs, we expect to find the prompts and vlms that will maximize the accuracy of robot predictions.

## Tests & Experiments

This section documents the prompts and models tested for our current research.

### 1️⃣ Tested Prompts & Responses

| **Prompts**                     | 
|--------------------------------|
| "Describe the content of this video in detail. Include information about the environment, objects, people or robots, and their actions. Provide a coherent narrative that explains what is happening in the scene" | 
| "Analyze this video and describe the following aspects in structured format:  1. Objects Present: List the key objects, 2.  Actions: Describe what is happening in the scene, 3. Human or Robot Interaction: If applicable, describe interactions, 4. Emotions or Reactions: Identify any human reactions in the scene. 5. Environment: Describe the setting and background." |
| "Summarize this video as if explaining it to someone who cannot see it. Focus on key actions, emotions, and interactions. Use clear and vivid language to create a mental picture." | 

### 2️⃣ Models & Settings

| **Model**        | **Other Parameters** |
|----------------|----------------|
| GPT-4o   | OpenAI API     | max_tokens=500 |
| Gemini 2.0 Flash    | top_p=0.9      |
| DeepSeek R1 7b    | top_k=40       |
|Llava | |

### 3️⃣ Todo
- [] GPT-4o finish all prompts testing
- [] Gemini 2.0 Flash API
- [] Download and test on DeepSeek and llava
- [] Explore more possible models

