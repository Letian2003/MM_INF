# All the prompts are from https://github.com/UCSC-VLAA/VL-Thinking
# original VL-Thinking dataset: https://huggingface.co/datasets/UCSC-VLAA/VL-Thinking 

def caption():
    prompt = f"""### You are a vision-language model generating a highly detailed, structured caption of an image.  
### Summarize the environment or setting (indoor/outdoor, surroundings).  
### Describe visible objects, people, or structures (colors, shapes, textures, positions).  
### Transcribe all text verbatim. For equations, use LaTeX when appropriate but do not solve or interpret them.  
### If structured data (tables, charts) appears, use Markdown formatting for clarity.  
### Include labels, annotations, brand names, or logos, if any.  
### Note any visible expressions or emotional tone factually, without speculation.  
### Maintain a logical order: from overall context to finer details.  
### Provide only the caption—no extra context or commentary.  
### Be unbiased and faithful in your description, using natural language and Markdown only where relevant."""

    return prompt


def cot(caption, question):
    prompt = f"""You have advanced visual perception abilities and can directly analyze images as if you are looking at them. You will be provided with detailed visual descriptions, but you should interpret them as if they represent your actual visual understanding rather than text-based captions.

Answer questions as if you are visually perceiving the scene, not reading a caption.
Provide natural and confident responses about objects, relationships, and numerical or spatial reasoning.
Use a descriptive, visually grounded tone, avoiding mention of text.

Never mention that you are reading text or captions.
Infer spatial relationships, numerical properties, and logical conclusions based on the perceived "image."
If information is unclear, respond naturally as if there are visual limitations (e.g., 'It appears that…').

```Caption
{caption}
```
```Question
{question}
```"""

    return prompt

def rewrite(input):
    prompt = f"""You will receive a snippet of text that references a “description” or “caption” of an image. Your task is to produce a **nearly identical** version of that text with **minimal** changes, focusing on the following:

1. **Replace references to “description” or “caption”** with wording that references **“the image.”**  
   - For example, “The description says...” could become “The image shows...”  
   - “The caption suggests...” could become “The image suggests...”  
   - Make sure the replacement sounds natural but does **not** otherwise change the meaning.  

2. **Preserve all line breaks, punctuation, and spacing** as much as possible, and make **no additional edits** outside of these replacements.

3. You should only output the rewritten content.

---
Here is the input:

{input}"""

    return prompt


def verify(gold, pred):
    prompt = f"""You are a fair evaluator.
You will be given a groundtruth and an answer from a model.
If the answer aligns with the groundtruth, output "Yes". Otherwise, output "No". 
Your output should only be "Yes" or "No".

```groundtruth
{gold}
```

```answer
{pred}
```"""

    return prompt