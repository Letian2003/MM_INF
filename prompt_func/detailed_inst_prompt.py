import json
import os



def Instruction_generate_prompt(text):
    prompt = f"""You will see a detailed description of an image. Based on this description, please write a detailed inquiry prompt. Your prompt should ask about the content and details shown in the image in a way that allows the description to effectively answer it. Note that your prompt should not mention overly specific details of the image and should only ask broad questions.

[Description]
{text}
"""

    return prompt
