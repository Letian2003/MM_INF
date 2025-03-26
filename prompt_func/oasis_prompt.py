import json
import os



def extract_query_prompt_new(text):
    prompt = f"""You will be given a text regarding an image. Your task is to determine whether the text contains any instructions. If it contains instructions, extract one instruction. You should extract the instruction, as well as any relevant contextual information that aids in understanding the instruction. 

NOTE:
1. The instruction may take the form of an interrogative sentence, an imperative sentence, a multiple-choice question, or other similar structures. Please identify carefully!
2. Extract ONLY the original instruction, WITHOUT extracting any answers.
3. If the instruction is a multiple-choice question, you should extract the question and the options.
4. If there are multiple instructions, you should extract only one instruction.

You MUST answer with the following format:
Instruction: [an instruction]

If it doesn't contain any instructions, output 'NO_INST'.

----- Example 1:
Text: 
1. Answer the following questions based on the text:\n\n    a. Who increased the number of insurgents in the valley? \n\n    b. When did Singh come to power? What act did he implement?\n\n    c. What is the purpose of the SC-ST Act?

Answer: 
Instruction: Who increased the number of insurgents in the valley?

----- Example 2:
Text: 
There is an animal behind the fence who is holding a bottle.

Answer: 
NO_INST

----- Example 3:
Text: 
In this problem, we have an elephant image that includes several lines and curves.\n\nWe want to transform this image into another animal using the least number of changes.\n\nPlease provide some suggestions on how to achieve this transformation with minimal effort.

Answer: 
Instruction: We want to transform this image into another animal using the least number of changes.\n\nPlease provide some suggestions on how to achieve this transformation with minimal effort.

----- Example 4:
Text: 
Could you please summarize the mission statement of the company and the benefits it promises to its customers in 30 seconds or less?\n The mission of our company is to provide innovative tech solutions for all your needs. We prioritize security and privacy for our users and are committed to excellence. With us by their side, customers can expect a simplified tech journey that feels more defined.

Answer: 
Instruction: Could you please summarize the mission statement of the company and the benefits it promises to its customers in 30 seconds or less?

----- Example 5:
Text: 
I would like to make a real estate agency website using HTML, CSS, and JavaScript.

Answer: 
Instruction: I would like to make a real estate agency website using HTML, CSS, and JavaScript.

----- Example 6:
Text: 
The scene has a window on the top left, a fire hydrant on the bottom right, and two signs in the middle right.

Answer: 
NO_INST
----- End of Example

[Begin of Text]
{text}
[End of Text]
    """

    return prompt



def inst_unsolvability_prompt(query):
    prompt = f"""Your task is to evaluate the solvability of a query to an image. The solvability can be quantitatively evaluated on a scale of 1 to 5, based on the presence of sufficient information within the image to formulate a complete answer. 

Here are the criteria:

Score 1 (Very Low Solvability): The image contains minimal or no relevant information related to the question, making it nearly impossible to derive a meaningful answer.

Score 2 (Low Solvability): The image provides some information, but key elements are missing, resulting in significant uncertainty.

Score 3 (Moderate Solvability): The image contains a reasonable amount of information that may lead to an answer, but ambiguities or lack of clarity hinder definitive conclusions.

Score 4 (High Solvability): The image offers substantial information that strongly supports answering the question, with only minor uncertainties remaining.

Score 5 (Very High Solvability): The image is rich in detail and clarity, providing all necessary information to answer the question comprehensively.

Please rate the query on a scale of 1 to 5. You MUST use "[[1]]", "[[2]]", "[[3]]", "[[4]]", "[[5]]" to indicate your evaluation score in the key 'Score'.

[Query]
{query}
    """

    return prompt


def inst_hallucination_prompt(query):
    prompt = f"""Your task is to evaluate whether a query to an image contains hallucination content. The determination of whether a question related to an image contains hallucinations can be assessed on a scale of 1 to 5. This scale evaluates the alignment between the question's content and the actual content of the image, identifying discrepancies that indicate hallucinations.

Here are the criteria:

Score 1 (Severe Hallucination): The question bears little to no relation to the image content, filled with substantial errors or completely unrelated information. The discrepancies are so pronounced that they render the question fundamentally flawed in context to the image.

Score 2 (Significant Hallucination): The question diverges considerably from the image, containing multiple erroneous statements or irrelevant details. The inaccuracies are significant enough that they compromise the integrity of the inquiry.

Score 3 (Moderate Hallucination): The question and image content have notable inconsistencies, with several inaccuracies present. While some relevant information is shared, the question includes errors that could lead to misleading conclusions.

Score 4 (Minor Hallucination): The question is largely consistent with the image, but there are minor discrepancies or inaccuracies that do not significantly alter the overall interpretation. These could include slight misinterpretations of color or detail that do not affect the main subject.

Score 5 (No Hallucination): The question aligns perfectly with the image content, containing no errors or irrelevant information. All aspects of the inquiry are directly supported by clear and accurate details within the image.

Please rate the query on a scale of 1 to 5. You MUST use "[[1]]", "[[2]]", "[[3]]", "[[4]]", "[[5]]" to indicate your evaluation score in the key 'Score'.

[Query]
{query}
    """
    return prompt

def inst_clarity_prompt(query):
    prompt = f"""Your task is to evaluate the clarity of a query to an image. The clarity of a question derived from an image can be evaluated on a scale of 1 to 5, reflecting how precisely the question conveys its intent and whether it allows for a definitive answer.

Here are the criteria:

Score 1 (Very Unclear): The question is exceedingly vague and unclear, with multiple interpretations possible. It fails to convey a coherent intent, resulting in uncertainty and an inability to arrive at a definitive answer.

Score 2 (Unclear): The question is largely ambiguous, making it difficult to discern its exact intent. The vagueness significantly hinders the ability to provide a clear answer, leading to potential misinterpretations and disagreements.

Score 3 (Moderately Clear): The question exhibits noticeable vagueness that may cause some confusion. While there are identifiable elements, the lack of precision can lead to varying interpretations and uncertainty in answering.

Score 4 (Clear): The question is generally clear but may contain minor ambiguities that could lead to slight misinterpretations. However, the overall intent remains understandable, allowing for a reasonably definitive answer.

Score 5 (Very Clear): The question is exceptionally clear, leaving no room for ambiguity. It conveys its intent explicitly, and the required answer is straightforward and unambiguous, making it easy to interpret.

Please rate the query on a scale of 1 to 5. You MUST use "[[1]]", "[[2]]", "[[3]]", "[[4]]", "[[5]]" to indicate your evaluation score in the key 'Score'.

[Query]
{query}
    """
    return prompt

def inst_nonsense_prompt(query):
    prompt = f"""Your task is to evaluate whether a query to an image contains nonsense. The presence of nonsense in a question related to an image can be assessed on a scale of 1 to 5.

Here are the criteria:

Score 1 (Severe Nonsense): The question is completely nonsensical, filled with severe grammatical issues, strange characters, or illogical phrases that render it unintelligible. It fails to convey any meaningful intent.

Score 2 (Significant Nonsense): The question is largely incoherent, containing multiple grammatical errors or strange characters that obstruct its meaning. Understanding the question is challenging and may lead to misinterpretations.

Score 3 (Moderate Nonsense): The question exhibits noticeable issues with clarity, such as awkward constructions or vague expressions. While some meaning is still discernible, these factors may lead to confusion.

Score 4 (Minimal Nonsense): The question is generally clear but may contain minor grammatical errors or awkward phrasing that slightly detract from its coherence. These issues do not significantly impede understanding.

Score 5 (No Nonsense): The question is coherent, grammatically correct, and free from any strange characters or phrases. It conveys its intent clearly and logically, allowing for a straightforward understanding.

Please rate the query on a scale of 1 to 5. You MUST use "[[1]]", "[[2]]", "[[3]]", "[[4]]", "[[5]]" to indicate your evaluation score in the key 'Score'.

[Query]
{query}
    """
    return prompt