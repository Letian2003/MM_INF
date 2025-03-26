def score_prompt(prompt, response):
    query = f"""你现在是一位提示工程专家，请帮我评估下面这对 prompt 和 response 是否匹配。
Prompt:
{prompt}
Response:
{response}

请基于以下几点进行分析：
1. Prompt 是否足够清晰、明确，能否有效引导模型输出预期的 response？
2. Prompt 中是否存在歧义或描述不足的地方？
3. Response 是否符合 prompt 所表达的意图？是否完整地回答了 prompt 中的问题？
4. 如果Prompt中要求了输出格式，严格检查Response是否遵循了prompt要求的格式？

请详细说明你的判断过程，并在回复最后给出一个1-10分之间的综合评分，输出格式要求如【5分】。注意不要用markdown格式输出。"""
    return query

def feedback_prompt(prompt, response):
    query = f"""你现在是一位提示工程专家，你会看到一对图片的问答对，请你针对prompt提出改进建议，使得prompt和response更加匹配。

Prompt:
{prompt}
Response:
{response}

注意：你的修改建议不应当使prompt偏离原意，你可以使prompt更加清晰、明确，也可以修改prompt的表述方式，使prompt更加符合response的意图。但是请只是修改prompt的表述方式，不要引入任何response中的信息。

请你提出prompt的改进建议。注意不要用markdown格式输出。仅提供改进prompt的策略、解释和方法。不要提出prompt的新版本。"""
    return query

def optimize_prompt(prompt, feedback):
    query = f"""你现在是一位提示工程专家，你会看到针对下面的 prompt 的一段改进建议，请你根据这个改进建议对prompt进行改进。

Prompt:
{prompt}
改进建议:
{feedback}

请你根据这个改进建议，提出prompt的改进版本，不要输出其他内容。你的输出应该是纯文本，不要用markdown格式或代码块输出。"""
    return query

def feedback_prompt_product(prompt, response):
    query = f"""你现在是一位提示工程专家，你会看到一对图片的问答对，请你针对prompt提出改进建议，使得prompt和response更加匹配。

Prompt:
{prompt}
Response:
{response}

注意：你的修改建议不应当使prompt偏离原意，你可以使prompt更加清晰、明确，从而使之与response契合度更高。但是请只是修改prompt的表述方式，不要引入任何response中的信息，也不要直接提示response所需内容。

请你提出prompt的改进建议。注意不要用markdown格式输出。仅提供改进prompt的策略、解释和方法。不要提出prompt的新版本。"""
    return query