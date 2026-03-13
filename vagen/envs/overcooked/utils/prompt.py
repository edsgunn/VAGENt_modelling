def system_prompt():
    """Return the system prompt for Overcooked agent"""
    return """You are an agent playing Overcooked.
Overcooked Quick Guide
Goal: Cook and deliver 3-onion soups to the serving area.
Rules:
1. Grab 3 onions and put them in a pot.
2. Wait for the pot to cook.
3. Grab a dish and interact with the cooked pot to get the soup.
4. Deliver the soup to the serving counter.
5. Work together with your teammate.
Actions you can take: Left, Down, Right, Up, Stay, Interact."""

def init_observation_template(img_str):
    """Template for initial observation"""
    return f"""[Initial Observation]:
{img_str}
Decide your next action."""

def action_template(valid_action, img_str):
    """Template for action feedback"""
    return f"""After your answer, the extracted valid action is {valid_action}.
After that, the observation is:
{img_str}
Decide your next action."""


def format_prompt(max_actions_per_step, action_sep, add_example=True, prompt_format="free_think"):
    """Generate format prompt based on the specified format"""
    if prompt_format == "free_think":
        return free_think_format_prompt(max_actions_per_step, action_sep, add_example)
    elif prompt_format == "wm":
        return wm_format_prompt(max_actions_per_step, action_sep, add_example)
    elif prompt_format == "free_wm":
        return free_wm_format_prompt(max_actions_per_step, action_sep, add_example)
    else:
        raise ValueError(f"Unknown prompt format: {prompt_format}")

def free_think_format_prompt(max_actions_per_step, action_sep, add_example=True):
    """Generate format prompt for free_think format"""
    base_prompt = f"""You can take up to {max_actions_per_step} action(s) at a time.
You should first give your reasoning, and then your answer.
Your response should be in the format of:
<think>...</think><answer>...</answer>"""
    
    if add_example:
        examples = f"""
Example 1:
<think>There is an onion to my right. I need to pick it up to start cooking soup.</think>
<answer>Right</answer>

Example 2:
<think>I am facing the onion dispenser. I should interact to grab an onion.</think>
<answer>Interact</answer>

Example 3:
<think>The pot is cooking. I should wait for it to finish.</think>
<answer>Stay</answer>
"""
        return base_prompt + "\n" + examples

    return base_prompt



def wm_format_prompt(max_actions_per_step, action_sep, add_example=True):
    """Generate format prompt for wm_new format"""
    base_prompt = f"""You can take up to {max_actions_per_step} action(s) at a time.
Your response must be in the format of:
<observation>...</observation><think>...</think><answer>...</answer><prediction>...</prediction>.

Rules for <observation> and <prediction>:
- You must describe the state of the kitchen, including ingredients, pots, dishes, and teammate.

Rules for <answer>:
- Output 1 action.
- Valid actions are: Up, Down, Left, Right, Stay, Interact.
"""

    if add_example:
        examples = f"""
Example 1:
<observation>There is an onion to my right.</observation>
<think>I should move right to grab it.</think>
<answer>Right</answer>
<prediction>I will be next to the onion dispenser.</prediction>
"""
        return base_prompt + "\n" + examples

    return base_prompt


def free_wm_format_prompt(max_actions_per_step, action_sep, add_example=True):
    """Generate format prompt for free_wm format: observation, answer, prediction with free reasoning between tags."""
    base_prompt = f"""You can take up to {max_actions_per_step} action(s) at a time.
Your response must be in the format of:
<observation>...</observation> your reasoning <answer>...</answer> your reasoning <prediction>...</prediction>.

You may include free-form reasoning text between the tags.

Rules for <answer>:
- Valid actions are: Up, Down, Left, Right, Stay, Interact.
"""
    return base_prompt
