import numpy as np
import asyncio
from typing import Any, Dict, Tuple, List
from dataclasses import dataclass

from overcooked_ai_py.mdp.actions import Action
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv as CoreOvercookedEnv, Overcooked
from overcooked_ai_py.mdp.pixel_env import PixelObservationWrapper
from overcooked_ai_py.agents.agent import RandomAgent

from ..gym_image_env import GymImageEnv
from .utils.prompt import format_prompt, init_observation_template, action_template, system_prompt
from .utils.utils import parse_response, numpy_to_pil

@dataclass
class OvercookedEnvConfig:
    layout_name: str = "cramped_room"
    horizon: int = 400
    render_mode: str = "vision"
    max_actions_per_step: int = 1
    action_sep: str = ","
    image_placeholder: str = "<image>"
    use_example_in_sys_prompt: bool = True
    prompt_format: str = "free_think"
    format_reward: float = 0.0
    success_reward: float = 1.0


class OvercookedVagenEnv(GymImageEnv):
    ACTION_LOOKUP = {
        "up": 0,
        "down": 1,
        "right": 2,
        "left": 3,
        "stay": 4,
        "interact": 5,
    }

    def __init__(self, env_config: Dict[str, Any]):
        super().__init__(env_config)
        self.config = OvercookedEnvConfig(**env_config)
        
        # Setting up native Overcooked environment
        mdp = OvercookedGridworld.from_layout_name(self.config.layout_name)
        base_env = CoreOvercookedEnv.from_mdp(mdp, horizon=self.config.horizon)
        # Using lossless encoding as featurize_fn
        gym_env = Overcooked(base_env, featurize_fn=base_env.featurize_state_mdp)
        self.pixel_env = PixelObservationWrapper(gym_env)
        
        self.counterpart_agent = RandomAgent(all_actions=True)
        
        self.total_reward = 0.0
        self.valid_actions = []

    async def close(self) -> None:
        pass

    async def reset(self, seed: int) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        self.total_reward = 0.0
        self.valid_actions = []
        
        # Reset the environment; this randomizes which agent is which (ego vs counterpart)
        # We need to catch output regardless if it is gym 0.21 or 0.26
        rets = await asyncio.to_thread(self.pixel_env.reset, seed=seed)
        
        self.counterpart_agent.reset()
        
        obs = await self._render_async(init_obs=True)
        return obs, {}

    async def system_prompt(self) -> Dict[str, Any]:
        return {
            "obs_str": self.get_system_prompt(),
        }

    async def step(self, action_str: str) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        parsed = parse_response(
            response=action_str,
            action_sep=self.config.action_sep,
            max_actions=self.config.max_actions_per_step,
            prompt_format=self.config.prompt_format,
        )
        reward = 0.0
        done = False
        info: Dict[str, Any] = {}
        self.valid_actions = []
        info.update(parsed)
        action_list: List[str] = parsed.get("actions", [])

        metrics = {
            "turn_metrics": {
                "action_is_valid": len(action_list) > 0 and parsed.get("format_correct", False),
            },
            "traj_metrics": {
                "success": False,
            },
        }

        action_int = Action.ACTION_TO_INDEX[Action.STAY]
        for action in action_list:
            if action in self.ACTION_LOOKUP:
                action_int = self.ACTION_LOOKUP[action]
                self.valid_actions.append(action)
                break 
            else:
                metrics["turn_metrics"]["action_is_valid"] = False
                break

        # Get counterpart agent action
        try:
           state = getattr(self.pixel_env.env, "base_env").state
        except:
           state = getattr(self.pixel_env.unwrapped, "base_env").state

        counterpart_action, _ = self.counterpart_agent.action(state)
        counterpart_action_idx = Action.ACTION_TO_INDEX[counterpart_action]

        joint_action_indices = (action_int, counterpart_action_idx)

        step_rets = await asyncio.to_thread(self.pixel_env.step, joint_action_indices)
        if len(step_rets) == 5:
            pixel_obs, step_reward, terminated, truncated, env_info = step_rets
            done = terminated or truncated
        elif len(step_rets) == 4:
            pixel_obs, step_reward, done, env_info = step_rets
        
        reward += float(step_reward)
        self.total_reward += reward

        if self.valid_actions:
            reward += self.config.format_reward
            
        metrics["traj_metrics"]["success"] = done and (self.total_reward > 0)

        info["metrics"] = metrics
        info["success"] = metrics["traj_metrics"]["success"]
        info["overcooked_info"] = env_info

        obs = await self._render_async(init_obs=False)
        return obs, reward, done, info

    def get_system_prompt(self) -> str:
        format_prompt_str = format_prompt(
            max_actions_per_step=self.config.max_actions_per_step,
            action_sep=self.config.action_sep,
            add_example=self.config.use_example_in_sys_prompt,
            prompt_format=self.config.prompt_format,
        )
        return system_prompt() + "\n" + format_prompt_str

    async def _render_async(self, init_obs: bool) -> Dict[str, Any]:
        img_str = self.config.image_placeholder
        
        try:
           pixel_array = await asyncio.to_thread(self.pixel_env.observation, None)
           multi_modal_input = {
               self.config.image_placeholder: [numpy_to_pil(pixel_array)]
           }
        except Exception as e:
           multi_modal_input = {}
           img_str = f"Error rendering: {e}"

        if init_obs:
            obs_str = init_observation_template(img_str) + "\n"
        else:
            obs_str = action_template(self.valid_actions, img_str) + "\n"

        obs: Dict[str, Any] = {"obs_str": obs_str}
        if multi_modal_input:
            obs["multi_modal_input"] = multi_modal_input
        return obs
