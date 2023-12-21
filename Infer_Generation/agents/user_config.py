from agents.dummy_agent import DummyResponseAgent
# from agents.bart_agent import BARTResponseAgent
from agents.LMEDR_agent import LMEDResponseAgent
from agents.prompt_agent import PromptAgent
# from agents.LMEDR_prompt_agent import LMEDPromptAgent

UserAgent = LMEDResponseAgent
# UserAgent = LMEDPromptAgent
# UserAgent = PromptAgent