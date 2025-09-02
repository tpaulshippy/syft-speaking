"""Bot Configuration Management.

This module handles loading and managing different bot personalities from text files.
It provides functionality to:
- Load system prompts from personality files
- List available bot personalities  
- Parse command line arguments for bot selection
- Validate bot configurations
"""

import argparse
import os
import sys
from typing import Optional

from loguru import logger


def load_bot_prompt(bot_name: str) -> str:
    """Load system prompt from bot personality file.
    
    Args:
        bot_name: Name of the bot (without .txt extension)
        
    Returns:
        System prompt content as string
        
    Raises:
        FileNotFoundError: If the bot file doesn't exist
        ValueError: If the bot file is empty
    """
    bot_file_path = os.path.join("bots", f"{bot_name}.txt")
    
    if not os.path.exists(bot_file_path):
        available_bots = []
        if os.path.exists("bots"):
            available_bots = [f.replace(".txt", "") for f in os.listdir("bots") if f.endswith(".txt")]
        
        error_msg = f"Bot file '{bot_file_path}' not found."
        if available_bots:
            error_msg += f" Available bots: {', '.join(available_bots)}"
        raise FileNotFoundError(error_msg)
    
    with open(bot_file_path, "r", encoding="utf-8") as f:
        content = f.read().strip()
    
    if not content:
        raise ValueError(f"Bot file '{bot_file_path}' is empty")
    
    return content


def list_available_bots() -> list[str]:
    """List all available bot personalities.
    
    Returns:
        List of bot names (without .txt extension)
    """
    if not os.path.exists("bots"):
        return []
    
    return [f.replace(".txt", "") for f in os.listdir("bots") if f.endswith(".txt")]


def parse_bot_args() -> tuple[Optional[str], list[str]]:
    """Parse command line arguments for bot configuration.
    
    Returns:
        Tuple of (selected_bot_name, remaining_args)
        selected_bot_name will be None if no bot was specified
    """
    parser = argparse.ArgumentParser(description="Run the bot with configurable personality")
    parser.add_argument(
        "--bot", 
        "-b", 
        dest="bot_name",
        help="Name of the bot personality to use (without .txt extension)"
    )
    parser.add_argument(
        "--list-bots",
        action="store_true",
        help="List all available bot personalities"
    )
    
    args, unknown_args = parser.parse_known_args()
    
    # Handle list bots command
    if args.list_bots:
        available_bots = list_available_bots()
        if available_bots:
            print("Available bot personalities:")
            for bot_name in sorted(available_bots):
                print(f"  - {bot_name}")
        else:
            print("No bot personalities found in the 'bots' directory.")
            print("Create .txt files in the 'bots' directory to add personalities.")
        sys.exit(0)
    
    # Validate bot name if provided
    if args.bot_name:
        try:
            load_bot_prompt(args.bot_name)
            logger.info(f"Bot personality '{args.bot_name}' validated successfully")
        except (FileNotFoundError, ValueError) as e:
            logger.error(f"Error validating bot personality: {e}")
            available_bots = list_available_bots()
            if available_bots:
                logger.info(f"Available bots: {', '.join(sorted(available_bots))}")
            sys.exit(1)
    
    return args.bot_name, unknown_args


def get_system_prompt(bot_name: Optional[str]) -> str:
    """Get system prompt for the specified bot or default.
    
    Args:
        bot_name: Name of the bot to load, or None for default
        
    Returns:
        System prompt string
    """
    if bot_name:
        try:
            prompt = load_bot_prompt(bot_name)
            logger.info(f"Loaded bot personality: {bot_name}")
            return prompt
        except (FileNotFoundError, ValueError) as e:
            logger.error(f"Error loading bot personality: {e}")
            sys.exit(1)
    
    # Default system prompt
    return "You are Chatbot, a friendly, helpful robot. Your goal is to demonstrate your capabilities in a succinct way. Your output will be converted to audio so don't include special characters in your answers. Respond to what the user said in a creative and helpful way, but keep your responses brief. Start by introducing yourself."
