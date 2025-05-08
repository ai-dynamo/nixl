
import argparse
from commands import plan, profile, kvcache

available_commands = [
    plan.Command(),
    kvcache.Command(),
    profile.Command()
]