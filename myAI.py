from poker.action import Action
from poker.game_state import GameState
from poker.player import Blind
import numpy as np

# TABLE[spot, hand, action] as one 3D array
TABLE = np.zeros((100, 169, 1000), dtype=np.float32)


def myAI(state: GameState) -> Action:
    me = state.my_player
    to_call = max(player.chips_bet for player in state.players) - me.chips_bet
    left_after_calling = me.chips_in_stack - to_call
    blind = state.big_blind
    
    if me.chips_in_stack < 10 * blind:
        return Action.raise_by(left_after_calling)
    
    return defaultAI(state)

    #

def defaultAI(state: GameState) -> Action: # This ai runs if myAI raises an exception, returns an invalid action or takes too long
    return Action.check() if not state.my_player.chips_in_stack or state.my_player.chips_bet == max(player.chips_bet for player in state.players) else Action.fold()