from ..game import Game, GameState
import numpy as np


class AttritionWar(Game):
    def __init__(self, resources, tile_rewards, starting_position):
        '''assert checks: making sure that inputs make sense. 
        For example, one should not have a negative amount of resources.'''
        assert resources > 0
        assert len(tile_rewards) > 3
        assert 0 < starting_position < len(tile_rewards) - 1
        
        self.resources = resources
        self.tile_rewards = tile_rewards
        self.starting_position = starting_position
        
        
    def players(self):
        return 2
    
    def initial_state(self):
        return AttritionWarState(None, None, game=self)
    
def determine_winner(resources0, resources1):
    if resources0 > resources1:
        return 0
    if resources0 < resources1:
        return 1
    return -1
    
class AttritionWarState(GameState):
    pass
    
    def resources(self, player):
        return self.game.resources - sum(self.history[player::2])
    
    def player(self):
        if len(self.history) % 2 == 1:
            return 1
        if self.resources(0) == 0:
            return self.P_TERMINAL
        if self.resources(1) == 0:
            return self.P_TERMINAL
        p = self.position()
        if p < 0 or p >= len(self.game.tile_rewards):
            return self.P_TERMINAL
        return 0
    
    #len(self.history) % 2 #this returns either 0 or 1 to tell us which player is playing
    
    
    def actions(self):
        return list(range(1, self.resources(len(self.history) % 2) + 1))    
    
    def position(self):
        position = self.game.starting_position
        for r1, r2 in zip(self.history[::2], self.history[1::2]):
            if r1 > r2:
                position += 1
            elif r2 > r1:
                position -= 1
        return position
        
        
    def values(self):
        p = self.position()
        r1, r2 = self.resources(0), self.resources(1)
        
        if r1 == 0 and r2 > 0:
            p -= r2
            p = max(p,-1)
            
        elif r2 == 0 and r1 > 0:
            p += r1
            p = min(p,len(self.game.tile_rewards)+1)
            
        elif r1 == 0 and r2 == 0:
            p = p
            
        if p < 0:
            return(-1,1)
        elif p >= len(self.game.tile_rewards):
            return(1,-1)
        
        else:
            r = self.game.tile_rewards[p]
        
        return(r, -r)
    
    def player_information(self, player):
        if player == 0: 
            return tuple(self.history)
        else:
            return tuple(self.history[:-1])
        
class WarValueStore:
    def __init__(self, game):
        self.mean_val = 1
        print(self.mean_val)
        self.values = np.zeros(3) + self.mean_val

    def features(self, state):
        "Return vector who won each card: player0=1, player1=-1, tie=0" 
        features = np.zeros_like(self.values)
        if state.position() > 2:
            features[state.position()-3] = 1
        if state.position() < 2:
            features[1 - state.position()] = -1
        return features

    def get_values(self, state):
        val = self.features(state).dot(self.values)
        return np.array((val, -val))
        
    def update_values(self, state, gradient):
        assert gradient.shape == (2,)
        f = self.features(state)
        self.values += f * (gradient[0] - gradient[1])
        # renormalize to the mean
        self.values += self.mean_val - np.mean(self.values)
        #self.values *= self.mean_val / np.mean(self.values)
        #v = (self.values[0] + self.values[6]) / 2
        #self.values[0] = v - 1
        #self.values[6] = v + 1        
