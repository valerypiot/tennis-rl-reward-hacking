# Actions robot can take
UP, DOWN, LEFT, RIGHT, PICK = 0, 1, 2, 3, 4
ACTION_NAMES = ["up", "down", "left", "right", "pick"]

class TennisCourtEnv: 
    # env: robot on tennis court (grid 5x5)
    def __init__(self, reward_mode="collected", size=5, max_steps=30):
        self.reward_mode = reward_mode
        self.size = size
        self.max_steps = max_steps
        self.start_balls = [(1, 3), (4, 1), (4, 4)] # ball position in grid
    
    # reset to startposition
    def reset(self):
        self.robot = (0, 0)
        self.balls = list(self.start_balls)
        self.steps = 0

        return self._observation(), {}
    
    # state where robot is and reamining balls
    def _observation(self): 
        remaining = tuple(ball in self.balls for ball in self.start_balls)
        return (self.robot, remaining)

    # check whether ball is front/back/left/right
    def _balls_in_view(self):
        r, c = self.robot
        return sum(1 for (br, bc) in self.balls if abs(br - r) + abs(bc -c) <= 1)
    
    # 
    def step(self, action): 
        r, c = self.robot
        picked = 0

        if action == UP:
            r = max(r - 1, 0)
        elif action == DOWN:
            r = min(r + 1, self.size - 1)
        elif action == LEFT:
            c = max(c - 1, 0)
        elif action == RIGHT:
            c = min(c + 1, self.size - 1)
        elif action == PICK and self.robot in self.balls:
            self.balls.remove(self.robot)
            picked = 1
        self.robot = (r, c)
        self.steps += 1

        # reward "function"
        if self.reward_mode == "seen":
            reward = 1.0 if self._balls_in_view() > 0 else 0.0 
        else: # collected
            reward = picked - 0.01 # +1 per ball, tiny cost per step 0.01
        
        terminated = len(self.balls) == 0 # no other balls -> task finished 
        truncated = self.steps >= self.max_steps # out of time

        info = {"picked": picked}
        return self._observation(), reward, terminated, truncated, info

    # prints the grid
    def render(self):
        for r in range(self.size):
            row = ""
            for c in range(self.size):
                if (r, c) == self.robot:
                    row += "R "
                elif (r, c) in self.balls:
                    row += "o "
                else:
                    row += ". "

            print(row)
        print()