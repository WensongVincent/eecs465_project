import numpy as np
import numpy.linalg as la
import re

class Particle():
    def __init__(self) -> None:
        self.x = 0.0
        self.y = 0.0
        self.z = 0.0
        self.w = 0.0
        
    
    
class ParticleFilter():
    def __init__(self) -> None:
        self.particles = []
        self.num_particles = len(self.particles)

    
    def action_model():
        pass
    
    def sensor_model():
        pass
    
    def low_var_resample():
        pass
    
def get_action():
    pass

def get_sensor():
    pass
    
def main():
    path = []
    pattern = r'\[((?:\n|.)*?)\]'
    with open('path_maze.txt', 'r') as file:
        content = file.read()
        matches = re.findall(pattern, content)
        for match in matches:
            nums_in_match = re.findall(r'(\d+\.?\d*)', match)
            path.extend(float(num) for num in nums_in_match)
    path = np.array(path).reshape(3,-1)

    # import pdb; pdb.set_trace()
    

    t = 0
    pf = ParticleFilter()
    while(True):
        t += 1
        u_t = get_action()
        

if __name__ == '__main__':
    main()
    
    