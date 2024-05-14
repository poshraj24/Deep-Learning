from pattern import *

# Checker=Checker(250,25)
# Checker.draw()
# Checker.show()

# Circle=Circle(50,10,(5,5))
# Circle.draw()
# Circle.show()

# Spectrum=Spectrum(100)
# Spectrum.draw()
# Spectrum.show()
from generator import *
gen=ImageGenerator(r'D:/MSc_DS/Sem_3/DL/Exercises/exercise0_material/src_to_implement/data/exercise_data/', r'D:/MSc_DS/Sem_3/DL/Exercises/exercise0_material/src_to_implement/data/Labels.json', 12, 
            [32, 32, 3], rotation=False, mirroring=False, shuffle=False)
gen.show()
