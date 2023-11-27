from pattern import *
from generator import *
# Checker=Checker(250,25)
# Checker.draw()
# Checker.show()

# #Circle=Circle(50,10,(5,5))
# #Circle.draw()
# #Circle.show()

# Spectrum=Spectrum(100)
# Spectrum.draw()
# Spectrum.show()
gen=ImageGenerator('D:/MSc Data Science/Sem 2/Deep Learning/exercise0_material/exercise0_material/src_to_implement/data/exercise_data/', 'D:/MSc Data Science/Sem 2/Deep Learning/exercise0_material/exercise0_material/src_to_implement/data/Labels.json', 12, 
            [32, 32, 3], rotation=False, mirroring=False, shuffle=False)
gen.show()
