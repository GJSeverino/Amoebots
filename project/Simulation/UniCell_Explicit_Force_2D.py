
from cc3d import CompuCellSetup
        

from UniCell_Explicit_Force_2DSteppables import UniCell_Explicit_Force_2DSteppable

CompuCellSetup.register_steppable(steppable=UniCell_Explicit_Force_2DSteppable(frequency=1))



        
from UniCell_Explicit_Force_2DSteppables import CalculationsSteppable
#CompuCellSetup.register_steppable(steppable=CalculationsSteppable(frequency=100))


        
from UniCell_Explicit_Force_2DSteppables import PersistentNeighborsSteppable
#CompuCellSetup.register_steppable(steppable=PersistentNeighborsSteppable(frequency=1))


        
from UniCell_Explicit_Force_2DSteppables import CollectivityCalcSteppable
#CompuCellSetup.register_steppable(steppable=CollectivityCalcSteppable(frequency=1))


        
from UniCell_Explicit_Force_2DSteppables import Position_OutputSteppable
#CompuCellSetup.register_steppable(steppable=Position_OutputSteppable(frequency=1))


        
from UniCell_Explicit_Force_2DSteppables import Interface_With_MediumSteppable
#CompuCellSetup.register_steppable(steppable=Interface_With_MediumSteppable(frequency=1))


        
from UniCell_Explicit_Force_2DSteppables import ClusterCountSteppable
#CompuCellSetup.register_steppable(steppable=ClusterCountSteppable(frequency=1))

CompuCellSetup.run()
