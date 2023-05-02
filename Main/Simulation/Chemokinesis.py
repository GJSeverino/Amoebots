
from cc3d import CompuCellSetup
        

from ChemokinesisSteppables import ChemokinesisSteppable

CompuCellSetup.register_steppable(steppable=ChemokinesisSteppable(frequency=1))


CompuCellSetup.run()
