from migen.fhdl.std import *
from migen.bank.description import CSRStorage, CSRStatus, AutoCSR


class Filter(Module, AutoCSR):
    def __init__(self, width):
        self.x = Signal((width, True))
        self.y = Signal((width, True))

        self.hold = Signal()
        self.clear = Signal()
        self.error = Signal()

        self.r_y = CSRStatus(width)

        ###

        self.comb += self.r_y.status.eq(self.y)
