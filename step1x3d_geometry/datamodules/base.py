import pytorch_lightning as pl


class BaseDataModule(pl.LightningDataModule):
    """Stub base class for data modules."""
    def __init__(self, **kwargs):
        super().__init__() 