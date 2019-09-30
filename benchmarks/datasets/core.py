class MetaDataset(type):
    def __repr__(cls):
        return cls.name


class Dataset(object, metaclass=MetaDataset):

    @classmethod
    def is_available(cls):
        return cls.dirpth.is_dir()
