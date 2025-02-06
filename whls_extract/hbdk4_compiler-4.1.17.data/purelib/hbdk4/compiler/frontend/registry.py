class OpType:
    def __init__(
        self, signature: str, namespace: str, version: int, description: str = None
    ):
        self.signature = signature
        self.namespace = namespace
        self.version = version
        self.description = description

    @property
    def key(self) -> str:
        return "{}::{}".format(self.namespace, self.signature)

    def __str__(self) -> str:
        return "{}::{} ver {} \n detail:{}".format(
            self.namespace, self.signature, self.version, self.description
        )


class OpConvertorRegistry:
    _instance = None
    impl = {}

    def __new__(cls):
        if not cls._instance:
            cls._instance = super(OpConvertorRegistry, cls).__new__(cls)
        return cls._instance

    def register(self, cvt):
        assert isinstance(cvt.type, OpType)
        key = cvt.type.key
        if key not in self.impl:
            self.impl[key] = [cvt]
        else:
            for impl in self.impl[key]:
                if impl.type.version == cvt.type.version:
                    raise ValueError(
                        "duplicate convertor of key{} version {}".format(
                            cvt.type.key, cvt.type.version
                        )
                    )
            self.impl[key].append(cvt)

    def find(self, op_type: OpType):
        key = op_type.key
        if key in self.impl:
            entries = sorted(
                self.impl[key], key=lambda x: -x.type.version
            )  # descending order
            for entry in entries:
                if entry.type.version <= op_type.version:
                    return entry
        raise ValueError(
            "no viable conversion from {} of version {} to hbdk. \n Op details: {}".format(
                key, op_type.version, op_type.description
            )
        )
